struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct BBUniforms {
    bb_min: vec3<f32>,
    _pad0: f32,
    bb_max: vec3<f32>,
    _pad1: f32,
    res_x: u32,
    res_y: u32,
    res_z: u32,
    show_bbox: u32,
    show_query: u32,
    gwn_mode: u32,   // 0 = flat color, 1 = GWN heatmap
    point_size: f32, // billboard half-size in pixels
    _pad2: u32,
    query_color: vec4<f32>,
    bbox_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> bb: BBUniforms;
@group(2) @binding(0) var<storage, read> gwn_values: array<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// ---- Bounding Box Wireframe ----

fn bbox_corner(index: u32) -> vec3<f32> {
    return vec3<f32>(
        select(bb.bb_min.x, bb.bb_max.x, (index & 1u) != 0u),
        select(bb.bb_min.y, bb.bb_max.y, (index & 2u) != 0u),
        select(bb.bb_min.z, bb.bb_max.z, (index & 4u) != 0u),
    );
}

const EDGE_INDICES = array<u32, 24>(
    0u, 1u,  1u, 3u,  3u, 2u,  2u, 0u,
    4u, 5u,  5u, 7u,  7u, 6u,  6u, 4u,
    0u, 4u,  1u, 5u,  2u, 6u,  3u, 7u,
);

@vertex
fn vs_bbox(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;
    let corner = bbox_corner(EDGE_INDICES[idx]);
    out.position = camera.proj * camera.view * vec4<f32>(corner, 1.0);
    out.color = bb.bbox_color;
    return out;
}

// ---- Colormap: maps [0,1] -> color ----
// Cool-warm diverging: blue(0) -> white(0.5) -> red(1)
fn gwn_colormap(t: f32) -> vec3<f32> {
    let s = clamp(t, 0.0, 1.0);
    // blue -> white -> red
    let r = clamp(2.0 * s, 0.0, 1.0);
    let b = clamp(2.0 * (1.0 - s), 0.0, 1.0);
    let g = 1.0 - abs(2.0 * s - 1.0);
    return vec3<f32>(r, g, b);
}

// ---- Query Points Grid (billboard quads) ----
// 6 vertices per point (2 triangles). vertex_index / 6 = point index.

const QUAD_OFFSETS = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
);

@vertex
fn vs_query(@builtin(vertex_index) vert_idx: u32) -> VertexOutput {
    var out: VertexOutput;

    let pt_idx  = vert_idx / 6u;
    let corner  = vert_idx % 6u;

    let total_per_layer = bb.res_x * bb.res_y;
    let iz = pt_idx / total_per_layer;
    let rem = pt_idx % total_per_layer;
    let iy = rem / bb.res_x;
    let ix = rem % bb.res_x;

    let t = vec3<f32>(
        (f32(ix) + 0.5) / f32(bb.res_x),
        (f32(iy) + 0.5) / f32(bb.res_y),
        (f32(iz) + 0.5) / f32(bb.res_z),
    );
    let world_pos = bb.bb_min + t * (bb.bb_max - bb.bb_min);

    // Project center to clip space, then offset in NDC pixels
    let clip = camera.proj * camera.view * vec4<f32>(world_pos, 1.0);
    let offset = QUAD_OFFSETS[corner] * bb.point_size / camera.viewport;
    out.position = vec4<f32>(clip.xy + offset * clip.w, clip.zw);

    if (bb.gwn_mode == 1u && pt_idx < arrayLength(&gwn_values)) {
        out.color = vec4<f32>(gwn_colormap(gwn_values[pt_idx]), 1.0);
    } else {
        out.color = bb.query_color;
    }

    return out;
}

// ---- Orient Camera Position Visualization (billboard quads) ----
// Reads camera positions from orient camera buffer.
// Each camera is 20 f32s: mat4x4 (16) + pos (4). Position at offset i*20 + 16.

@group(1) @binding(1) var<storage, read> orient_cameras: array<f32>;

@vertex
fn vs_cameras(@builtin(vertex_index) vert_idx: u32) -> VertexOutput {
    var out: VertexOutput;

    let cam_idx = vert_idx / 6u;
    let corner  = vert_idx % 6u;

    let base = cam_idx * 20u + 16u;
    let world_pos = vec3<f32>(
        orient_cameras[base],
        orient_cameras[base + 1u],
        orient_cameras[base + 2u],
    );

    let clip = camera.proj * camera.view * vec4<f32>(world_pos, 1.0);
    let cam_point_size = 6.0; // pixels
    let offset = QUAD_OFFSETS[corner] * cam_point_size / camera.viewport;
    out.position = vec4<f32>(clip.xy + offset * clip.w, clip.zw);
    out.color = vec4<f32>(1.0, 1.0, 0.0, 1.0); // yellow
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}