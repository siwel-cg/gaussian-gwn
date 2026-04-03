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

// // ---- Colormap: maps [0,1] -> color ----
// // Cool-warm diverging: blue(0) -> white(0.5) -> red(1)
// fn gwn_colormap(t: f32) -> vec3<f32> {
//     let s = clamp(t, 0.0, 1.0);
//     // blue -> white -> red
//     let r = clamp(2.0 * s, 0.0, 1.0);
//     let b = clamp(2.0 * (1.0 - s), 0.0, 1.0);
//     let g = 1.0 - abs(2.0 * s - 1.0);
//     return vec3<f32>(r, g, b);
// }


// ---- Colormap: maps [0, 2] -> color ----
// 5-stop gradient:
//   0.0  deep blue    (outside, GWN ≈ 0)
//   0.5  cyan/teal    (approaching surface)
//   1.0  bright green  (inside, GWN ≈ 1 — the "correct" value)
//   1.5  yellow/orange (overshoot)
//   2.0  hot red       (anomalous / double-winding)
fn gwn_colormap(value: f32) -> vec3<f32> {
    let t = clamp(value, 0.0, 2.0);

    // 5 color stops
    let c0 = vec3<f32>(0.05, 0.05, 0.40);  // deep blue
    let c1 = vec3<f32>(0.10, 0.55, 0.65);  // teal
    let c2 = vec3<f32>(0.30, 0.85, 0.20);  // bright green
    let c3 = vec3<f32>(0.95, 0.75, 0.10);  // amber
    let c4 = vec3<f32>(0.85, 0.10, 0.10);  // hot red

    // piecewise linear interpolation across 4 segments over [0, 2]
    // each segment spans 0.5 units
    if (t < 0.5) {
        return mix(c0, c1, t * 2.0);           // [0.0, 0.5)
    } else if (t < 1.0) {
        return mix(c1, c2, (t - 0.5) * 2.0);   // [0.5, 1.0)
    } else if (t < 1.5) {
        return mix(c2, c3, (t - 1.0) * 2.0);   // [1.0, 1.5)
    } else {
        return mix(c3, c4, (t - 1.5) * 2.0);   // [1.5, 2.0]
    }
}

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

// ---- Normal Direction Visualization (line segments) ----
// Reads precomputed splat data (position + normal) and draws a line per splat.

struct PrecomputedSplat {
    nx: f32, ny: f32, nz: f32,
    area: f32,
    px: f32, py: f32, pz: f32,
    _pad: f32,
};

struct NormalVisParams {
    num_splats: u32,
    normal_length: f32,
    _pad0: u32,
    _pad1: u32,
};

@group(3) @binding(0) var<storage, read> precomputed: array<PrecomputedSplat>;
@group(3) @binding(1) var<uniform> normal_vis_params: NormalVisParams;

@vertex
fn vs_normals(@builtin(vertex_index) vert_idx: u32) -> VertexOutput {
    var out: VertexOutput;

    let splat_idx = vert_idx / 2u;
    let is_tip    = vert_idx % 2u;  // 0 = base, 1 = tip

    if (splat_idx >= normal_vis_params.num_splats) {
        out.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.color = vec4<f32>(0.0);
        return out;
    }

    let s = precomputed[splat_idx];
    let pos = vec3<f32>(s.px, s.py, s.pz);
    let normal = vec3<f32>(s.nx, s.ny, s.nz);

    var world_pos = pos;
    if (is_tip == 1u) {
        world_pos = pos + normal * normal_vis_params.normal_length;
    }

    out.position = camera.proj * camera.view * vec4<f32>(world_pos, 1.0);

    // Color-code by normal direction: RGB = abs(nx, ny, nz)
    let abs_n = abs(normal);
    out.color = vec4<f32>(abs_n.x, abs_n.y, abs_n.z, 1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}