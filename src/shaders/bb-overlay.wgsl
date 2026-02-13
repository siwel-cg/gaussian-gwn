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
    resolution: u32,
    show_bbox: u32,
    show_query: u32,
    _pad2: u32,
    query_color: vec4<f32>,
    bbox_color: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<uniform> bb: BBUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// ---- Bounding Box Wireframe ----
// 24 vertices = 12 edges * 2 endpoints
// Using line-list topology

fn bbox_corner(index: u32) -> vec3<f32> {
    // 8 corners of the box, indexed by 3 bits
    return vec3<f32>(
        select(bb.bb_min.x, bb.bb_max.x, (index & 1u) != 0u),
        select(bb.bb_min.y, bb.bb_max.y, (index & 2u) != 0u),
        select(bb.bb_min.z, bb.bb_max.z, (index & 4u) != 0u),
    );
}

// 12 edges as pairs of corner indices
const EDGE_INDICES = array<u32, 24>(
    // bottom face (y=0)
    0u, 1u,  1u, 3u,  3u, 2u,  2u, 0u,
    // top face (y=1)
    4u, 5u,  5u, 7u,  7u, 6u,  6u, 4u,
    // vertical edges
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

// ---- Query Points Grid ----

@vertex
fn vs_query(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;

    let res = bb.resolution;
    let total_per_layer = res * res;

    let iz = idx / total_per_layer;
    let remainder = idx % total_per_layer;
    let iy = remainder / res;
    let ix = remainder % res;

    // Normalize to [0, 1] with half-cell offset for centering
    let t = vec3<f32>(
        (f32(ix) + 0.5) / f32(res),
        (f32(iy) + 0.5) / f32(res),
        (f32(iz) + 0.5) / f32(res),
    );

    let pos = bb.bb_min + t * (bb.bb_max - bb.bb_min);

    out.position = camera.proj * camera.view * vec4<f32>(pos, 1.0);
    out.color = bb.query_color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}