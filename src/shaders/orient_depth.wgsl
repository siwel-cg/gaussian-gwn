// orient_depth.wgsl — Pass 1: splat depth into per-camera atomic depth buffers
//
// For each splat, project its center through every camera's VP matrix.
// Write atomicMin into a flat depth buffer indexed by (camera, pixel).
//
// Depth buffer layout: depth_buf[cam_idx * (res * res) + py * res + px]

override workgroupSize: u32 = 64;

struct PrecomputedSplat {
    nx: f32, ny: f32, nz: f32,
    area: f32,
    px: f32, py: f32, pz: f32,
    _pad: f32,
};

struct OrientCamera {
    viewProj: mat4x4<f32>,  // 16 floats
    pos: vec4<f32>,          // 4 floats
};

struct OrientParams {
    num_cameras: u32,
    depth_res: u32,
    num_splats: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       splats      : array<PrecomputedSplat>;
@group(0) @binding(1) var<storage, read>       cameras     : array<OrientCamera>;
@group(0) @binding(2) var<uniform>             params      : OrientParams;
@group(0) @binding(3) var<storage, read_write> depth_buf   : array<atomic<u32>>;

@compute @workgroup_size(workgroupSize, 1, 1)
fn depth_splat(@builtin(global_invocation_id) gid: vec3<u32>) {
    let splat_idx = gid.x;
    if (splat_idx >= params.num_splats) { return; }

    let s = splats[splat_idx];
    let pos = vec4<f32>(s.px, s.py, s.pz, 1.0);
    let res = params.depth_res;
    let pixels_per_cam = res * res;

    for (var c = 0u; c < params.num_cameras; c++) {
        let cam = cameras[c];
        let clip = cam.viewProj * pos;

        // behind camera
        if (clip.w <= 0.0) { continue; }

        let ndc = clip.xyz / clip.w;

        // frustum cull
        if (ndc.x < -1.0 || ndc.x > 1.0 ||
            ndc.y < -1.0 || ndc.y > 1.0 ||
            ndc.z < 0.0  || ndc.z > 1.0) { continue; }

        // ndc → pixel coords
        let px = u32(clamp((ndc.x * 0.5 + 0.5) * f32(res), 0.0, f32(res - 1u)));
        let py = u32(clamp((1.0 - (ndc.y * 0.5 + 0.5)) * f32(res), 0.0, f32(res - 1u)));

        // encode depth as u32 for atomicMin (ndc.z is [0,1], map to [0, 2^24])
        let depth_uint = u32(clamp(ndc.z, 0.0, 1.0) * 16777215.0);

        let buf_idx = c * pixels_per_cam + py * res + px;
        atomicMin(&depth_buf[buf_idx], depth_uint);
    }
}