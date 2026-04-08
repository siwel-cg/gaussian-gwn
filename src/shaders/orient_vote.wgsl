// orient_vote.wgsl — Pass 2: vote on normal orientation using depth visibility
//
// For each splat, re-project through every camera. If the splat passes the
// depth tolerance test (near-front), accumulate a weighted vote.
// Weight = dot(normal, view_dir) which encodes both sign and view-angle quality.
// After all cameras, flip the normal in precomputed buffer if vote < 0.

override workgroupSize: u32 = 64;

struct PrecomputedSplat {
    nx: f32, ny: f32, nz: f32,
    area: f32,
    px: f32, py: f32, pz: f32,
    _pad: f32,
};

struct OrientCamera {
    viewProj: mat4x4<f32>,
    pos: vec4<f32>,
};

struct OrientParams {
    num_cameras: u32,
    depth_res: u32,
    num_splats: u32,
    depth_tolerance: f32,
};

@group(0) @binding(0) var<storage, read_write> splats    : array<PrecomputedSplat>;
@group(0) @binding(1) var<storage, read>       cameras   : array<OrientCamera>;
@group(0) @binding(2) var<uniform>             params    : OrientParams;
@group(0) @binding(3) var<storage, read>       depth_buf : array<u32>;
@group(0) @binding(4) var<storage, read_write> vote_debug : array<vec4<f32>>;

@compute @workgroup_size(workgroupSize, 1, 1)
fn orient_vote(@builtin(global_invocation_id) gid: vec3<u32>) {
    let splat_idx = gid.x;
    if (splat_idx >= params.num_splats) { return; }

    let s = splats[splat_idx];
    let pos = vec4<f32>(s.px, s.py, s.pz, 1.0);
    let normal = vec3<f32>(s.nx, s.ny, s.nz);
    let res = params.depth_res;
    let pixels_per_cam = res * res;

    var vote = 0.0;
    var in_frustum_count = 0.0;
    var depth_pass_count = 0.0;
    var grazing_skip_count = 0.0;

    for (var c = 0u; c < params.num_cameras; c++) {
        let cam = cameras[c];
        let clip = cam.viewProj * pos;

        if (clip.w <= 0.0) { continue; }

        let ndc = clip.xyz / clip.w;

        if (ndc.x < -1.0 || ndc.x > 1.0 ||
            ndc.y < -1.0 || ndc.y > 1.0 ||
            ndc.z < 0.0  || ndc.z > 1.0) { continue; }

        in_frustum_count += 1.0;

        let px = u32(clamp((ndc.x * 0.5 + 0.5) * f32(res), 0.0, f32(res - 1u)));
        let py = u32(clamp((1.0 - (ndc.y * 0.5 + 0.5)) * f32(res), 0.0, f32(res - 1u)));

        let buf_idx = c * pixels_per_cam + py * res + px;
        let stored_depth = depth_buf[buf_idx];

        // this pixel was never written to
        if (stored_depth == 0xFFFFFFFFu) { continue; }

        // depth tolerance test: is this splat near the front surface?
        let my_depth = u32(clamp(ndc.z, 0.0, 1.0) * 16777215.0);
        let tolerance = u32(params.depth_tolerance * 16777215.0);

        if (my_depth > stored_depth + tolerance) { continue; }

        depth_pass_count += 1.0;

        let view_dir = normalize(cam.pos.xyz - vec3<f32>(s.px, s.py, s.pz));
        let d = dot(normal, view_dir);

        if (abs(d) < 0.1) {
            grazing_skip_count += 1.0;
            continue;
        }

        vote += d;
    }

    // write debug info: vote, in_frustum, depth_passes, grazing_skips
    vote_debug[splat_idx] = vec4<f32>(vote, in_frustum_count, depth_pass_count, grazing_skip_count);

    // flip normal if consensus says it's backwards
    if (vote < 0.0) {
        splats[splat_idx].nx = -s.nx;
        splats[splat_idx].ny = -s.ny;
        splats[splat_idx].nz = -s.nz;
    }

}
