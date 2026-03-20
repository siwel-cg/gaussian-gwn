override workgroupSize: u32 = 64;

// THIS IS THE STUFF FROM THE PRECOMPUTE SHADER  
struct PrecomputedSplat {
    nx: f32, ny: f32, nz: f32,
    area: f32,
    px: f32, py: f32, pz: f32,
    _pad: f32,
};

struct GWNUniforms {
    num_gaussians: u32,
    num_query_pts: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       splats       : array<PrecomputedSplat>;
@group(0) @binding(1) var<storage, read_write> gwn_values   : array<f32>;
@group(0) @binding(2) var<uniform>             gwn_uniforms : GWNUniforms;
@group(0) @binding(3) var<uniform>             bb_min       : vec4<f32>;
@group(0) @binding(4) var<uniform>             bb_max       : vec4<f32>;
@group(0) @binding(5) var<uniform>             grid_res     : vec4<u32>;

fn query_pos_from_idx(idx: u32) -> vec3<f32> {
    let rx = grid_res.x;
    let ry = grid_res.y;
    let rz = grid_res.z;
    let iz  = idx / (rx * ry);
    let rem = idx % (rx * ry);
    let iy  = rem / rx;
    let ix  = rem % rx;
    let t = vec3<f32>(
        (f32(ix) + 0.5) / f32(rx),
        (f32(iy) + 0.5) / f32(ry),
        (f32(iz) + 0.5) / f32(rz),
    );
    return bb_min.xyz + t * (bb_max.xyz - bb_min.xyz);
}

@compute @workgroup_size(workgroupSize, 1, 1)
fn compute_gwn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q_idx = gid.x;
    if (q_idx >= gwn_uniforms.num_query_pts) { return; }

    let q = query_pos_from_idx(q_idx);
    var winding = 0.0;

    for (var i = 0u; i < gwn_uniforms.num_gaussians; i++) {
        let s = splats[i];

        let p      = vec3<f32>(s.px, s.py, s.pz);
        let normal = vec3<f32>(s.nx, s.ny, s.nz);
        let area   = s.area;

        let diff  = p - q;
        let dist2 = dot(diff, diff);
        if (dist2 < 1e-8) { continue; }
        let dist3 = dist2 * sqrt(dist2);

        winding += (area / (4.0 * 3.14159265)) * dot(diff, normal) / dist3;
    }

    gwn_values[q_idx] = winding;
}