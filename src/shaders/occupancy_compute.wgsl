// occupancy_compute.wgsl — Compute occupancy field O(x) from fat (interior) splats.
//
// For each query point, sums Gaussian density from splats with flatness > threshold.
// Each fat splat contributes opacity * exp(-0.5 * mahalanobis_dist^2).
// Output: O(x) = 1 - exp(-gamma * sum)

override workgroupSize: u32 = 64;

struct Gaussian {
    pos_opacity: array<u32, 2>,
    rot:         array<u32, 2>,
    scale:       array<u32, 2>,
};

struct PrecomputedSplat {
    nx: f32, ny: f32, nz: f32,
    area: f32,
    px: f32, py: f32, pz: f32,
    _pad: f32,  // flatness ratio (0=flat, 1=sphere)
};

struct OccUniforms {
    num_gaussians: u32,
    num_query_pts: u32,
    flatness_threshold: f32,
    gamma: f32,
};

@group(0) @binding(0) var<storage, read>       gaussians    : array<Gaussian>;
@group(0) @binding(1) var<storage, read>       precomputed  : array<PrecomputedSplat>;
@group(0) @binding(2) var<storage, read_write> occ_values   : array<f32>;
@group(0) @binding(3) var<uniform>             occ_uniforms : OccUniforms;
@group(0) @binding(4) var<uniform>             bb_min       : vec4<f32>;
@group(0) @binding(5) var<uniform>             bb_max       : vec4<f32>;
@group(0) @binding(6) var<uniform>             grid_res     : vec4<u32>;

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

fn quat_to_mat(q: vec4<f32>) -> mat3x3<f32> {
    let r = q.x; let x = q.y; let y = q.z; let z = q.w;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0*(y*y + z*z),  2.0*(x*y + r*z),        2.0*(x*z - r*y)),
        vec3<f32>(2.0*(x*y - r*z),        1.0 - 2.0*(x*x + z*z),  2.0*(y*z + r*x)),
        vec3<f32>(2.0*(x*z + r*y),        2.0*(y*z - r*x),        1.0 - 2.0*(x*x + y*y))
    );
}

@compute @workgroup_size(workgroupSize, 1, 1)
fn compute_occupancy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q_idx = gid.x;
    if (q_idx >= occ_uniforms.num_query_pts) { return; }

    let q = query_pos_from_idx(q_idx);
    var occ_raw = 0.0;

    for (var i = 0u; i < occ_uniforms.num_gaussians; i++) {
        let sp = precomputed[i];

        // only fat splats contribute to occupancy
        if (sp._pad <= occ_uniforms.flatness_threshold) { continue; }

        let p = vec3<f32>(sp.px, sp.py, sp.pz);
        let d = q - p;

        // early distance cull (skip if very far — beyond 6 sigma in any axis)
        let dist2 = dot(d, d);
        if (dist2 > 100.0) { continue; }

        let g = gaussians[i];

        // unpack scale and rotation from raw Gaussian
        let s1 = unpack2x16float(g.scale[0]);
        let s2 = unpack2x16float(g.scale[1]);
        let scale = exp(vec3<f32>(s1.x, s1.y, s2.x));

        let r1 = unpack2x16float(g.rot[0]);
        let r2 = unpack2x16float(g.rot[1]);
        let rot = normalize(vec4<f32>(r1.x, r1.y, r2.x, r2.y));
        let R = quat_to_mat(rot);

        // Mahalanobis distance: ||diag(1/s) * R^T * d||^2
        let local_d = transpose(R) * d;
        let scaled = local_d / scale;
        let maha_sq = dot(scaled, scaled);

        // skip if too far in Mahalanobis space
        if (maha_sq > 25.0) { continue; }

        // opacity weight
        let cd = unpack2x16float(g.pos_opacity[1]);
        let opacity = 1.0 / (1.0 + exp(-cd.y));

        occ_raw += opacity * exp(-0.5 * maha_sq);
    }

    occ_values[q_idx] = 1.0 - exp(-occ_uniforms.gamma * occ_raw);
}
