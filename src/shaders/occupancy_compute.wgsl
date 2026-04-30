// occupancy_compute.wgsl — Interior (O) field via bounded soft union of Gaussian
// membership kernels.  No gamma, no flatness threshold.
//
// For each query point x, every splat contributes a bounded membership value
//     o_i(x) = alpha_i * b_i * exp(-0.5 * d_i^2(x))         in [0,1]
// where b_i = 1 - s_i is the blob-likeness (precomputed _pad holds s_i), and
// d_i^2 is squared Mahalanobis distance.  The aggregate is the saturating
// soft union (probabilistic OR):
//     O(x) = 1 - prod_i (1 - o_i(x))
// implemented per-query as a running product (occ_product *= (1 - o_i)),
// then O = 1 - occ_product.  Bounded by construction.
//
// d_i^2 > 9 (3-sigma) splats are skipped — efficiency cutoff, not a parameter.

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
    _pad: f32,   // s_i (surface-likeness, 0..1).  b_i = 1 - s_i.
};

struct OccUniforms {
    num_gaussians: u32,
    num_query_pts: u32,
    _pad0: u32,
    _pad1: u32,
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

    // Running product for the soft union  O = 1 - prod(1 - o_i).
    // Saturates gracefully — multiple weak splats reinforce, but the result
    // can never exceed 1 and accumulation costs no extra GPU primitives.
    var occ_product = 1.0;

    for (var i = 0u; i < occ_uniforms.num_gaussians; i++) {
        let sp = precomputed[i];
        let s_i = sp._pad;
        let b_i = 1.0 - s_i;        // blob-likeness in [0,1]

        // Pure-disk splats contribute nothing here (b_i = 0).  Skip for speed.
        if (b_i < 1e-4) { continue; }

        let p = vec3<f32>(sp.px, sp.py, sp.pz);
        let d = q - p;

        // Coarse early distance cull (units of world space, conservative).
        let dist2 = dot(d, d);
        if (dist2 > 100.0) { continue; }

        let g = gaussians[i];

        // Unpack scale & rotation from the raw Gaussian.
        let s1 = unpack2x16float(g.scale[0]);
        let s2 = unpack2x16float(g.scale[1]);
        let scale = exp(vec3<f32>(s1.x, s1.y, s2.x));

        let r1 = unpack2x16float(g.rot[0]);
        let r2 = unpack2x16float(g.rot[1]);
        let rot = normalize(vec4<f32>(r1.x, r1.y, r2.x, r2.y));
        let R = quat_to_mat(rot);

        // Mahalanobis distance:  ||diag(1/s) · R^T · d||²
        let local_d = transpose(R) * d;
        let scaled  = local_d / scale;
        let maha_sq = dot(scaled, scaled);

        // 3-sigma support cutoff (efficiency only, not a tunable parameter).
        if (maha_sq > 9.0) { continue; }

        // Trained opacity.
        let cd = unpack2x16float(g.pos_opacity[1]);
        let alpha_i = 1.0 / (1.0 + exp(-cd.y));

        // o_i(x) ∈ [0, 1] — bounded membership, NOT a normalized density.
        let o_i = clamp(alpha_i * b_i * exp(-0.5 * maha_sq), 0.0, 1.0);

        // Accumulate the soft union as a running product of complements.
        occ_product *= (1.0 - o_i);
    }

    // O(x) = 1 - prod(1 - o_i)  →  bounded in [0,1] by construction.
    occ_values[q_idx] = 1.0 - occ_product;
}
