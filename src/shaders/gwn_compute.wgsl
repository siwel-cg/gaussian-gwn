// gwn_compute.wgsl — Surface (W) field via generalized winding number.
//
// Every splat contributes continuously, weighted by alpha_i * s_i, where
//   s_i = 1 - sigma3/sigma2  (precomputed in PrecomputedSplat._pad)
// is the surface-likeness score.  Planar splats dominate; isotropic blobs
// contribute almost nothing to W.  No hard threshold.
//
// Final W is clamped to [0,1] (W_tilde) so that fusion sees a bounded value.

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
    _pad: f32,   // s_i (surface-likeness, 0..1)
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
@group(0) @binding(6) var<storage, read>       gaussians    : array<Gaussian>;

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
        let s_i = s._pad;  // surface-likeness in [0,1]

        // Skip pure-blob splats outright (s_i = 0 contributes nothing anyway).
        // This is an early-out for efficiency, NOT a hand-tuned cutoff.
        if (s_i < 1e-4) { continue; }

        // Read trained opacity (alpha_i) from raw Gaussian buffer.
        let g  = gaussians[i];
        let cd = unpack2x16float(g.pos_opacity[1]);
        let alpha_i = 1.0 / (1.0 + exp(-cd.y));

        let p      = vec3<f32>(s.px, s.py, s.pz);
        let normal = vec3<f32>(s.nx, s.ny, s.nz);
        let area   = s.area;

        let diff  = p - q;
        let dist2 = dot(diff, diff);
        if (dist2 < 1e-8) { continue; }
        let dist3 = dist2 * sqrt(dist2);

        // W_i(x) = (area / 4π) · ⟨p - q, n⟩ / ||p - q||³
        let w_i = (area / (4.0 * 3.14159265)) * dot(diff, normal) / dist3;

        // Weighted by alpha_i * s_i  →  only surface-like, opacity-supported splats matter.
        winding += alpha_i * s_i * w_i;
    }

    // W_tilde(x) = clamp(W, 0, 1).  This is a modeling prior: the field is a
    // soft inside-indicator, not a topological invariant.  Overshoots from
    // overlapping splats are not real interior mass.
    gwn_values[q_idx] = clamp(winding, 0.0, 1.0);
}
