// fusion_compute.wgsl — Fuse W_tilde(x) and O(x) via soft OR (probabilistic union).
//
//     F(x) = 1 - (1 - W_tilde(x)) * (1 - O(x))
//          = W_tilde + O - W_tilde * O
//
// Properties:
//   • Both inputs are dimensionless in [0,1] → no manual scale factor needed.
//   • If either source is confident "inside", F is high.
//   • If both agree, F saturates toward 1.
//   • If both are weak, F stays low.
//
// This replaces the old confidence-weighted blend.  No gamma, no flatness.

override workgroupSize: u32 = 64;

struct FusionUniforms {
    num_query_pts: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> gwn_values    : array<f32>;
@group(0) @binding(1) var<storage, read>       occ_values    : array<f32>;
@group(0) @binding(2) var<uniform>             fusion_uniforms : FusionUniforms;

@compute @workgroup_size(workgroupSize, 1, 1)
fn fuse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= fusion_uniforms.num_query_pts) { return; }

    // gwn_values already holds W_tilde (clamped in gwn_compute);
    // occ_values is bounded in [0,1] by construction.  Extra clamp is
    // defensive in case upstream invariants ever drift.
    let w = clamp(gwn_values[idx], 0.0, 1.0);
    let o = clamp(occ_values[idx], 0.0, 1.0);

    // Soft OR: F = 1 - (1 - W)(1 - O)
    gwn_values[idx] = 1.0 - (1.0 - w) * (1.0 - o);
}
