// fusion_compute.wgsl — Fuse winding number W(x) and occupancy O(x) into final field F(x).
//
// W is primary (defines boundary). O is secondary (resolves ambiguity).
// When W is confident (near 0 or 1), trust it. When ambiguous (near 0.5), use O.

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

    let w = gwn_values[idx];
    let o = occ_values[idx];

    // confidence: 1 when W near 0 or 1, 0 when W near 0.5
    let confidence = 4.0 * (w - 0.5) * (w - 0.5);

    // fuse: trust W when confident, blend toward O when ambiguous
    gwn_values[idx] = confidence * w + (1.0 - confidence) * o;
}
