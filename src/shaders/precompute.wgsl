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
    _pad: f32,
};

@group(0) @binding(0) var<storage, read>       gaussians   : array<Gaussian>;
@group(0) @binding(1) var<storage, read_write> precomputed : array<PrecomputedSplat>;
@group(0) @binding(2) var<uniform>             num_gaussians : u32;

fn build_covariance(g: Gaussian) -> mat3x3<f32> {
    let s1    = unpack2x16float(g.scale[0]);
    let s2    = unpack2x16float(g.scale[1]);
    let scale = exp(vec3<f32>(s1.x, s1.y, s2.x));

    let r1 = unpack2x16float(g.rot[0]);
    let r2 = unpack2x16float(g.rot[1]);
    let q  = normalize(vec4<f32>(r1.x, r1.y, r2.x, r2.y));
    let r = q.x; let x = q.y; let y = q.z; let z = q.w;

    let R = mat3x3<f32>(
        vec3<f32>(1.0 - 2.0*(y*y + z*z),  2.0*(x*y + r*z),        2.0*(x*z - r*y)),
        vec3<f32>(2.0*(x*y - r*z),        1.0 - 2.0*(x*x + z*z),  2.0*(y*z + r*x)),
        vec3<f32>(2.0*(x*z + r*y),        2.0*(y*z - r*x),        1.0 - 2.0*(x*x + y*y))
    );

    let sx2 = scale.x * scale.x;
    let sy2 = scale.y * scale.y;
    let sz2 = scale.z * scale.z;

    return mat3x3<f32>(R[0]*sx2, R[1]*sy2, R[2]*sz2) * transpose(R);
}

fn sym3_eigenvalues(A: mat3x3<f32>) -> vec3<f32> {
    let off_sq = A[1][0]*A[1][0] + A[2][0]*A[2][0] + A[2][1]*A[2][1];

    if (off_sq < 1e-12) {
        var ev = vec3<f32>(A[0][0], A[1][1], A[2][2]);
        if (ev.x > ev.y) { let t = ev.x; ev.x = ev.y; ev.y = t; }
        if (ev.y > ev.z) { let t = ev.y; ev.y = ev.z; ev.z = t; }
        if (ev.x > ev.y) { let t = ev.x; ev.x = ev.y; ev.y = t; }
        return ev;
    }

    let q  = (A[0][0] + A[1][1] + A[2][2]) / 3.0;
    let b0 = A[0][0] - q;
    let b1 = A[1][1] - q;
    let b2 = A[2][2] - q;
    let p2 = sqrt((b0*b0 + b1*b1 + b2*b2 + 2.0*off_sq) / 6.0);
    if (p2 < 1e-12) { return vec3<f32>(q, q, q); }

    let ip2 = 1.0 / p2;
    let B00 = b0*ip2; let B11 = b1*ip2; let B22 = b2*ip2;
    let B01 = A[1][0]*ip2; let B02 = A[2][0]*ip2; let B12 = A[2][1]*ip2;

    var r = 0.5 * (B00*(B11*B22 - B12*B12) - B01*(B01*B22 - B12*B02) + B02*(B01*B12 - B11*B02));
    r = clamp(r, -1.0, 1.0);

    let phi = acos(r) / 3.0;
    var ev = vec3<f32>(
        q + 2.0*p2*cos(phi + 2.09439510239),
        q + 2.0*p2*cos(phi + 4.18879020479),
        q + 2.0*p2*cos(phi)
    );
    if (ev.x > ev.y) { let t = ev.x; ev.x = ev.y; ev.y = t; }
    if (ev.y > ev.z) { let t = ev.y; ev.y = ev.z; ev.z = t; }
    if (ev.x > ev.y) { let t = ev.x; ev.x = ev.y; ev.y = t; }
    return ev;
}

fn sym3_min_eigenvector(A: mat3x3<f32>, lam: f32) -> vec3<f32> {
    let B = mat3x3<f32>(
        A[0] - vec3<f32>(lam, 0.0, 0.0),
        A[1] - vec3<f32>(0.0, lam, 0.0),
        A[2] - vec3<f32>(0.0, 0.0, lam)
    );
    let r0 = vec3<f32>(B[0][0], B[1][0], B[2][0]);
    let r1 = vec3<f32>(B[0][1], B[1][1], B[2][1]);
    let r2 = vec3<f32>(B[0][2], B[1][2], B[2][2]);

    let c0 = cross(r0, r1);
    let c1 = cross(r0, r2);
    let c2 = cross(r1, r2);
    let d0 = dot(c0, c0);
    let d1 = dot(c1, c1);
    let d2 = dot(c2, c2);

    if (d0 >= d1 && d0 >= d2) { return c0 / sqrt(max(d0, 1e-12)); }
    if (d1 >= d2)              { return c1 / sqrt(max(d1, 1e-12)); }
    return                              c2 / sqrt(max(d2, 1e-12));
}

@compute @workgroup_size(workgroupSize, 1, 1)
fn precompute(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= num_gaussians) { return; }

    let g = gaussians[idx];

    // position
    let ab = unpack2x16float(g.pos_opacity[0]);
    let cd = unpack2x16float(g.pos_opacity[1]);
    let pos = vec3<f32>(ab.x, ab.y, cd.x);

    // covariance → eigendecomposition
    let Sigma = build_covariance(g);
    let ev = sym3_eigenvalues(Sigma);
    let normal = sym3_min_eigenvector(Sigma, ev.x); // unsigned for now

    // disk area from the two larger eigenvalues
    let area = 3.14159265 * sqrt(max(0.0, ev.y) * max(0.0, ev.z));

    // surface-likeness score: s_i = 1 - sigma3 / sigma2.
    //   s_i ~ 1  → planar (disk-like)  → contributes to W (winding) field
    //   s_i ~ 0  → isotropic (blob)    → contributes to O (occupancy) field
    // ev is sorted ascending, so ev.x = sigma3, ev.y = sigma2, ev.z = sigma1.
    // No threshold; every splat contributes continuously to BOTH fields with
    // weights s_i and b_i = 1 - s_i.
    let s_i = clamp(1.0 - ev.x / max(ev.y, 1e-12), 0.0, 1.0);

    precomputed[idx] = PrecomputedSplat(
        normal.x, normal.y, normal.z,
        area,
        pos.x, pos.y, pos.z,
        s_i
    );
}