"""
Generate a test .ply file of Gaussian splats arranged on a sphere surface.
Each splat is a flat disk tangent to the sphere — ideal for GWN:
  - smallest eigenvalue axis  = radial (surface normal)
  - two larger eigenvalue axes = tangent plane
  - covariance is perfectly oriented for winding number computation
"""

import numpy as np
import struct

# ---- Parameters ----
N        = 300    # number of splats (enough to cover a sphere reasonably)
RADIUS   = 1.0    # sphere radius
DISK_R   = 0.18   # tangential semi-axis (how wide each splat is)
THICK    = 0.01   # radial semi-axis    (how thin/flat each splat is)
OPACITY  = 3.0    # pre-sigmoid logit: sigmoid(3) ≈ 0.95

# ---- Distribute points uniformly on sphere (Fibonacci lattice) ----
golden = np.pi * (3.0 - np.sqrt(5.0))
i = np.arange(N)
y = 1.0 - (i / (N - 1)) * 2.0          # y in [-1, 1]
r = np.sqrt(np.clip(1.0 - y*y, 0, 1))
theta = golden * i

positions = np.stack([r * np.cos(theta), y, r * np.sin(theta)], axis=1) * RADIUS

# ---- Build rotation quaternion for each splat ----
# We want the splat's local Z axis (scale[2] = THICK direction) to align
# with the outward normal so the Gaussian is flat against the surface.
p0# Rotation: align world-Z (0,0,1) -> normal n

def quat_from_z_to_normal(n):
    """Quaternion rotating (0,0,1) to n. Returns (w, x, y, z)."""
    z = np.array([0.0, 0.0, 1.0])
    n = n / np.linalg.norm(n)
    dot = np.dot(z, n)
    if dot > 0.9999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -0.9999:
        # 180° rotation around X
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(z, n)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot, -1, 1))
    s = np.sin(angle / 2)
    return np.array([np.cos(angle / 2), axis[0]*s, axis[1]*s, axis[2]*s])

normals = positions / np.linalg.norm(positions, axis=1, keepdims=True)
quats = np.array([quat_from_z_to_normal(n) for n in normals])  # (N, 4) wxyz

# ---- Scales: log-scale stored (3DGS convention: scale = exp(stored)) ----
# Two large tangential axes, one thin radial axis.
log_scale = np.array([np.log(DISK_R), np.log(DISK_R), np.log(THICK)])
scales = np.tile(log_scale, (N, 1))  # (N, 3)

# ---- SH coefficients: just DC term for a flat white color ----
# DC SH value that produces white: color = SH_C0 * coef + 0.5 = 1.0
# => coef = (1.0 - 0.5) / SH_C0 = 0.5 / 0.28209... ≈ 1.7725
SH_C0 = 0.28209479177387814
dc_val = (1.0 - 0.5) / SH_C0
# 16 coefficients * 3 channels = 48 floats, but we only set DC (coef 0)
sh = np.zeros((N, 48), dtype=np.float32)
sh[:, 0] = dc_val  # R dc
sh[:, 1] = dc_val  # G dc
sh[:, 2] = dc_val  # B dc

# ---- Pack into 3DGS .ply format ----
# Properties (matching load.ts packing):
#   x, y, z, nx, ny, nz          (f32 × 6, normals unused = 0)
#   f_dc_0..2                    (f32 × 3, DC color)
#   f_rest_0..44                 (f32 × 45, higher SH = 0)
#   opacity                      (f32, logit)
#   scale_0..2                   (f32 × 3)
#   rot_0..3                     (f32 × 4, wxyz)

def write_ply(filename, positions, quats, scales, sh, opacity_logit):
    N = len(positions)
    
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y", 
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    for i in range(45):
        header_lines.append(f"property float f_rest_{i}")
    header_lines += [
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ]
    header = "\n".join(header_lines) + "\n"

    with open(filename, 'wb') as f:
        f.write(header.encode('ascii'))
        for i in range(N):
            # x y z
            f.write(struct.pack('<3f', *positions[i]))
            # nx ny nz (unused)
            f.write(struct.pack('<3f', 0.0, 0.0, 0.0))
            # f_dc_0..2
            f.write(struct.pack('<3f', sh[i,0], sh[i,1], sh[i,2]))
            # f_rest_0..44 (zeros)
            f.write(struct.pack('<45f', *([0.0]*45)))
            # opacity
            f.write(struct.pack('<f', opacity_logit))
            # scale_0..2
            f.write(struct.pack('<3f', *scales[i]))
            # rot_0..3 (w x y z)
            f.write(struct.pack('<4f', *quats[i]))

    print(f"Wrote {N} splats to {filename}")
    print(f"  Sphere radius: {RADIUS}")
    print(f"  Disk radius:   {DISK_R}  (tangential semi-axis)")
    print(f"  Thickness:     {THICK}   (radial semi-axis, = surface normal direction)")
    print(f"  Eigenvalue ratio: {DISK_R/THICK:.0f}:1 (large means very flat = good normals)")

write_ply("test_sphere.ply", positions, quats, scales, sh, OPACITY)
