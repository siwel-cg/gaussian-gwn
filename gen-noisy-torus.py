"""
Noisy torus test scene for GWN normal orientation robustness testing.

A torus is a good stress test because:
  - The inner ring has normals pointing into a concave region, partially
    occluded from external cameras
  - Self-occlusion is significant — many splats are hidden from most viewpoints
  - The genus-1 topology means GWN should read ~1 inside and ~0 outside

Degradation controls:
  1. MISSING REGION  - random patch of splats removed
  2. FLIPPED NORMALS - random subset has badly perturbed rotation
  3. JITTER          - position and rotation noise
  4. DENSITY VARIATION - non-uniform sampling density
"""

import numpy as np
import struct

# ---- Torus parameters ----
MAJOR_R   = 1.0       # distance from center of tube to center of torus
MINOR_R   = 0.35      # radius of the tube
N_MAJOR   = 80        # samples around the major circle
N_MINOR   = 40        # samples around the minor circle (tube cross-section)
DISK_R    = 0.08      # splat disk radius
THICK     = 0.008     # splat thickness (minor eigenvalue scale)
OPACITY   = 3.0       # opacity logit

# Degradation controls
HOLE_PATCHES    = 2       # number of random patches to remove
HOLE_ANGLE_DEG  = 25.0    # angular radius of each hole patch (degrees)
FLIP_FRACTION   = 0.05    # fraction with ~90° wrong orientation
POS_JITTER      = 0.02    # std of position noise
ROT_JITTER_DEG  = 3.0     # std of rotation noise on all splats
THIN_BAND_FRAC  = 0.15    # fraction of splats randomly culled for density variation

# ---- Generate torus points ----
u_vals = np.linspace(0, 2*np.pi, N_MAJOR, endpoint=False)
v_vals = np.linspace(0, 2*np.pi, N_MINOR, endpoint=False)

positions = []
normals = []
uv_coords = []  # store (u, v) for patch removal

for u in u_vals:
    for v in v_vals:
        # torus parametric surface
        x = (MAJOR_R + MINOR_R * np.cos(v)) * np.cos(u)
        y = MINOR_R * np.sin(v)
        z = (MAJOR_R + MINOR_R * np.cos(v)) * np.sin(u)

        # outward normal
        nx = np.cos(v) * np.cos(u)
        ny = np.sin(v)
        nz = np.cos(v) * np.sin(u)

        positions.append([x, y, z])
        normals.append([nx, ny, nz])
        uv_coords.append([u, v])

positions = np.array(positions)
normals = np.array(normals)
normals /= np.linalg.norm(normals, axis=1, keepdims=True)
uv_coords = np.array(uv_coords)

N_total = len(positions)
print(f"Generated {N_total} torus splats (R={MAJOR_R}, r={MINOR_R})")

# ---- Remove patches (holes) ----
keep = np.ones(N_total, dtype=bool)

np.random.seed(42)
for h in range(HOLE_PATCHES):
    # pick a random (u, v) center for the hole
    hole_u = np.random.uniform(0, 2*np.pi)
    hole_v = np.random.uniform(0, 2*np.pi)

    # compute geodesic-ish angular distance on the torus parameter space
    du = np.minimum(np.abs(uv_coords[:, 0] - hole_u),
                    2*np.pi - np.abs(uv_coords[:, 0] - hole_u))
    dv = np.minimum(np.abs(uv_coords[:, 1] - hole_v),
                    2*np.pi - np.abs(uv_coords[:, 1] - hole_v))
    dist = np.sqrt(du**2 + dv**2)

    hole_rad = np.deg2rad(HOLE_ANGLE_DEG)
    patch_remove = dist < hole_rad
    keep &= ~patch_remove
    print(f"  Hole patch {h}: center=({np.rad2deg(hole_u):.0f}°, {np.rad2deg(hole_v):.0f}°), "
          f"removed {np.sum(patch_remove)} splats")

# Random density thinning
thin_mask = np.random.rand(N_total) > THIN_BAND_FRAC
keep &= thin_mask

positions = positions[keep]
normals = normals[keep]
N = len(positions)
print(f"Kept {N}/{N_total} splats after holes + thinning")

# ---- Build quaternions ----
def quat_from_z_to_vec(n):
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    dot = np.dot(z, n)
    if dot >  0.9999: return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -0.9999: return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(z, n); axis /= np.linalg.norm(axis)
    a = np.arccos(np.clip(dot, -1, 1))
    s = np.sin(a/2)
    return np.array([np.cos(a/2), axis[0]*s, axis[1]*s, axis[2]*s])

def quat_mul(a, b):
    w1,x1,y1,z1 = a;  w2,x2,y2,z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def random_rot_quat(angle_std_deg):
    axis = np.random.randn(3); axis /= np.linalg.norm(axis)
    angle = np.random.randn() * np.deg2rad(angle_std_deg)
    s = np.sin(angle/2)
    return np.array([np.cos(angle/2), axis[0]*s, axis[1]*s, axis[2]*s])

def flip_quat(q):
    axis = np.random.randn(3); axis /= np.linalg.norm(axis)
    s = np.sin(np.pi/4)
    flip = np.array([np.cos(np.pi/4), axis[0]*s, axis[1]*s, axis[2]*s])
    return quat_mul(flip, q)

quats = np.array([quat_from_z_to_vec(n) for n in normals])

# Rotation jitter on all
for j in range(N):
    quats[j] = quat_mul(random_rot_quat(ROT_JITTER_DEG), quats[j])
    quats[j] /= np.linalg.norm(quats[j])

# Flip random subset
flip_idx = np.random.choice(N, size=int(N * FLIP_FRACTION), replace=False)
for j in flip_idx:
    quats[j] = flip_quat(quats[j])
    quats[j] /= np.linalg.norm(quats[j])

print(f"Flipped {len(flip_idx)} splat orientations ({FLIP_FRACTION*100:.0f}%)")

# Position jitter
positions += np.random.randn(N, 3) * POS_JITTER

# ---- Scales ----
log_scale = np.array([np.log(DISK_R), np.log(DISK_R), np.log(THICK)])
scales = np.tile(log_scale, (N, 1))

# ---- SH (light blue-ish, slightly varied per splat for visual distinction) ----
SH_C0 = 0.28209479177387814
base_color = np.array([0.6, 0.75, 0.9])
sh = np.zeros((N, 48), dtype=np.float32)
for j in range(N):
    color = base_color + np.random.randn(3) * 0.05
    color = np.clip(color, 0.1, 1.0)
    dc = (color - 0.5) / SH_C0
    sh[j, 0] = dc[0]; sh[j, 1] = dc[1]; sh[j, 2] = dc[2]

# ---- Write PLY ----
def write_ply(filename, positions, quats, scales, sh, opacity_logit):
    n = len(positions)
    header_lines = [
        "ply", "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x", "property float y", "property float z",
        "property float nx", "property float ny", "property float nz",
        "property float f_dc_0", "property float f_dc_1", "property float f_dc_2",
    ]
    for k in range(45): header_lines.append(f"property float f_rest_{k}")
    header_lines += [
        "property float opacity",
        "property float scale_0", "property float scale_1", "property float scale_2",
        "property float rot_0", "property float rot_1", "property float rot_2", "property float rot_3",
        "end_header",
    ]
    with open(filename, 'wb') as f:
        f.write(("\n".join(header_lines) + "\n").encode('ascii'))
        for j in range(n):
            f.write(struct.pack('<3f', *positions[j]))
            f.write(struct.pack('<3f', 0.0, 0.0, 0.0))
            f.write(struct.pack('<3f', sh[j,0], sh[j,1], sh[j,2]))
            f.write(struct.pack('<45f', *([0.0]*45)))
            f.write(struct.pack('<f',  opacity_logit))
            f.write(struct.pack('<3f', *scales[j]))
            f.write(struct.pack('<4f', *quats[j]))
    print(f"Wrote {n} splats to {filename}")

write_ply("test_torus_noisy.ply", positions, quats, scales, sh, OPACITY)

print(f"\nDegradation summary:")
print(f"  Holes:          {HOLE_PATCHES} patches of ~{HOLE_ANGLE_DEG}° removed")
print(f"  Density thin:   {THIN_BAND_FRAC*100:.0f}% randomly culled")
print(f"  Flipped:        {len(flip_idx)} splats ({FLIP_FRACTION*100:.0f}%)")
print(f"  Position noise: σ={POS_JITTER}")
print(f"  Rotation noise: σ={ROT_JITTER_DEG}° on all splats")
print(f"  Torus:          R={MAJOR_R}, r={MINOR_R}, genus=1")
print(f"  Expected GWN:   ~1 inside tube, ~0 outside, anomalies near holes")
