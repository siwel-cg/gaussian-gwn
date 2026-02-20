"""
Noisy sphere test scene for GWN robustness testing.
Introduces three types of degradation:
  1. MISSING REGION  - a cap of splats removed (simulates a hole)
  2. FLIPPED NORMALS - random subset has rotation perturbed 90deg (wrong orientation)
  3. JITTER          - position and rotation noise on all splats
"""

import numpy as np
import struct

# ---- Parameters ----
N         = 400
RADIUS    = 1.0
DISK_R    = 0.18
THICK     = 0.01
OPACITY   = 3.0

# Degradation controls
HOLE_THETA      = np.pi * 0.0   # angular radius of missing cap (0 = no hole)
FLIP_FRACTION   = 0.0           # fraction of splats with badly perturbed rotation
POS_JITTER      = 0.3           # std of position noise (world units)
ROT_JITTER_DEG  = 0.0           # std of rotation noise (degrees, applied to all)

# ---- Fibonacci sphere ----
golden = np.pi * (3.0 - np.sqrt(5.0))
i = np.arange(N)
y = 1.0 - (i / (N - 1)) * 2.0
r = np.sqrt(np.clip(1.0 - y*y, 0, 1))
theta = golden * i
positions = np.stack([r * np.cos(theta), y, r * np.sin(theta)], axis=1) * RADIUS
normals   = positions / np.linalg.norm(positions, axis=1, keepdims=True)

# ---- Remove a cap (hole) ----
hole_dir  = np.array([0.0, 1.0, 0.0])  # top of sphere
cos_thresh = np.cos(HOLE_THETA)
keep = np.array([np.dot(n, hole_dir) < cos_thresh for n in normals])
positions = positions[keep]
normals   = normals[keep]
print(f"Kept {len(positions)}/{N} splats after removing cap hole")

N = len(positions)

# ---- Build quaternions aligning local-Z to outward normal ----
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
    """Small random rotation quaternion."""
    axis = np.random.randn(3); axis /= np.linalg.norm(axis)
    angle = np.random.randn() * np.deg2rad(angle_std_deg)
    s = np.sin(angle/2)
    return np.array([np.cos(angle/2), axis[0]*s, axis[1]*s, axis[2]*s])

def flip_quat(q):
    """90-degree rotation around a random tangent axis — badly wrong orientation."""
    axis = np.random.randn(3); axis /= np.linalg.norm(axis)
    s = np.sin(np.pi/4)
    flip = np.array([np.cos(np.pi/4), axis[0]*s, axis[1]*s, axis[2]*s])
    return quat_mul(flip, q)

quats = np.array([quat_from_z_to_vec(n) for n in normals])

# Apply small rotation jitter to all splats
for j in range(N):
    quats[j] = quat_mul(random_rot_quat(ROT_JITTER_DEG), quats[j])
    quats[j] /= np.linalg.norm(quats[j])

# Flip a random subset (bad orientation)
flip_idx = np.random.choice(N, size=int(N * FLIP_FRACTION), replace=False)
for j in flip_idx:
    quats[j] = flip_quat(quats[j])
    quats[j] /= np.linalg.norm(quats[j])

print(f"Flipped {len(flip_idx)} splat orientations ({FLIP_FRACTION*100:.0f}%)")

# Apply position jitter
positions += np.random.randn(N, 3) * POS_JITTER

# ---- Scales ----
log_scale = np.array([np.log(DISK_R), np.log(DISK_R), np.log(THICK)])
scales = np.tile(log_scale, (N, 1))

# ---- SH (white) ----
SH_C0 = 0.28209479177387814
dc_val = (1.0 - 0.5) / SH_C0
sh = np.zeros((N, 48), dtype=np.float32)
sh[:, 0] = dc_val; sh[:, 1] = dc_val; sh[:, 2] = dc_val

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

write_ply("test_sphere_noisy.ply", positions, quats, scales, sh, OPACITY)
print(f"\nDegradation summary:")
print(f"  Hole:          cap of {np.rad2deg(HOLE_THETA):.0f}° removed from top")
print(f"  Flipped:       {len(flip_idx)} splats have ~90° wrong orientation")
print(f"  Position noise: σ={POS_JITTER} (sphere radius={RADIUS})")
print(f"  Rotation noise: σ={ROT_JITTER_DEG}° on all splats")
