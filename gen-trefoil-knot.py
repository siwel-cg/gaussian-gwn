"""
Noisy trefoil knot tube test scene for GWN normal orientation robustness testing.

The trefoil knot tubular neighborhood is a compelling stress test because:
  - The tube crosses over/under itself 3 times, creating heavy self-occlusion
  - No viewpoint can see the entire surface — cameras always miss the "underpass"
  - The Frenet frame twists along the curve, so normals rotate non-trivially
  - Genus 1 (boundary of a solid torus topologically), so GWN should read
    ~1 deep inside the tube and ~0 outside
  - Concave pockets form where the knot crosses itself

Parametric trefoil knot:
  x(t) = sin(t) + 2 sin(2t)
  y(t) = cos(t) - 2 cos(2t)
  z(t) = -sin(3t)

Degradation controls (mild — shape should be reconstructable):
  1. FLIPPED NORMALS - small random subset with ~90° wrong orientation
  2. JITTER          - position and rotation noise
  3. DENSITY VARIATION - slight non-uniform sampling
"""

import numpy as np
import struct

# ---- Knot parameters ----
TUBE_R      = 0.25      # tube radius
N_ALONG     = 200       # samples along the knot curve
N_AROUND    = 32        # samples around each tube cross-section
DISK_R      = 0.06      # splat disk radius
THICK       = 0.006     # splat thickness (minor eigenvalue scale)
OPACITY     = 3.0       # opacity logit
KNOT_SCALE  = 0.3       # overall scale factor so the knot fits in a ~2-unit box

# Degradation controls (mild)
FLIP_FRACTION   = 0.03    # fraction with ~90° wrong orientation
POS_JITTER      = 0.008   # std of position noise
ROT_JITTER_DEG  = 2.0     # std of rotation noise on all splats
THIN_FRAC       = 0.05    # fraction of splats randomly culled for density variation

# ---- Trefoil knot curve ----
def trefoil(t):
    """Parametric trefoil knot, t in [0, 2pi)."""
    x = np.sin(t) + 2.0 * np.sin(2.0 * t)
    y = np.cos(t) - 2.0 * np.cos(2.0 * t)
    z = -np.sin(3.0 * t)
    return np.array([x, y, z]) * KNOT_SCALE

def trefoil_deriv(t, dt=1e-5):
    """Numerical tangent vector."""
    return (trefoil(t + dt) - trefoil(t - dt)) / (2.0 * dt)

# ---- Build Frenet-like frame along the knot ----
def compute_frames(t_vals):
    """Compute position, tangent, normal, binormal at each sample."""
    positions = np.array([trefoil(t) for t in t_vals])
    tangents  = np.array([trefoil_deriv(t) for t in t_vals])

    # normalize tangents
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

    # approximate normal via finite difference of tangent (curvature direction)
    # use the "rotation minimizing frame" approach: propagate an initial normal
    # via parallel transport to avoid Frenet frame discontinuities
    normals  = np.zeros_like(tangents)
    binormals = np.zeros_like(tangents)

    # seed: pick an arbitrary vector not parallel to T[0]
    seed = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tangents[0], seed)) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    n0 = seed - np.dot(seed, tangents[0]) * tangents[0]
    n0 /= np.linalg.norm(n0)
    normals[0] = n0
    binormals[0] = np.cross(tangents[0], normals[0])

    # parallel transport via rotation minimizing frame (double reflection method)
    for i in range(1, len(t_vals)):
        v1 = positions[i] - positions[i-1]
        c1 = np.dot(v1, v1)
        if c1 < 1e-12:
            normals[i] = normals[i-1]
            binormals[i] = binormals[i-1]
            continue
        rL = normals[i-1] - (2.0 / c1) * np.dot(v1, normals[i-1]) * v1
        rT = tangents[i-1] - (2.0 / c1) * np.dot(v1, tangents[i-1]) * v1

        v2 = tangents[i] - rT
        c2 = np.dot(v2, v2)
        if c2 < 1e-12:
            normals[i] = rL
        else:
            normals[i] = rL - (2.0 / c2) * np.dot(v2, rL) * v2

        normals[i] /= np.linalg.norm(normals[i])
        binormals[i] = np.cross(tangents[i], normals[i])
        binormals[i] /= np.linalg.norm(binormals[i])

    return positions, tangents, normals, binormals

# ---- Generate tube surface ----
t_vals = np.linspace(0, 2 * np.pi, N_ALONG, endpoint=False)
centers, T, N_frame, B = compute_frames(t_vals)

theta_vals = np.linspace(0, 2 * np.pi, N_AROUND, endpoint=False)

positions = []
normals = []

for i, t in enumerate(t_vals):
    for j, theta in enumerate(theta_vals):
        # point on tube cross-section
        offset = TUBE_R * (np.cos(theta) * N_frame[i] + np.sin(theta) * B[i])
        p = centers[i] + offset

        # outward normal is just the radial direction
        n = np.cos(theta) * N_frame[i] + np.sin(theta) * B[i]

        positions.append(p)
        normals.append(n)

positions = np.array(positions)
normals = np.array(normals)
normals /= np.linalg.norm(normals, axis=1, keepdims=True)

N_total = len(positions)
print(f"Generated {N_total} trefoil knot tube splats")
print(f"  Knot scale={KNOT_SCALE}, tube_r={TUBE_R}")

# ---- Density thinning ----
np.random.seed(42)
keep = np.random.rand(N_total) > THIN_FRAC
positions = positions[keep]
normals = normals[keep]
N = len(positions)
print(f"Kept {N}/{N_total} splats after {THIN_FRAC*100:.0f}% density thinning")

# ---- Build quaternions ----
def quat_from_z_to_vec(n):
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    dot = np.dot(z, n)
    if dot >  0.9999: return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -0.9999: return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(z, n); axis /= np.linalg.norm(axis)
    a = np.arccos(np.clip(dot, -1, 1))
    s = np.sin(a / 2)
    return np.array([np.cos(a / 2), axis[0]*s, axis[1]*s, axis[2]*s])

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
    s = np.sin(angle / 2)
    return np.array([np.cos(angle / 2), axis[0]*s, axis[1]*s, axis[2]*s])

def flip_quat(q):
    """Apply a ~90° random rotation to badly perturb orientation."""
    axis = np.random.randn(3); axis /= np.linalg.norm(axis)
    s = np.sin(np.pi / 4)
    flip = np.array([np.cos(np.pi / 4), axis[0]*s, axis[1]*s, axis[2]*s])
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

# ---- SH (warm copper/gold tones, slight variation) ----
SH_C0 = 0.28209479177387814
base_color = np.array([0.85, 0.55, 0.3])  # copper
sh = np.zeros((N, 48), dtype=np.float32)
for j in range(N):
    color = base_color + np.random.randn(3) * 0.04
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
            f.write(struct.pack('<3f', 0.0, 0.0, 0.0))   # dummy normals in PLY
            f.write(struct.pack('<3f', sh[j,0], sh[j,1], sh[j,2]))
            f.write(struct.pack('<45f', *([0.0]*45)))
            f.write(struct.pack('<f',  opacity_logit))
            f.write(struct.pack('<3f', *scales[j]))
            f.write(struct.pack('<4f', *quats[j]))
    print(f"Wrote {n} splats to {filename}")

write_ply("test_trefoil_knot.ply", positions, quats, scales, sh, OPACITY)

print(f"\nDegradation summary:")
print(f"  Density thin:   {THIN_FRAC*100:.0f}% randomly culled")
print(f"  Flipped:        {len(flip_idx)} splats ({FLIP_FRACTION*100:.0f}%)")
print(f"  Position noise: σ={POS_JITTER}")
print(f"  Rotation noise: σ={ROT_JITTER_DEG}° on all splats")
print(f"  Trefoil knot:   scale={KNOT_SCALE}, tube_r={TUBE_R}")
print(f"  Topology:       genus 1 (solid torus homeomorphic)")
print(f"  Expected GWN:   ~1 inside tube, ~0 outside")
print(f"  Key challenge:  3 self-crossings create heavy mutual occlusion")
