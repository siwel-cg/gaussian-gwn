"""
Generate a synthetic Gaussian splat PLY file of a torus.

Produces two types of splats for GWN testing:
  - Surface splats: flat (pancake-like), oriented with normals aligned to the surface
  - Interior splats: rounder (more isotropic), filling the volume inside the torus

The PLY follows the standard 3DGS format with SH coefficients (DC only),
scale, rotation (quaternion), and opacity.

Lewis Ghrist — gaussian-gwn test data generation
"""

import numpy as np
from dataclasses import dataclass
import struct
import argparse
import sys


@dataclass
class SplatCloud:
    positions: np.ndarray   # (N, 3)
    normals: np.ndarray     # (N, 3)
    scales: np.ndarray      # (N, 3) log-scale
    rotations: np.ndarray   # (N, 4) quaternion wxyz
    opacities: np.ndarray   # (N,) logit-space
    colors_dc: np.ndarray   # (N, 3) SH DC (f_dc_0, f_dc_1, f_dc_2)


# ─── Quaternion helpers ────────────────────────────────────────────────

def quat_from_normal(n: np.ndarray) -> np.ndarray:
    """
    Build a quaternion that rotates [0, 0, 1] to align with normal n.
    Returns (N, 4) array of wxyz quaternions.
    """
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)
    N = n.shape[0]

    # Reference direction: z-axis
    z = np.array([0.0, 0.0, 1.0])

    # Cross product z × n  →  rotation axis
    cross = np.cross(np.broadcast_to(z, (N, 3)), n)
    dot = n[:, 2]  # z · n = n_z

    # Quaternion: q = [1 + dot, cross] then normalize
    w = 1.0 + dot
    quats = np.column_stack([w, cross])  # (N, 4) wxyz

    # Handle anti-parallel case (n ≈ -z)
    anti = dot < -0.999
    if np.any(anti):
        quats[anti] = [0.0, 1.0, 0.0, 0.0]  # 180° around x

    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    quats = quats / (norms + 1e-12)
    return quats


def random_tangent_rotation(normals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Apply a random rotation around the normal axis to break grid artifacts.
    """
    N = normals.shape[0]
    angles = rng.uniform(0, 2 * np.pi, N)
    half = angles / 2.0
    n = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)
    # Quaternion for rotation around n by angle
    q_twist = np.column_stack([
        np.cos(half),
        n[:, 0] * np.sin(half),
        n[:, 1] * np.sin(half),
        n[:, 2] * np.sin(half),
    ])
    return q_twist


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two (N,4) wxyz quaternion arrays."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.column_stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


# ─── Torus geometry ───────────────────────────────────────────────────

def torus_surface_point(theta, phi, R, r):
    """Parametric torus: (R + r·cos φ)(cos θ, sin θ, 0) + r·sin φ · ẑ"""
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return np.column_stack([x, y, z])


def torus_normal(theta, phi, R, r):
    """Outward unit normal on a torus."""
    nx = np.cos(phi) * np.cos(theta)
    ny = np.cos(phi) * np.sin(theta)
    nz = np.sin(phi)
    return np.column_stack([nx, ny, nz])


def signed_distance_torus(pts, R, r):
    """SDF for a torus centered at origin, major axis in XY plane."""
    xy = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    return np.sqrt((xy - R)**2 + pts[:, 2]**2) - r


# ─── Splat generators ────────────────────────────────────────────────

def generate_surface_splats(
    R: float, r: float,
    n_theta: int, n_phi: int,
    flatness: float,
    rng: np.random.Generator,
    color_rgb: tuple = (0.2, 0.5, 0.9),
) -> SplatCloud:
    """
    Generate flat (pancake) surface splats on a torus.

    flatness: ratio of tangent scale to normal scale.
              e.g. flatness=10 → tangent axes are 10x the normal axis.
    """
    # Uniform-ish parameterization with jitter
    theta_base = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi_base = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta_base, phi_base)
    theta = theta_grid.ravel()
    phi = phi_grid.ravel()

    # Add small jitter to break regularity
    dtheta = (2 * np.pi / n_theta)
    dphi = (2 * np.pi / n_phi)
    theta += rng.uniform(-0.3 * dtheta, 0.3 * dtheta, theta.shape)
    phi += rng.uniform(-0.3 * dphi, 0.3 * dphi, phi.shape)

    N = len(theta)
    positions = torus_surface_point(theta, phi, R, r)
    normals = torus_normal(theta, phi, R, r)

    # Scale: tangent directions large, normal direction small
    # We want the splat to be a thin disk aligned with the surface
    # Approximate inter-splat spacing for overlap
    tangent_scale_theta = (R + r) * dtheta * 0.6  # along major circle
    tangent_scale_phi = r * dphi * 0.6             # along minor circle
    normal_scale = min(tangent_scale_theta, tangent_scale_phi) / flatness

    # Log-space scales (3DGS convention: exp(log_scale) = actual scale)
    log_s1 = np.full(N, np.log(tangent_scale_theta))
    log_s2 = np.full(N, np.log(tangent_scale_phi))
    log_s3 = np.full(N, np.log(normal_scale))
    scales = np.column_stack([log_s1, log_s2, log_s3])

    # Add slight random variation to scales
    scales += rng.normal(0, 0.1, scales.shape)

    # Rotation: align local z with surface normal, then random twist
    q_align = quat_from_normal(normals)
    q_twist = random_tangent_rotation(normals, rng)
    rotations = quat_multiply(q_twist, q_align)

    # Opacity in logit space (sigmoid⁻¹). We want ~0.9 opacity.
    opacities = np.full(N, 2.2)  # sigmoid(2.2) ≈ 0.9

    # SH DC color (the 3DGS convention: color = SH_DC * C0 + 0.5)
    # C0 = 0.28209479177387814
    C0 = 0.28209479177387814
    colors_dc = np.full((N, 3), [(c - 0.5) / C0 for c in color_rgb])

    return SplatCloud(positions, normals, scales, rotations, opacities, colors_dc)


def generate_interior_splats(
    R: float, r: float,
    n_interior: int,
    rng: np.random.Generator,
    color_rgb: tuple = (0.9, 0.3, 0.2),
    margin: float = 0.8,
) -> SplatCloud:
    """
    Generate rounder (near-isotropic) interior/occupancy splats inside the torus.

    These fill the volume and are more spherical than the surface splats.
    margin: fraction of tube radius to stay inside (0.8 = 80% of r from center tube)
    """
    # Strategy: sample in toroidal coordinates with rejection
    # For each splat, pick (theta, phi, rho) where rho ∈ [0, margin * r]
    # Then place at (R + rho·cos φ)(cos θ, sin θ, 0) + rho·sin φ · ẑ
    # Use volumetric density proportional to (R + rho·cos φ) · rho for uniformity

    positions = []
    max_attempts = n_interior * 20
    count = 0
    attempts = 0

    while count < n_interior and attempts < max_attempts:
        batch = min(n_interior * 4, max_attempts - attempts)
        theta = rng.uniform(0, 2 * np.pi, batch)
        phi = rng.uniform(0, 2 * np.pi, batch)
        rho = rng.uniform(0, margin * r, batch)

        # Jacobian-based acceptance for uniform volume sampling
        # density ∝ rho * (R + rho * cos(phi))
        jacobian = rho * (R + rho * np.cos(phi))
        max_jacobian = (margin * r) * (R + margin * r)
        accept_prob = jacobian / max_jacobian
        mask = rng.random(batch) < accept_prob

        x = (R + rho[mask] * np.cos(phi[mask])) * np.cos(theta[mask])
        y = (R + rho[mask] * np.cos(phi[mask])) * np.sin(theta[mask])
        z = rho[mask] * np.sin(phi[mask])
        pts = np.column_stack([x, y, z])

        positions.append(pts)
        count += pts.shape[0]
        attempts += batch

    positions = np.concatenate(positions, axis=0)[:n_interior]
    N = positions.shape[0]

    # Interior splats are rounder — roughly isotropic with slight randomness
    # Target scale: fill the interior with some overlap
    # Approximate: each interior splat covers a ball of radius ~(volume / N)^(1/3)
    torus_volume = 2 * np.pi**2 * R * r**2
    splat_radius = (torus_volume / N) ** (1.0 / 3.0) * 0.6

    # Near-isotropic: all 3 axes similar, with slight random eccentricity
    log_base = np.log(splat_radius)
    scales = np.full((N, 3), log_base)
    scales += rng.normal(0, 0.15, scales.shape)  # slight variation

    # Random rotations (isotropic splats, so rotation matters less)
    quats = rng.normal(0, 1, (N, 4))
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    rotations = quats

    # Normals: point radially outward from tube center (for reference, not critical)
    # Tube center for each splat:
    theta_approx = np.arctan2(positions[:, 1], positions[:, 0])
    tube_center = np.column_stack([R * np.cos(theta_approx), R * np.sin(theta_approx), np.zeros(N)])
    normals = positions - tube_center
    normals /= (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)

    # Lower opacity for interior splats (they accumulate)
    opacities = np.full(N, 1.0)  # sigmoid(1.0) ≈ 0.73

    # Color (reddish for interior, to distinguish from blue surface)
    C0 = 0.28209479177387814
    colors_dc = np.full((N, 3), [(c - 0.5) / C0 for c in color_rgb])

    return SplatCloud(positions, normals, scales, rotations, opacities, colors_dc)


# ─── PLY writer ──────────────────────────────────────────────────────

def write_ply(filename: str, cloud: SplatCloud, n_sh_extra: int = 0):
    """
    Write a 3DGS-compatible PLY file.

    Properties:
      x, y, z           — position
      nx, ny, nz         — normal
      f_dc_0..2          — SH DC coefficients (RGB)
      f_rest_0..N        — higher-order SH (zeros)
      opacity            — logit-space opacity
      scale_0..2         — log-space scales
      rot_0..3           — quaternion (wxyz)
    """
    N = cloud.positions.shape[0]

    # Number of extra SH coefficients (per channel)
    # Degree 0 = 1 coeff (DC only, included separately)
    # Degree 1 = 3 extra
    # Degree 2 = 5 extra
    # Degree 3 = 7 extra
    # Total rest per channel for max_sh_degree=3: 15
    n_rest = n_sh_extra  # total rest coefficients (across all channels)

    header = f"""ply
format binary_little_endian 1.0
element vertex {N}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
"""
    for i in range(n_rest):
        header += f"property float f_rest_{i}\n"

    header += """property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""

    with open(filename, 'wb') as f:
        f.write(header.encode('ascii'))

        rest_zeros = np.zeros(n_rest, dtype=np.float32)

        for i in range(N):
            # Position
            f.write(struct.pack('<fff', *cloud.positions[i]))
            # Normal
            f.write(struct.pack('<fff', *cloud.normals[i]))
            # SH DC
            f.write(struct.pack('<fff', *cloud.colors_dc[i]))
            # SH rest (zeros)
            if n_rest > 0:
                f.write(rest_zeros.tobytes())
            # Opacity
            f.write(struct.pack('<f', cloud.opacities[i]))
            # Scale
            f.write(struct.pack('<fff', *cloud.scales[i]))
            # Rotation (wxyz)
            f.write(struct.pack('<ffff', *cloud.rotations[i]))

    print(f"  Wrote {N} splats to {filename}")


def merge_clouds(*clouds: SplatCloud) -> SplatCloud:
    return SplatCloud(
        positions=np.concatenate([c.positions for c in clouds]),
        normals=np.concatenate([c.normals for c in clouds]),
        scales=np.concatenate([c.scales for c in clouds]),
        rotations=np.concatenate([c.rotations for c in clouds]),
        opacities=np.concatenate([c.opacities for c in clouds]),
        colors_dc=np.concatenate([c.colors_dc for c in clouds]),
    )


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Gaussian splat PLY test data")
    parser.add_argument("--output", "-o", default="torus_splats.ply", help="Output PLY filename")
    parser.add_argument("--R", type=float, default=1.5, help="Torus major radius")
    parser.add_argument("--r", type=float, default=0.5, help="Torus minor (tube) radius")
    parser.add_argument("--n-theta", type=int, default=80, help="Surface samples around major axis")
    parser.add_argument("--n-phi", type=int, default=40, help="Surface samples around minor axis")
    parser.add_argument("--n-interior", type=int, default=2000, help="Number of interior splats")
    parser.add_argument("--flatness", type=float, default=8.0, help="Surface splat flatness ratio (tangent/normal)")
    parser.add_argument("--sh-degree", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Max SH degree (0=DC only, 3=full). Higher adds zero-filled rest coefficients.")
    parser.add_argument("--surface-only", action="store_true", help="Only generate surface splats")
    parser.add_argument("--interior-only", action="store_true", help="Only generate interior splats")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--surface-color", type=float, nargs=3, default=[0.2, 0.5, 0.9],
                        metavar=("R", "G", "B"), help="Surface splat color (0-1 RGB)")
    parser.add_argument("--interior-color", type=float, nargs=3, default=[0.9, 0.3, 0.2],
                        metavar=("R", "G", "B"), help="Interior splat color (0-1 RGB)")

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    # SH rest coefficients count
    sh_rest_per_channel = {0: 0, 1: 3, 2: 8, 3: 15}
    n_rest = sh_rest_per_channel[args.sh_degree] * 3  # 3 channels

    print(f"Generating torus splats: R={args.R}, r={args.r}")
    print(f"  SH degree: {args.sh_degree} ({n_rest} rest coefficients)")

    clouds = []

    if not args.interior_only:
        print(f"  Surface: {args.n_theta}×{args.n_phi} = {args.n_theta * args.n_phi} splats, flatness={args.flatness}")
        surface = generate_surface_splats(
            args.R, args.r, args.n_theta, args.n_phi,
            args.flatness, rng,
            color_rgb=tuple(args.surface_color),
        )
        clouds.append(surface)

    if not args.surface_only:
        print(f"  Interior: {args.n_interior} splats")
        interior = generate_interior_splats(
            args.R, args.r, args.n_interior, rng,
            color_rgb=tuple(args.interior_color),
        )
        clouds.append(interior)

    combined = merge_clouds(*clouds)
    write_ply(args.output, combined, n_sh_extra=n_rest)

    # Summary stats
    print(f"\nSummary:")
    print(f"  Total splats: {combined.positions.shape[0]}")
    print(f"  Bounding box: [{combined.positions.min(axis=0)}] → [{combined.positions.max(axis=0)}]")
    s_exp = np.exp(combined.scales)
    print(f"  Scale range:  [{s_exp.min():.4f}, {s_exp.max():.4f}]")
    print(f"  Opacity range (sigmoid): [{1/(1+np.exp(-combined.opacities.min())):.3f}, {1/(1+np.exp(-combined.opacities.max())):.3f}]")


if __name__ == "__main__":
    main()
