import numpy as np
from scipy.interpolate import CubicSpline


def fit_parametric_spline(pts):
    """
    Fit separate cubic splines for x, y, z over a normalized parameter [0,1].
    """
    N = pts.shape[0]
    t_orig = np.linspace(0, 1, N)
    cs_x = CubicSpline(t_orig, pts[:, 0])
    cs_y = CubicSpline(t_orig, pts[:, 1])
    cs_z = CubicSpline(t_orig, pts[:, 2])
    return cs_x, cs_y, cs_z


def compute_arc_length(cs_x, cs_y, cs_z, num_samples=1000):
    """
    Approximate the total arc length of the parametric spline by dense sampling.
    Returns sampled parameters `t_dense` and cumulative lengths `cumlen`.
    """
    t_dense = np.linspace(0, 1, num_samples)
    x = cs_x(t_dense)
    y = cs_y(t_dense)
    z = cs_z(t_dense)
    pts_dense = np.vstack((x, y, z)).T
    diffs = np.diff(pts_dense, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate(([0], np.cumsum(seg_lens)))
    return t_dense, cumlen


def sample_by_arc_length(cs_x, cs_y, cs_z, t_dense, cumlen, end_time):
    """
    Sample positions and unit direction vectors at integer times.
    """
    total_length = cumlen[-1]
    speed = total_length / end_time
    times = np.round(np.linspace(0, end_time, end_time))
    s_targets = speed * times

    u_samples = np.interp(s_targets, cumlen, t_dense)

    x_s = cs_x(u_samples)
    y_s = cs_y(u_samples)
    z_s = cs_z(u_samples)
    positions = np.vstack((x_s, y_s, z_s)).T

    dx = cs_x.derivative()(u_samples)
    dy = cs_y.derivative()(u_samples)
    dz = cs_z.derivative()(u_samples)
    v = np.vstack((dx, dy, dz)).T

    norms = np.linalg.norm(v, axis=1, keepdims=True)
    directions = v / norms

    return total_length, speed, positions, directions


def calculate_position_and_directions(points: np.ndarray, end_time: int) -> tuple:
    cx, cy, cz = fit_parametric_spline(points)
    timing_density, cumulative_length = compute_arc_length(cx,
                                                           cy,
                                                           cz,
                                                           num_samples=(end_time - 1) * 1000 + 1)
    return sample_by_arc_length(cx,
                                cy,
                                cz,
                                timing_density,
                                cumulative_length,
                                end_time)
