import numpy as np
import pyvista as pv
import spline as sp

if __name__ == "__main__":
    points = np.array([[450, 0, 0],
                       [0, 450, 0],
                       [50, 0, 0],
                       [0, 50, 0]])

    end_time = 11
    total_length, speed, positions, directions = sp.calculate_position_and_directions(points, end_time)

    a_f, a_r, b, c = 5, 10, 5, 5

    for i in range(end_time):
        # build the grid centered at current position
        x = np.linspace(0, 500, 501) - positions[i, 0]
        y = np.linspace(0, 500, 501) - positions[i, 1]
        z = np.linspace(0, 500, 501) - positions[i, 2]
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        coords = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

        u = directions[i]
        u = u / np.linalg.norm(u)

        if abs(u[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])

        v = np.cross(u, ref)
        v /= np.linalg.norm(v)
        w = np.cross(u, v)

        R = np.vstack([u, v, w])  # now shape (3,3)

        rotated = coords.dot(R.T)

        a = (0.5 * (np.sign(rotated[:, 0]) + 1) * a_f
             + 0.5 * (1 - np.sign(rotated[:, 0])) * a_r)
        inside = ((rotated[:, 0] ** 2 / a ** 2) +
                  (rotated[:, 1] ** 2 / b ** 2) +
                  (rotated[:, 2] ** 2 / c ** 2)) < 1.0

        inside_pts = coords[inside]
        inside_pts[:, 0] = inside_pts[:, 0] + positions[i, 0]
        inside_pts[:, 1] = inside_pts[:, 1] + positions[i, 1]
        inside_pts[:, 2] = inside_pts[:, 2] + positions[i, 2]

        plotter = pv.Plotter()
        plotter.add_points(inside_pts,
                           render_points_as_spheres=True,
                           point_size=10)

        plotter.show_grid(bounds=[0, 500, 0, 500, 0, 500])
        plotter.show()
