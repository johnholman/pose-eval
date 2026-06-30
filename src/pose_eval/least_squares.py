import numpy as np
from math import pi, sqrt, atan2
from scipy.optimize import least_squares, OptimizeResult

from pose_eval.utils.angles import signed_angular_distance


def optimize_scipy(lm_relpos: list[tuple[float, float]], lm_abspos: list[tuple[float, float]],
             start_pose:
             tuple[float, float, float]) -> OptimizeResult:
    # result = least_squares(fun=residuals, x0=start_pose, args=[lm_relpos, lm_abspos], method='lm', jac=jacobian)
    residuals.iteration=0

    result = least_squares(fun=residuals, x0=start_pose, args=[lm_relpos, lm_abspos],
                           ftol=1e-3, xtol=1e-3, max_nfev=15)
    # result = least_squares(fun=residuals, x0=start_pose, args=[lm_relpos, lm_abspos])

    return result


def residuals(pose, lm_relpos, lm_abspos) -> list:
    x, y, theta = pose
    d_res = []
    phi_res = []
    for (phi, d), (lmx, lmy) in zip(lm_relpos, lm_abspos):
        rpx = lmx - x
        rpy = lmy - y
        d_res.append(sqrt(rpx ** 2 + rpy ** 2) - d)
        phi_res.append(signed_angular_distance(atan2(rpy, rpx) - theta, phi))
    current_residuals = d_res + phi_res

    # dist_loss = 0.5 * sum(res ** 2 for res in d_res)
    # angular_loss = 0.5 * sum(res ** 2 for res in phi_res)
    # total_loss = dist_loss + angular_loss
    # print(f'{residuals.iteration}: distance loss: {dist_loss} angular loss: {angular_loss} total: {total_loss}')
    # residuals.iteration+=1

    return current_residuals


def jacobian(pose, lm_relpos, lm_abspos) -> np.ndarray:
    x, y, theta = pose
    n_beacons = len(lm_abspos)

    # Initialize a native NumPy matrix
    J = np.zeros((2 * n_beacons, 3))

    for i, (lmx, lmy) in enumerate(lm_abspos):
        rp_x = lmx - x
        rp_y = lmy - y
        D2 = rp_x ** 2 + rp_y ** 2
        D = sqrt(D2)

        if D2 == 0:
            continue

        J[i, 0] = -rp_x / D
        J[i, 1] = -rp_y / D
        # J[i, 2] is already 0.0

        J[n_beacons + i, 0] = rp_y / D2
        J[n_beacons + i, 1] = -rp_x / D2
        J[n_beacons + i, 2] = -1.0

    return J  # Returns a native NumPy array




def optimize(start_pose, lm_relpos, lm_abspos, max_iters=15, tol=0.01):
    pose = np.array(start_pose, dtype=np.float64)
    # residuals.iteration=0

    for i in range(max_iters):
        r = residuals(pose, lm_relpos, lm_abspos)
        J = jacobian(pose, lm_relpos, lm_abspos)

        JT = J.T
        H = JT @ J + 1e-4 * np.eye(3)
        g = -JT @ r

        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            # Geometry exploded (e.g., collinear beacons or bad circles).
            # Return original pose and infinite cost to force rejection.
            return start_pose, float('inf')

        pose += delta
        if np.linalg.norm(delta) < tol:
            break

    # Get final clean residual for the cost calculation
    final_r = residuals(pose, lm_relpos, lm_abspos)
    final_loss = 0.5 * np.dot(final_r, final_r)

    pose[2] = (pose[2] + np.pi) % (2 * np.pi) - np.pi
    return pose.tolist(), float(final_loss), i+1

if __name__ == "__main__":

    marks = [(1, 0), (0., -1), (0, 1)]  # landmark positions

    obs = [(0.0, 1.0), (-pi / 2, 1.0), (pi / 2, 1.0)]  # relative position observations
    obs = [(0.05, 1.01), (-pi / 2 + 0.01, 1.0), (pi / 2 + 0.05, 0.99)]  # relative position observations
    # obs = [(0.1, 1.2), (-pi+0.1 / 2, 0.8), (pi+0.1 / 2, 1.3)]  # relative position observations

    start = (-3, 2, 2)
    start = (4, 4, pi)

    print(f'start: {start} obs {obs}')
    pose, loss, iters = optimize(start_pose=start, lm_relpos=obs, lm_abspos=marks)
    print(f'optimal pose {pose} in {iters} iterations with loss {loss}')

    import time

    # Benchmark TRF
    residuals.iteration = 0
    t0 = time.perf_counter()
    for _ in range(1000):
        result = optimize(lm_relpos=obs, lm_abspos=marks, start_pose=start)

    t1 = time.perf_counter()
    print(f"TRF average execution time: {(t1 - t0) / 1000 * 1000:.3f} ms")
