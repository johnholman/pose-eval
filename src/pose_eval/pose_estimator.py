import time
from dataclasses import dataclass
from itertools import combinations
from math import pi
from pprint import pprint

from pose_eval.least_squares import optimize
from pose_eval.beacon import BeaconRegistry, BeaconData
from pose_eval.utils.angles import angular_distance, average_angle
from pose_eval.utils.pose import Pose
from pose_eval.two_beacons_no_opt import estimate_pose as geom_estimate_pose


@dataclass
class LandmarkPoseData:
    """Class holding current pose data """
    pose: tuple[float, float, float]  # landmark pose estimate
    loss: float | None  # optimization loss
    adopted: bool  # whether the landmark pose estimate was adopted as the current pose

class LandmarkPoseEstimator:
    eps = 0.02

    def __init__(self, beacon_registry: BeaconRegistry):
        self.beacon_registry: BeaconRegistry = beacon_registry
        self.geom_estimate_pose = geom_estimate_pose
        # self.send = get_message_service().send
        # self.log = get_logging_service().log
        # self.optimize_pose = self.make_scaled_least_squares_estimate
        self.optimize_pose = self.make_both_least_squares_estimates
        self.init()

    def init(self):
        self.still_wheel_positions = (0.0, 0.0)
        self.stationary_since = 0
        self.pose = None
        self.pose_merit = None
        self.loss = None


    def estimate_pose(self, step_num: int, wheel_positions: tuple[float, float]) -> tuple[tuple[
        float, float, float], int] | None:

        # get most recent landmark observations made in this stationary period
        if (result := self.stationary_observations(step_num, wheel_positions)) is None:
            return None
        unscaled_landmarks, landmarks = result
        n_landmarks = len(landmarks)

        # no estimate if a geometric estimate could not be made
        if n_landmarks < 2 or (result := self.geometric_pose_estimate(step_num, landmarks)) is None:
            return None

        geom_pose, merit = result
        if n_landmarks == 2:
            # if only two landmarks just use the geometric-based pose and set loss to None
            # optimization would leave the pose unchanged with zero loss so not useful here
            self.pose = geom_pose
            self.loss = None
            print(f'{step_num}: two-beacon pose estimate {Pose(*geom_pose)}')

        elif n_landmarks >= 3:
            # estimate using nonlinear least squares optimization with the geometry-derived pose as the starting point
            self.pose, self.loss = self.optimize_pose(step_num, geom_pose, landmarks, unscaled_landmarks)
        # relpos = [lm.relpos for lm in landmarks.values()]
        # abspos = [lm.abspos for lm in landmarks.values()]
        # start = time.perf_counter()
        # pose, loss, iters = optimize(start_pose=pose, lm_relpos=relpos, lm_abspos=abspos)
        # duration_ms = (time.perf_counter() - start) * 1000
        # # print(f'{step_num}: optimization time {duration_ms:.2f} ms')
        # print(
        #     f'{step_num}: three-beacon pose estimate {Pose(*pose)} loss {loss:.4f} iters {iters} opt time {duration_ms:.2f} ms')

        # TODO why do these need to be attributes?
        return self.pose, self.loss

    def make_least_squares_estimate(self, step_num: int, start_pose, landmarks):
        relpos = [lm.relpos for lm in landmarks.values()]
        abspos = [lm.abspos for lm in landmarks.values()]
        start = time.perf_counter()
        pose, loss, iters = optimize(start_pose=start_pose, lm_relpos=relpos, lm_abspos=abspos)
        duration_ms = (time.perf_counter() - start) * 1000
        return pose, loss, iters

    def make_scaled_least_squares_estimate(self, step_num: int, start_pose, landmarks, _unscaled_landmarks):
        pose, loss, iters = self.make_least_squares_estimate(step_num, start_pose, landmarks)
        return pose, loss

    def make_both_least_squares_estimates(self, step_num: int, start_pose, scaled_landmarks, unscaled_landmarks):
        results = {}
        # send_msg = get_message_service().send
        pose, loss, iters = self.make_least_squares_estimate(step_num=step_num, start_pose=start_pose,
                                                             landmarks=unscaled_landmarks)
        print(
            f'{step_num}: unscaled estimate {Pose(*pose)} loss {loss:.4f} iters {iters}')
        results['scaled'] = { 'pose': pose, 'loss': loss, 'iters': iters}

        # self.send('test/unscaled', {'pose': pose, 'loss': loss, 'iters': iters})
        pose, loss, iters = self.make_least_squares_estimate(step_num=step_num, start_pose=start_pose,
                                                             landmarks=scaled_landmarks)
        results['unscaled'] = { 'pose': pose, 'loss': loss, 'iters': iters}

        print(
            f'{step_num}: scaled estimate {Pose(*pose)} loss {loss:.4f} iters {iters}')

        print(f'{step_num}:')
        pprint(results)

        return results


    def stationary_observations(self, step_num: int, wheel_positions: tuple[float, float]) -> \
            tuple[dict[int, BeaconData], dict[int, BeaconData]] | None:
        """Return most recent observations of each landmark made during the current stationary period"""

        # check whether wheels moved significantly since the start of this stationary period.
        # if so start a new period and return None
        if (abs(self.still_wheel_positions[0] - wheel_positions[0]) > self.eps or
                abs(self.still_wheel_positions[1] - wheel_positions[1]) > self.eps):
            self.still_wheel_positions = wheel_positions
            self.stationary_since = step_num
            print(f"{step_num}: wheels moved to {wheel_positions}")
            # send pose invalidation message when movement first detected
            if self.pose is not None:
                # self.send('rp2/pose/landmark', None)
                msg = f'{step_num}: landmark pose invalidated'
                print(msg)
                # self.log(msg)
            self.pose = None
            self.pose_merit = None
            self.loss = None
            return None

        # otherwise robot is still (almost) stationary so return landmark observations made while static
        print(f"{step_num}: wheels stationery since step {self.stationary_since}")
        if (result := self.beacon_registry.get_landmarks_since(self.stationary_since)) is None:
            print('no eligible landmark observations')
            return None
        unscaled_landmarks, landmarks = result
        print(f"{step_num}: eligible landmark observations after distance scaling:")
        for id, landmark in landmarks.items():
            print(f'  {landmark}')

        return unscaled_landmarks, landmarks

    def geometric_pose_estimate(self, step_num: int, landmarks: dict[int, BeaconData]) -> tuple[tuple[
        float, float, float], int] | None:
        # attempt to make a pose estimates from each pair of landmark observations
        estimates = {}
        for bid1, bid2 in combinations(landmarks, 2):
            b1 = landmarks[bid1]
            b2 = landmarks[bid2]
            estimate = self.estimate_from_pair(b1, b2)
            if estimate is not None:
                estimates[(bid1, bid2)] = estimate

        # report the estimates
        for est_id, (x, y, theta) in estimates.items():
            print(f'{step_num}: estimate ({x:.2f}, {y:.2f}) {theta * 180 / pi : .0f}\u00b0 from beacons {est_id}')

        # print(f'{step_num}: landmark pose estimates: {estimates=}')

        # if no estimates could be made, return None
        nestimates = len(estimates)
        if nestimates == 0:
            print('{step_num}: pose cannot be estimated from landmark observations')
            return None

        # if only one estimate available return that with figure of merit 1
        if nestimates == 1:
            x, y, theta = est = next(iter(estimates.values()))
            # print(
            #     f'pose estimate ({x:.2f}, {y:.2f}) {theta * 180 / pi : .0f}\u00b0')
            return est, 1

        # otherwise accept both estimates from pairs whose difference in orientation is less than some threshold
        accepted_estimates = set()
        threshold = 0.5  # acceptance threshold - about 29 degrees
        for est1, est2 in combinations(estimates, 2):
            theta1 = estimates[est1][2]
            theta2 = estimates[est2][2]
            diff = angular_distance(theta1, theta2)
            if diff < threshold:
                print(f'{step_num}: accepting {est1} and {est2} difference {diff * 180 / pi : .1f}\u00b0')
                accepted_estimates.add(estimates[est1])
                accepted_estimates.add(estimates[est2])

        # if there are no acceptable estimates return None
        naccepted = len(accepted_estimates)
        if naccepted == 0:
            print(f'no acceptable pose estimates from these landmark observations')
            return None

        # report the accepted estimates
        for x, y, theta in accepted_estimates:
            print(f'{step_num}: contributing estimate ({x:.2f}, {y:.2f}) {theta * 180 / pi : .0f}\u00b0')

        # average the accepted estimates
        xs, ys, thetas = zip(*accepted_estimates)
        xm = sum(xs) / naccepted
        ym = sum(ys) / naccepted
        thetam = average_angle(thetas)
        print(f'{step_num}: mean pose estimate ({xm:.2f}, {ym:.2f}) {thetam * 180 / pi : .0f}\u00b0 merit {naccepted}')

        return (xm, ym, thetam), naccepted

    def estimate_from_pair(self, b1, b2):
        p = None
        if b1 is not None and b2 is not None:
            h1, d1 = b1.relpos
            xb1, yb1 = b1.abspos
            h2, d2 = b2.relpos
            xb2, yb2 = b2.abspos
            p = self.geom_estimate_pose(d1, h1, xb1, yb1, d2, h2, xb2, yb2)
        return p
