import json
from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import combinations
from math import pi, sqrt, cos

@dataclass
class BeaconObservation:
    """Class to hold beacon observation data
    """
    data: int  # transmitted data sent with the flash
    relpos: tuple[float, float]  # estimated relative position (relative bearing, distance)
    start_step: int # timestep in which long flash start was reported
    end_step: int  # timestep in which the observation was completed
    ambient_ir: int | None = None   # ambient IR when long flash start was reported

@dataclass
class BeaconData:
    """Class holding most recent data for an individual beacon """
    ident: int  # beacon identifier
    marker: bool  # whether the beacon's marker property is set
    data: int  # data transmitted as part of the beacon flash
    relpos: tuple[float, float]  # estimated relative position (relative bearing, distance)
    pose: tuple[float, float, float]  # estimated pose at timestep when observation started
    start_step: int  # timestep when observation started, i.e. long flash first reported
    end_step: int  # timestep when the observation completed
    ambient_ir: int | None = None  # ambient IR at step when observation started
    abspos: tuple[float, float] | None = None  # absolute position if known

    def __str__(self):
        bearing = self.relpos[0] * 180 / pi
        dist = self.relpos[1]
        return f'  beacon {self.ident}: start step {self.start_step} rpos {bearing:.0f}\u00b0 {dist:.2f} marked: {self.marker}'

# TODO mutated to hard code landmark positions
class BeaconRegistry:
    """Class to hold the most recent information for each beacon"""
    id_map = {1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 2, 7: 3, 8: 4}

    def __init__(self):
        self._beacon_data: dict[int, BeaconData] = {}  # dictionary of beacon data records keyed by id
        # self.pose_estimator: PoseEstimator | None = None  # reference to pose estimator for access to pose history
        self.pose_estimator = None  # reference to pose estimator for access to pose history

        # json_spec = json.loads(cfg.get('body', 'landmark.positions'))
        # self.positions = {int(key): (x, y) for key, (x, y) in json_spec.items()}

        # absolute position of real beacons with associated data 1, 2, 3
        self.positions = {1: [0.0, 0.0], 2: [ 2.54, 0.0 ], 3: [ 1.27, 1.90 ]}  # conservatory positions
        # self.positions = {1: [1.0, 0.0], 2: [ 0, -1 ], 3: [ 0, 1 ]}            # three beacons scene

        print(f'Landmark positions: {self.positions}')

        # whether there's been a new landmark observation since the last time the client asked
        self.new_landmark_observation = False

    def get_beacon_data(self, beacon_id) -> BeaconData | None:
        return self._beacon_data.get(beacon_id)

    def get_landmarks_since(self, step: int | None) -> tuple[dict[int, BeaconData], dict[int, BeaconData]] | None:
        """Return all landmark observations since the given step if any are new since last query
        and there are at least two of them
        """
        if step is None or not self.new_landmark_observation:
            return None
        landmarks = {bid: bd for bid, bd in self._beacon_data.items()
                     if bd.start_step > step and bd.abspos is not None}
        self.new_landmark_observation = False
        if landmarks is None or len(landmarks) < 2:
            return None

        scaled_landmarks = self.scale_distances(landmarks)
        return landmarks, scaled_landmarks

    def process_observation(self, step_num: int, obs: BeaconObservation) -> BeaconData:
        # beacon data ranges from 1 to 8, and ids from 1 to 4, leaving one bit to indicate whether the marker is present
        beacon_id = BeaconRegistry.id_map[obs.data]

        updates = {
            "data": obs.data,
            "marker": obs.data > 4,
            "relpos": obs.relpos,
            # TODO mutation to check
            # "pose": self.pose_estimator.get_pose_at_step(obs.start_step),
            "pose": (0, 0, 0),
            "start_step": obs.start_step,
            "end_step": obs.end_step,
            "ambient_ir": obs.ambient_ir,
        }

        bd = self._beacon_data.get(beacon_id)

        if bd is None:
            bd = BeaconData(ident=beacon_id, abspos=self.positions.get(beacon_id), **updates)
        else:
            bd = replace(bd, **updates)
        self._beacon_data[beacon_id] = bd

        # set flag to indicate a new landmark observation has been made
        if bd.abspos is not None:
            self.new_landmark_observation = True

        print(f'{step_num}: {self}')

        return bd

    def scale_distances(self, landmarks: dict[int, BeaconData]) -> dict[int, BeaconData]:

        indiv_scales = defaultdict(list)
        # for each pair of landmarks calculate scale value required to make the estimated distance between
        # those beacons match the true value
        for bid1, bid2 in combinations(landmarks, 2):
            # get relative position estimates for a pair of landmark beacons
            b1 = landmarks[bid1]
            b2 = landmarks[bid2]
            rb1, d1 = b1.relpos
            rb2, d2 = b2.relpos
            # estimate distance between landmarks from these observations
            est_dist = sqrt(d1 ** 2 + d2 ** 2 - 2 * d1 * d2 * cos(rb1 - rb2))
            # calculate true distance
            x1, y1 = b1.abspos
            x2, y2 = b2.abspos
            true_dist = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            scale = true_dist / est_dist
            # for each beacon in the pair, remember that scale value
            indiv_scales[bid1].append(scale)
            indiv_scales[bid2].append(scale)

        # multiply each estimated beacon distance by the average of the scale values associated
        # that beacon to give the scaling value to apply to that beacon
        scaled_landmarks = {}
        for bid, bdata in landmarks.items():
            scales = indiv_scales[bid]
            indiv_scale = sum(scales) / len(scales)
            scaled_landmarks[bid] = replace(bdata, relpos=(bdata.relpos[0], bdata.relpos[1] * indiv_scale))
            contrib_scales_str = ", ".join(f'{scale:.3f}' for scale in scales)
            print(f'{bid}: scale: {indiv_scale}  ({contrib_scales_str})')

        return scaled_landmarks

    def clear(self):
        self._beacon_data.clear()

    def __str__(self):
        lines = [f"registered beacons: {len(self._beacon_data)}"]
        for b_id, bd in sorted(self._beacon_data.items()):
            rb = bd.relpos[0]
            dist = bd.relpos[1]
            if bd.abspos is not None:
                line = f'  landmark beacon {b_id}: start {bd.start_step} marker {bd.marker} rp {rb * 180 / pi:.0f}\u00b0 {dist:.2f}'
            else:
                line = f'  beacon {b_id}: start {bd.start_step} marker {bd.marker} rp {rb * 180 / pi:.0f}\u00b0 {dist:.2f}'

            lines.append(line)
        s = '\n'.join(lines)
        return s
# class BeaconRegistry:
#     """Class to hold the most recent information for each beacon"""
#     id_map = {1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 2, 7: 3, 8: 4}
#
#     def __init__(self, cfg):
#         self._beacon_data: dict[int, BeaconData] = {}  # dictionary of beacon data records keyed by id
#         # self.pose_estimator: PoseEstimator | None = None  # reference to pose estimator for access to pose history
#         self.pose_estimator = None  # reference to pose estimator for access to pose history
#
#         # absolute position of real beacons with associated data 1, 2, 3
#         json_spec = json.loads(cfg.get('body', 'landmark.positions'))
#         self.positions = {int(key): (x, y) for key, (x, y) in json_spec.items()}
#         print(f'Landmark positions: {self.positions}')
#
#         # whether there's been a new landmark observation since the last time the client asked
#         self.new_landmark_observation = False
#
#     def get_beacon_data(self, beacon_id) -> BeaconData | None:
#         return self._beacon_data.get(beacon_id)
#
#     def get_landmarks_since(self, step: int | None) -> tuple[dict[int, BeaconData], dict[int, BeaconData]] | None:
#         """Return all landmark observations since the given step if any are new since last query
#         and there are at least two of them
#         """
#         if step is None or not self.new_landmark_observation:
#             return None
#         landmarks = {bid: bd for bid, bd in self._beacon_data.items()
#                      if bd.start_step > step and bd.abspos is not None}
#         self.new_landmark_observation = False
#         if landmarks is None or len(landmarks) < 2:
#             return None
#
#         scaled_landmarks = self.scale_distances(landmarks)
#         return landmarks, scaled_landmarks
#
#     def process_observation(self, step_num: int, obs: BeaconObservation) -> BeaconData:
#         # beacon data ranges from 1 to 8, and ids from 1 to 4, leaving one bit to indicate whether the marker is present
#         beacon_id = BeaconRegistry.id_map[obs.data]
#
#         updates = {
#             "data": obs.data,
#             "marker": obs.data > 4,
#             "relpos": obs.relpos,
#             "pose": self.pose_estimator.get_pose_at_step(obs.start_step),
#             "start_step": obs.start_step,
#             "end_step": obs.end_step,
#             "ambient_ir": obs.ambient_ir,
#         }
#
#         bd = self._beacon_data.get(beacon_id)
#
#         if bd is None:
#             bd = BeaconData(ident=beacon_id, abspos=self.positions.get(beacon_id), **updates)
#         else:
#             bd = replace(bd, **updates)
#         self._beacon_data[beacon_id] = bd
#
#         # set flag to indicate a new landmark observation has been made
#         if bd.abspos is not None:
#             self.new_landmark_observation = True
#
#         print(f'{step_num}: {self}')
#
#         return bd
#
#     def scale_distances(self, landmarks: dict[int, BeaconData]) -> dict[int, BeaconData]:
#
#         indiv_scales = defaultdict(list)
#         # for each pair of landmarks calculate scale value required to make the estimated distance between
#         # those beacons match the true value
#         for bid1, bid2 in combinations(landmarks, 2):
#             # get relative position estimates for a pair of landmark beacons
#             b1 = landmarks[bid1]
#             b2 = landmarks[bid2]
#             rb1, d1 = b1.relpos
#             rb2, d2 = b2.relpos
#             # estimate distance between landmarks from these observations
#             est_dist = sqrt(d1 ** 2 + d2 ** 2 - 2 * d1 * d2 * cos(rb1 - rb2))
#             # calculate true distance
#             x1, y1 = b1.abspos
#             x2, y2 = b2.abspos
#             true_dist = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
#             scale = true_dist / est_dist
#             # for each beacon in the pair, remember that scale value
#             indiv_scales[bid1].append(scale)
#             indiv_scales[bid2].append(scale)
#
#         # multiply each estimated beacon distance by the average of the scale values associated
#         # that beacon to give the scaling value to apply to that beacon
#         scaled_landmarks = {}
#         for bid, bdata in landmarks.items():
#             scales = indiv_scales[bid]
#             indiv_scale = sum(scales) / len(scales)
#             scaled_landmarks[bid] = replace(bdata, relpos=(bdata.relpos[0], bdata.relpos[1] * indiv_scale))
#             contrib_scales_str = ", ".join(f'{scale:.3f}' for scale in scales)
#             print(f'{bid}: scale: {indiv_scale}  ({contrib_scales_str})')
#
#         return scaled_landmarks
#
#     def clear(self):
#         self._beacon_data.clear()
#
#     def __str__(self):
#         lines = [f"registered beacons: {len(self._beacon_data)}"]
#         for b_id, bd in sorted(self._beacon_data.items()):
#             rb = bd.relpos[0]
#             dist = bd.relpos[1]
#             if bd.abspos is not None:
#                 line = f'  landmark beacon {b_id}: start {bd.start_step} marker {bd.marker} rp {rb * 180 / pi:.0f}\u00b0 {dist:.2f}'
#             else:
#                 line = f'  beacon {b_id}: start {bd.start_step} marker {bd.marker} rp {rb * 180 / pi:.0f}\u00b0 {dist:.2f}'
#
#             lines.append(line)
#         s = '\n'.join(lines)
#         return s
