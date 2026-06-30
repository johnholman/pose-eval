from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from pose_eval.beacon import BeaconObservation, BeaconRegistry
from pose_eval.utils.persist import set_base, load_data
from pose_eval.pose_estimator import LandmarkPoseEstimator

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', '{:.2f}'.format)


def process(data_path: str | Path):
    """
    """
    base_path = Path(r'~/Google Drive/data/robot/pose').expanduser().resolve()
    set_base(str(base_path))

    load_path = Path(data_path)
    print(f"processing {data_path}")

    collected_df, md = load_data(str(load_path), data_fname='robot')
    print(collected_df.head())

    # remove rows without beacon data (triggered e.g. by landmark pose invalidation messages)
    df = collected_df.dropna(subset=['b_data'])

    beacon_registry = BeaconRegistry()
    pose_estimator = LandmarkPoseEstimator(beacon_registry)

    for d in df.itertuples():
        step_num = d.step_num
        beacon_obs = BeaconObservation(data=d.b_data, start_step=d.b_start, end_step=d.b_end,
                                       relpos=(d.b_relb, d.b_dist))
        print(f'{step_num}: {beacon_obs}')
        beacon_data = beacon_registry.process_observation(step_num, beacon_obs)
        print(beacon_data)
        result = pose_estimator.estimate_pose(step_num, wheel_positions=(0, 0))
        if result is not None:
            pose, loss = result
            print(f'{step_num}: {pose=}, {loss=}')
        else:
            print(f'{step_num}: no pose')


def cli():
    parser = ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument('data_path', type=Path,
                        help="relative directory path to data (e.g. sess1/trial3)")

    # parser.add_argument('-o', '--output', type=Path, required=True,
    #                     metavar="DEST", help="output dataset relative path")

    # parser.add_argument('-d', '--decimal-places', type=int, default=3,
    #                     help='number of decimal places for CSV float fields (default: 3)')

    args = parser.parse_args()
    data_path = args.data_path
    # dataset_path = args.output
    # float_format = f'%.{args.decimal_places}f'

    process(data_path=data_path)


if __name__ == '__main__':
    expt = Path('gather/vc1')
    process(expt)
