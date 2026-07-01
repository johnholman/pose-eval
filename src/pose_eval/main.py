from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from pose_eval.beacon import BeaconObservation, BeaconRegistry
from pose_eval.utils.flatten import flatten
from pose_eval.utils.persist import set_base, load_data, save_data
from pose_eval.pose_estimator import LandmarkPoseEstimator

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', '{:.2f}'.format)


def process(data_path: str | Path, output_path: str | Path, ff: str = None):
    """
    """
    base_path = Path(r'~/Google Drive/data/pose-eval').expanduser().resolve()
    set_base(str(base_path))

    load_path = Path('robocap') / data_path
    save_path = Path('expts') / output_path
    print(f"processing data from {base_path / load_path} saving in {base_path / save_path}")

    collected_df, md = load_data(str(load_path), data_fname='robot')
    print(collected_df.head())

    # remove rows without a validated pose estimate as indicated by presence of true data
    df = collected_df.dropna(subset=['actual_x'])
    print(df)

    beacon_registry = BeaconRegistry()
    pose_estimator = LandmarkPoseEstimator(beacon_registry)

    output_rows = []  # output as list of row dictionaries
    for d in df.itertuples():
        step_num = d.step_num
        beacon_obs = BeaconObservation(data=d.b_data, start_step=d.b_start, end_step=d.b_end,
                                       relpos=(d.b_relb, d.b_dist))
        print(f'{step_num}: {beacon_obs}')
        beacon_data = beacon_registry.process_observation(step_num, beacon_obs)
        print(beacon_data)
        result = pose_estimator.estimate_pose(step_num, wheel_positions=(0, 0))
        if result is not None:
            print(f'{step_num}: result: {result}')
            # flattened_result = flatten(result)
            # print(f'{step_num}: flattened result: {flattened_result}')
            # true_pose = { 'true_x': d.actual_x, 'true_y': d.actual_y, 'true_theta': d.actual_theta }
            # output_row = {'step_num': step_num } | true_pose | flattened_result
            # output_rows.append(output_row)
        else:
            print(f'{step_num}: no pose')

    output_df = DataFrame(output_rows)
    print(output_df.head())
    save_data(output_df, dirpath=str(save_path), data_fname='poses', format_spec=['csv', 'feather'])


def cli():
    parser = ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument('-d', '--data_path', type=Path, required=True,
                        help="relative directory path to captured data (e.g. c1)")

    parser.add_argument('-o', '--output', type=Path, required=True,
                        metavar="DEST", help="output directory")

    parser.add_argument('-f', '--decimal-places', type=int, default=3,
                        help='number of decimal places for CSV float fields (default: 3)')

    args = parser.parse_args()
    data_path = Path(args.data_path)
    output_path = Path(args.output)
    float_format = f'%.{args.decimal_places}f'

    process(data_path=data_path, output_path=output_path, ff=float_format)


if __name__ == '__main__':
    expt = Path('gather/vc1')
    process(expt)
