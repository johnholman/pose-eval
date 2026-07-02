from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from pose_eval.beacon import BeaconObservation, BeaconRegistry
from pose_eval.pose_estimator import LandmarkPoseEstimator
from pose_eval.utils.flatten import flatten
from pose_eval.utils.persist import set_base, load_data, save_data

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', '{:.2f}'.format)


def process(data_path: str | Path, output_path: str | Path, float_format: str | None = None):
    """
    """
    base_path = Path(r'~/Google Drive/data/pose-eval').expanduser().resolve()
    set_base(str(base_path))

    load_path = Path('robocap') / data_path
    save_path = Path('expts') / output_path
    print(f"processing data from {base_path / load_path} saving in {base_path / save_path}")

    collected_df, md = load_data(str(load_path), data_fname='robot')
    print(collected_df.head())

    # remove any rows without a beacon estimate
    df = collected_df.dropna(subset=['b_data'])
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
        # save any resulting pose estimate provided it is based only on valid observations
        if result is not None and d.valid_obs:
            print(f'{step_num}: result: {result}')
            flattened_result = flatten(result)
            print(f'{step_num}: flattened result: {flattened_result}')
            true_pose = { 'true_x': d.actual_x, 'true_y': d.actual_y, 'true_theta': d.actual_theta }
            output_row = {'step_num': step_num } | true_pose | flattened_result
            print(f'{step_num}: output_row: {output_row}')
            output_rows.append(output_row)
        else:
            print(f'{step_num}: three-landmark pose not made')

    output_df = DataFrame(output_rows)
    print(output_df.head())
    save_data(output_df, dirpath=str(save_path), data_fname='poses', format_spec=['csv', 'feather'], float_format=float_format)

def analyse(results_path: str | Path, float_format: str | None  = None):
    base_path = Path(r'~/Google Drive/data/pose-eval').expanduser().resolve()
    set_base(str(base_path))

    save_path = load_path = Path('expts') / results_path
    print(f"analysing results from {base_path / load_path}")

    df = load_data(str(load_path), data_fname='poses')

    # remove rows without true pose data as these may be based on stale observations
    df = df.dropna(subset=['true_x'])
    print(df.head())

    true = df.loc[:, ['true_x', 'true_y', 'true_theta']].to_numpy()
    scaled = df.loc[:, ['s_pose_x', 's_pose_y', 's_pose_theta']].to_numpy()
    unscaled = df.loc[:, ['u_pose_x', 'u_pose_y', 'u_pose_theta']].to_numpy()
    # true = df.loc[:, ['true_x', 'true_y', 'true_theta']].to_numpy()
    # scaled = df.loc[:, ['scaled_pose_x', 'scaled_pose_y', 'scaled_pose_theta']].to_numpy()
    # unscaled = df.loc[:, ['unscaled_pose_x', 'unscaled_pose_y', 'unscaled_pose_theta']].to_numpy()

    print(true)
    print(scaled)

    scaled_se = pose_difference(true, scaled) ** 2
    print(f'scaled_se: {scaled_se}')
    scaled_sse = scaled_se.sum(axis=1)  # sum of squared errors for each scaled result
    print(f'scaled_sse: {scaled_sse}')
    df.loc[:, 'scaled_sse'] = scaled_sse
    print(df)

    unscaled_se = pose_difference(true, unscaled) ** 2
    # unscaled_se = (true - unscaled) ** 2
    unscaled_sse = unscaled_se.sum(axis=1)  # sum of squared errors for each unscaled result
    print(f'{unscaled_sse=}')
    df.loc[:, 'unscaled_sse'] = unscaled_sse
    print(df)

    save_data(df, str(save_path), data_fname='analysis', format_spec=['csv', 'feather'], float_format=float_format)



def pose_difference(pose_array1, pose_array2, degrees=False):
    """Calculates the difference between two N x 3 arrays of poses (p2 - p1).

    Each row is expected to be (x, y, theta).
    Angular difference is correctly wrapped to [-pi, pi]
    """
    diff = pose_array2 - pose_array1

    theta_diff = diff[:, 2]

    # Standard radian wrapping using atan2(sin(x), cos(x))
    wrapped_theta = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))

    # 3. Overwrite the raw angular difference with the wrapped one
    diff[:, 2] = wrapped_theta

    return diff

def main(data_path, save_path, float_format=None):
    process(data_path=data_path, output_path=save_path, float_format=float_format)
    analyse(save_path, float_format=float_format)


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

    main(data_path=data_path, save_path=output_path, float_format=float_format)


if __name__ == '__main__':
    data_path = Path('vc1')
    save_path = Path('vc1')
    process(data_path=data_path, output_path=save_path)
    analyse(save_path)
