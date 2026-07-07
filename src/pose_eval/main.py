from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from pose_eval.analysis import analyse
from pose_eval.beacon import BeaconObservation, BeaconRegistry
from pose_eval.pose_estimator import LandmarkPoseEstimator
from pose_eval.utils.persist import load_data, save_data, set_base

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', '{:.2f}'.format)


def process(data_path: str | Path, output_path: str | Path, input_types, start_types, float_format: str | None = None):
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

    # get data common to all algorithms that should appear in the results
    common_columns = ['step_num', 'actual_x', 'actual_y', 'actual_theta']
    common_df = df.loc[df['valid_obs'], common_columns]
    # and set index to the step number
    common_df = common_df.set_index('step_num')

    # # get positions of the landmark beacons from the step data
    # # 1. Get the unique true positions of your 3 beacons from the dataset
    # # Drop duplicates based on beacon ID to get a clean lookup table
    # beacon_lookup = df[['b_id', 'b_x', 'b_y']].drop_duplicates(subset=['b_id'])
    # print(beacon_lookup)
    # centroid = beacon_lookup[['b_x', 'b_y']].mean(axis=0).to_numpy()
    #
    # # This gives you your static starting pose: [mean_x, mean_y, 0.0]
    # centroid_initial_pose = np.array([centroid[0], centroid[1], 0.0])

    dfs = []  # list of result dfs from each algorithm variant

    # for prefix, scaled_observations in zip(prefixes, scaled_observations):
    for input_type in input_types:
        for start_type in start_types:
            prefix = start_type + '_' + input_type

            cfg = {'prefix': prefix, 'input_type': input_type, 'start_type': start_type}
            estimator = EstimatorRunner(cfg)
            output_df = estimator.run(df)
            dfs.append(output_df)

    for output_df in dfs:
        print(output_df.head())

    # combine the results
    combined_df = pd.concat([common_df] + dfs, axis=1, join='outer')

    combined_df = combined_df.reset_index().sort_values('step_num')

    save_data(combined_df, dirpath=str(save_path), data_fname='results', format_spec=['csv', 'feather'],
              float_format=float_format)


class EstimatorRunner:
    def __init__(self, cfg: dict):
        self.beacon_registry = BeaconRegistry()
        self.pose_estimator = LandmarkPoseEstimator(self.beacon_registry)
        self.prefix = cfg['prefix']
        self.input_type = cfg['input_type']
        self.start_type = cfg['start_type']

    def run(self, df: DataFrame) -> DataFrame:
        print(f'EstimatorRunner.run() algorithm {self.prefix}')

        # calculate centroid of beacons in case need it later
        beacon_lookup = df[['b_id', 'b_x', 'b_y']].drop_duplicates(subset=['b_id'])
        if (n := len(beacon_lookup)) != 3:
            print(f'{n} landmark beacons found - only 3 supported')
            exit(1)

        centroid = beacon_lookup[['b_x', 'b_y']].mean(axis=0).to_numpy()

        output_rows = []  # output as list of row dictionaries
        for d in df.itertuples():
            step_num = d.step_num
            beacon_obs = BeaconObservation(data=d.b_data, start_step=d.b_start, end_step=d.b_end,
                                           relpos=(d.b_relb, d.b_dist))
            print(f'{step_num}: {beacon_obs}')
            beacon_data = self.beacon_registry.process_observation(step_num, beacon_obs)
            print(beacon_data)

            # get most recent landmark observations made in this stationary period
            observations = self.pose_estimator.stationary_observations(step_num, wheel_positions=(0.0, 0.0))
            if observations is None:
                continue

            unscaled_landmarks, scaled_landmarks = observations
            if len(unscaled_landmarks) < 3:
                continue

            if self.start_type in ('sg', 'ug'):
                # if start pose for optimisation uses the geometry estimate, make it as required
                landmarks = scaled_landmarks if self.start_type == 'sg' else unscaled_landmarks
                geom_result = self.pose_estimator.geometric_pose_estimate(step_num, landmarks)
                if geom_result is None:
                    continue
                start_pose = geom_result[0]
            elif self.start_type == 'c':
                # set start position to centroid of the landmarks and orientation to zero
                start_pose = np.array([centroid[0], centroid[1], 0.0])
            else:
                print(f'invalid start type {self.start_type}')
                exit(1)

            if self.input_type == 's':
                landmarks = scaled_landmarks
            elif self.input_type == 'u':
                landmarks = unscaled_landmarks
            else:
                print(f'invalid input type {self.input_type}')
                exit(1)

            result = self.pose_estimator.optimize_pose(start_pose, landmarks)

            # only save pose estimates based on valid observations
            if not d.valid_obs:
                print(f'{step_num}: *** invalid observation')
                continue

            # discard any estimates with high optimization loss of number of iterations
            if result is None or result['loss'] > 0.25 or result['iters'] > 5:
            # if result is None or not d.valid_obs:
                print(f'{step_num}: three-landmark pose not made')
                continue

            # add the algorithm prefix to each column label
            result = {self.prefix + '_' + key: value for key, value in result.items()}

            # add the step number
            result |= {'step_num': step_num}
            print(f'{step_num}: result: {result}')

            # flattened_result = flatten(result)
            # print(f'{step_num}: flattened result: {flattened_result}')
            # true_pose = {'true_x': d.actual_x, 'true_y': d.actual_y, 'true_theta': d.actual_theta}
            # output_row = {'step_num': step_num} | flattened_result
            # print(f'{step_num}: output_row: {output_row}')
            output_rows.append(result)

        output_df = DataFrame(output_rows).set_index('step_num')
        return output_df


def main(data_path, save_path, float_format=None):
    input_types = ('s', 'u')
    start_types = ('c', 'sg', 'ug')
    process(data_path=data_path, output_path=save_path, input_types=input_types, start_types=start_types,
            float_format='%.3f')
    analyse(save_path, input_types=input_types, start_types=start_types, float_format='%.3f')

    # prefixes = ('s', 'u')
    # process(data_path=data_path, output_path=save_path, float_format=float_format)
    # analyse(save_path, prefixes=prefixes, float_format=float_format)


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
    data_path = Path('c2')
    save_path = Path('c2filter')
    input_types = ('s', 'u')
    start_types = ('c', 'sg', 'ug')
    process(data_path=data_path, output_path=save_path, input_types=input_types, start_types=start_types,
            float_format='%.3f')
    analyse(save_path, input_types=input_types, start_types=start_types, float_format='%.3f')
