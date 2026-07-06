from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from pose_eval.analysis import analyse
from pose_eval.beacon import BeaconObservation, BeaconRegistry
from pose_eval.pose_estimator import LandmarkPoseEstimator
from pose_eval.utils.flatten import flatten
from pose_eval.utils.persist import load_data, save_data, set_base

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

    # get data common to all algorithms that should appear in the results
    common_columns = ['step_num', 'actual_x', 'actual_y', 'actual_theta']
    common_df = df.loc[df['valid_obs'], common_columns]
    # and set index to the step number
    common_df = common_df.set_index('step_num')

    dfs = []  # list of result dfs from each algorithm variant

    prefixes = ('s', 'u')
    scaled_observations = (True, False)
    for prefix, scaled_observations in zip(prefixes, scaled_observations):
        cfg = {'scaled_observations': scaled_observations}
        estimator = EstimatorRunner(prefix, cfg)
        output_df = estimator.run(df)
        dfs.append(output_df)
        # data_fname = 'sposes' if scaled_observations else 'uposes'
        # save_data(output_df, dirpath=str(save_path), data_fname=data_fname, format_spec=['csv', 'feather'], float_format=float_format)

    for output_df in dfs:
        print(output_df.head())

    # combine the results
    combined_df = pd.concat([common_df] + dfs, axis=1, join='outer')

    combined_df = combined_df.reset_index().sort_values('step_num')

    save_data(combined_df, dirpath=str(save_path), data_fname='results', format_spec=['csv', 'feather'],
              float_format=float_format)


class EstimatorRunner:
    def __init__(self, prefix: str, cfg: dict):
        self.prefix = prefix
        self.beacon_registry = BeaconRegistry()
        self.pose_estimator = LandmarkPoseEstimator(self.beacon_registry)
        self.scaled_observations = cfg['scaled_observations']

    def run(self, df: DataFrame) -> DataFrame:
        print('EstimatorRunner.run() starting with standard algorithm')
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

            geom_result = self.pose_estimator.geometric_pose_estimate(step_num, scaled_landmarks)
            if geom_result is None:
                continue

            geom_pose, merit = geom_result

            landmarks = scaled_landmarks if self.scaled_observations else unscaled_landmarks

            result = self.pose_estimator.optimize_pose(geom_pose, landmarks)

            # save any resulting pose estimate provided it is based only on valid observations
            if not d.valid_obs:
                print(f'{step_num}: *** invalid observation')

            if result is None or not d.valid_obs:
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
    prefixes = ('s', 'u')
    process(data_path=data_path, output_path=save_path, float_format=float_format)
    analyse(save_path, prefixes=prefixes, float_format=float_format)


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
    data_path = Path('vc5')
    save_path = Path('vc5')
    process(data_path=data_path, output_path=save_path, float_format='%.3f')
    analyse(save_path, prefixes=('s', 'u'), float_format='%.3f')
