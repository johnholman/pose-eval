from pathlib import Path

import numpy as np
import pandas as pd

from pose_eval.utils.persist import load_data, save_data, set_base


def analyse(results_path: str | Path, input_types, start_types, float_format: str | None = None):
    base_path = Path(r'~/Google Drive/data/pose-eval').expanduser().resolve()
    set_base(str(base_path))

    save_path = load_path = Path('expts') / results_path
    print(f"analysing results from {base_path / load_path}")

    df = load_data(str(load_path), data_fname='results')
    ntrials = len(df)

    # filter out estimates with loss > 0.25 or iters > 5

    # add error columns for each algorithm
    prefixes = []
    for input_type in input_types:
        for start_type in start_types:
            prefix = f'{start_type}_{input_type}'
            # dff = df[(df[f'{prefix}_loss'] <= 0.25) & (df[f'{prefix}_iters'] <= 5)]

            df = add_errors(df, prefix)
            prefixes.append(prefix)

    save_data(df, str(save_path), data_fname='analysis', format_spec=['csv', 'feather'], float_format=float_format)

    summary_df = summary_report(df, prefixes, ntrials)
    print(summary_df)


def add_errors(df, prefix):
    """ Add distance error, angular error and combined error for each pose estimate"""

    # euclidan distance for the position error
    df[f'{prefix}_pos_err'] = np.sqrt(
        (df[f'{prefix}_x'] - df['actual_x']) ** 2 +
        (df[f'{prefix}_y'] - df['actual_y']) ** 2
    )

    # angular error wrapped to [-pi, pi]
    angle_diff = df[f'{prefix}_theta'] - df['actual_theta']
    df[f'{prefix}_theta_err'] = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

    # 1:1 normalized figure of merit
    df[f'{prefix}_err'] = np.sqrt(0.5 * df[f'{prefix}_pos_err'] ** 2 + df[f'{prefix}_theta_err'] ** 2)

    return df


def summary_report(df, prefixes, ntrials):
    summary_data = []
    for prefix in prefixes:
        mean_pos = df[f'{prefix}_pos_err'].mean()
        rmse_theta = np.sqrt((df[f'{prefix}_theta_err'] ** 2).mean())
        # mean_err = df[f'{prefix}_err'].mean()
        mean_err = np.sqrt((df[f'{prefix}_err'] ** 2).mean())
        success_pc = (df[f'{prefix}_err'].count()*100)/ntrials

        summary_data.append({
            'Variant': prefix,
            'Success rate %': success_pc,
            'Position Error (Mean m)': round(mean_pos, 3),
            'Orientation Error (RMSE rad)': round(rmse_theta, 3),
            'Orientation Error (RMSE deg)': round(np.degrees(rmse_theta), 1),
            'Merit (Mean)': round(mean_err, 3)
        })
    return pd.DataFrame(summary_data)
