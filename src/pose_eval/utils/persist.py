"""Pandas dataframe persistence functions """
# import datetime

import json
import os

import pandas as pd

base_data_folder = None

csv_ext = '.csv'
feather_ext = '.feather'
DEFAULT_DFNAME = '_data'
DEFAULT_MFNAME = 'meta.json'


def set_base(base_folder):
    """Set base data folder.
    
    Args:
        base_folder: Pathname of the base folder.
    
    Results from individual sessions are stored in sub-folders of
    this folder
    """
    global base_data_folder
    if len(base_folder) == 0:
        raise ValueError('base folder required')

    base_folder = os.path.normpath(os.path.expanduser(base_folder))
    if not (os.path.isabs(base_folder) and os.path.isdir(base_folder)):
        raise ValueError('base folder must be the absolute path for an existing directory')

    base_data_folder = base_folder


def get_base():
    return base_data_folder


def load_data(dpath, *, data_fname=DEFAULT_DFNAME, md_fname=DEFAULT_MFNAME, csvargs=None):
    """Return data for the given file or directory path
    
    Args:
        dpath: absolute or relative path to save directory
        data_fname: base filename for saved data (default: _data)
        md_fname: base filename for saved metadata (default: meta.json)
        csvargs: additional arguments for Pandas read_csv()
        
    Returns:
        Primary data and metadata if path is a directory path
        Data loaded from path if path is a file path

        Metadata is returned as a JSON-compatible object generated from the meta.json file.

        Data stored as .csv or .feather files are returned as Pandas objects.

    A relative path is made absolute by prepending the base folder
    """

    if csvargs is None:
        csvargs = {}

    assert len(dpath) > 0
    dpath = os.path.normpath(os.path.expanduser(dpath))
    if not os.path.isabs(dpath):
        dpath = os.path.join(base_data_folder, dpath)

    if not os.path.isdir(dpath):
        raise FileNotFoundError(f'Data directory {dpath} not found.')

    df = load_best_format(os.path.join(dpath, data_fname), csvargs)

    md = None
    mpath = os.path.join(dpath, md_fname)
    if os.path.isfile(mpath):
        with open(mpath) as jfile:
            md = json.load(jfile)

    return (df, md) if md else df


def load_best_format(fpath, csvargs):
    if os.path.isfile(fname := fpath + feather_ext):
        data = pd.read_feather(fname)
    elif os.path.isfile(fname := fpath + csv_ext):
        data = pd.read_csv(fname, **csvargs)
    else:
        raise Exception(f'data path {fpath} not found')
    print(f'data loaded from {fname}')
    return data


def save_data(df, dirpath, *, data_fname=DEFAULT_DFNAME, metadata=None, md_fname=DEFAULT_MFNAME, format_spec='feather',
              float_format=None):
    """
    Save data to the data store.

    Args:
        df: pandas DataFrame object to persist
        dirpath: save directory, either an absolute path or a path relative to the base folder
        metadata: JSON-compatible object to be saved as metadata (optional)
        data_fname: base filename for saved data (default: _data)
        md_fname: base filename for saved metadata (default: meta.json)
        format: save format - either 'feather' or 'csv' (default: feather)
        float_format: floating point format for CSV files (default: full precision)

    A relative directory path is made absolute by prepending the base folder
    """
    assert isinstance(df, pd.DataFrame), 'can only save pandas dataframes'

    # calculate the data directory path and create the directory if necessary
    assert len(dirpath) > 0
    dpath_norm = os.path.normpath(os.path.expanduser(dirpath))

    # disallow absolute paths for saving
    if os.path.isabs(dpath_norm):
        raise ValueError(f"Save directory '{dirpath}' must be a relative path, not absolute.")
    dpath = os.path.join(base_data_folder, dpath_norm)
    if not os.path.isdir(dpath):
        os.makedirs(dpath)
        print(f"created new save directory {dpath}")

    # save metadata as a JSON file, if provided
    if metadata is not None:
        mdpath = os.path.join(dpath, md_fname)
        with open(mdpath, 'w') as f:
            json.dump(metadata, f, indent=4, sort_keys=False)
            print(f'saved metadata to {mdpath}')

    fpath = os.path.join(dpath, data_fname)

    # format argument can either be a string or a list or tuple of strings
    formats = format_spec if isinstance(format_spec, (list, tuple)) else (format_spec,)

    for format in formats:
        if format == 'csv':
            path = fpath + csv_ext
            df.to_csv(path, index=False, float_format=float_format)
        elif format == 'feather':
            path = fpath + feather_ext
            df.to_feather(path)
        else:
            raise ValueError(f'unsupported format {format}')

        print(f'saved data to {path}')


if __name__ == "__main__":
    # Test script only below here

    def test():

        sess_name = r'cabbage\persist_test'

        base = r'~/temp/data/persist'

        set_base(base)

        md = {'name': 'john', 'age': 52}
        df = pd.DataFrame([[1, 2, 3], [3, 2, 1]], columns=['a', 'b', 'c'])

        # data and metadata saved in default feather 2 format
        dpath = sess_name + '1'
        save_data(df, dpath, metadata=md)
        df_loaded, md_loaded = load_data(dpath)
        print(f'metadata: {md_loaded}')
        print(f'data: {df_loaded}')
        assert df.equals(df_loaded)

        dpath = sess_name + '2'
        save_data(df, dpath)
        df_loaded = load_data(dpath)
        print(f'data: {df_loaded}')
        assert df.equals(df_loaded)

        dpath = sess_name + '3'
        try:
            save_data("hello", dpath, metadata=md)
        except Exception as exc:
            print(exc)

        dpath = 'c:/temp/bad'
        try:
            save_data(df, dpath, metadata=md)
        except Exception as exc:
            print(exc)

        # save in CSV format
        dpath = sess_name + '4'
        save_data(df, dpath, metadata=md)
        md_loaded, df_loaded = load_data(dpath)
        # assert df.equals(df_loaded)
        print(f'metadata: {md_loaded}')
        print(f'{df_loaded}')

        # save in multiple formats
        dpath = sess_name + '5'
        save_data(df, dpath, metadata=md)
        save_data(df, dpath)
        df_loaded, md_loaded = load_data(dpath)
        assert df.equals(df_loaded)
        print(f'metadata: {md_loaded}')
        print(f'{df_loaded}')

        # save in multiple files
        df2 = pd.DataFrame([[9, 8, 7], [3, 2, 1]], columns=['z', 'y', 'x'])

        dpath = sess_name + '6'
        save_data(df, dpath, metadata=md)
        save_data(df2, dpath, data_fname='data2')
        df_loaded, md_loaded = load_data(dpath)
        # assert df.equals(df_loaded)
        print(f'metadata: {md_loaded}')
        print(f'{df_loaded}')

        df_loaded, md_loaded = load_data(dpath, data_fname='data2')
        assert df2.equals(df_loaded)
        print(f'metadata: {md_loaded}')
        print(f'{df_loaded}')


    test()
