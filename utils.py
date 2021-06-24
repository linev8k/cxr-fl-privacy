"""Utility methods"""

import os
import pandas as pd


def check_path(path, warn_exists=True, require_exists=False):

    """Check path to directory.
        warn_exists: Warn and require validation by user to use the specified path if it already exists.
        require_exists: Abort if the path does not exist. """

    if path[-1] != '/':
        path = path + '/'

    create_path = True

    if os.path.exists(path):
        create_path = False
        if warn_exists:
            replace = ''
            while replace not in ['y', 'n']:
                replace = input(f"Path {path} already exists. Files may be replaced. Continue? (y/n): ")
                if replace == 'y':
                    pass
                elif replace == 'n':
                    exit('Aborting, run again with a different path.')
                else:
                    print("Invalid input")


    if require_exists:
        if not os.path.exists(path):
            exit(f"{path} does not exist. Aborting")

    if create_path:
        os.mkdir(path)
        print(f"Created {path}")

    return path

def merge_eval_csv(result_path, out_file='train_results.csv'):

    """Create a merged CSV from CSVs in round-client-subdirectories for central storage of training results.
    Assumes CSVs to be named like 'round{n}_client{n}.csv'. Returns the merged dataframe and saves it as CSV.
    Args:
        result_path: Absolute path to where subdirectories with training results are located.
        out_file: Name of CSV file with merged results. Will be stored in result_path."""

    # change path if necessary
    result_path = os.path.abspath(result_path)
    if  result_path != os.getcwd():
        os.chdir(result_path)

    out_path = result_path+'/'+out_file

    result_df = pd.DataFrame()

    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.csv'):

                cur_csv_path = os.path.realpath(os.path.join(root,file)) # whole path to csv
                cur_csv = cur_csv_path.split("/")[-1] # name of csv
                print(f'Reading {cur_csv}')

                cur_df = pd.read_csv(cur_csv_path)

                # extract round and client info from csv name
                n_round = cur_csv[5]
                n_client = cur_csv[13]
                try:
                    cur_df.insert(0,'round',n_round)
                    cur_df.insert(1,'client',n_client)
                except ValueError: # exit with error if a merged file is detected
                    print(f'{file} seems to already be a merged file. Delete or move to be able to create a new file.')
                    raise

                result_df = result_df.append(cur_df)

    result_df.reset_index(inplace=True, drop=True)


    result_df.to_csv(out_file, na_rep='nan')
    print(f"Merged CSV saved in {out_path}")

    return result_df
