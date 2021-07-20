
"""Create CSV files of random subset from Mendeley data. Patient overlap cannot occur."""


import json
import argparse
import pandas as pd
import random

from utils import check_path

#ADJUST THIS
ORIG_CSV = 'train_mendeley.csv' #name of csv to split
SUB_DIR = 'client5/' #subdirectory within data path in which to save csv files
CSV_NAMES = ['client_test.csv'] #filenames for saving (at most 2)
CSV_NAMES = [SUB_DIR + csv_name for csv_name in CSV_NAMES]
SPLIT_PERC = 0.002 #fraction of original data
KEEP_REST = False #whether to also save other part of the data


def main():

    assert len(CSV_NAMES)==1 if not KEEP_REST else len(CSV_NAMES)==2, "Number of file names is different from required number of splits"

    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('cfg_path', type = str, help = 'Path to the config file in json format.')
    #set path to chexpert data
    parser.add_argument('--data_path', '-d', dest='data_path', help='Path to data.', default='./')
    args = parser.parse_args()
    with open(args.cfg_path) as f:
        cfg = json.load(f)

    data_path = check_path(args.data_path, warn_exists=False, require_exists=True)
    random_seed = cfg['random_seed']
    random.seed(random_seed+3)

    # Read from CSV files
    data = pd.read_csv(data_path + ORIG_CSV)
    total_idx = len(data)
    subsample_idx = random.sample(list(range(total_idx)), int(total_idx*SPLIT_PERC)) # specify number of images instead of fraction
    sub_df = data.iloc[subsample_idx]

    sub_df.to_csv(data_path+CSV_NAMES[0], index=False)
    print(f"{len(sub_df)} images saved in {CSV_NAMES[0]}")

    if KEEP_REST:
        rest_df = data.drop(index=subsample_idx)
        rest_df.to_csv(data_path+CSV_NAMES[1], index=False)
        print(f"{len(rest_df)} images saved in {CSV_NAMES[1]}")

    return None



if __name__ == "__main__":
    main()
