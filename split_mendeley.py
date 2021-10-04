
"""Create CSV files of random subset from Mendeley data. Patient overlap cannot occur."""


import json
import argparse
import pandas as pd
import random

from utils import check_path

client_str = 'client15'
#ADJUST THIS
ORIG_CSV = f'mendeley_clients/{client_str}/{client_str}.csv' #name of csv to split
SUB_DIR = f'mendeley_clients/{client_str}/' #subdirectory within data path in which to save csv files
CSV_NAMES = ['client_train.csv', 'client_val.csv', 'client_test.csv'] #filenames for saving
CSV_NAMES = [SUB_DIR + csv_name for csv_name in CSV_NAMES]
SPLIT_NUM = [4,3,3] #fraction of original data, or number of images
random_seed = 208
split_by = 'int' # 'perc' or 'int'; whether number or percentage is specified

def main():

    assert len(CSV_NAMES)==len(SPLIT_NUM), "Number of splits must match number of file names"

    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #set path to chexpert data
    parser.add_argument('--data_path', '-d', dest='data_path', help='Path to data.', default='./')
    args = parser.parse_args()

    data_path = check_path(args.data_path, warn_exists=False, require_exists=True)
    random.seed(random_seed)

    # Read from CSV files
    data = pd.read_csv(data_path + ORIG_CSV)
    total_idx = len(data)
    idx_list = list(range(total_idx))
    random.shuffle(idx_list)

    start = 0
    if split_by == 'int':
        for csv_name, i in zip(CSV_NAMES, SPLIT_NUM):
            sub_df = data.iloc[idx_list[start:start+i]]
            sub_df.to_csv(data_path+csv_name, index=False)
            print(f"{len(sub_df)} images saved in {csv_name}")
            start += i

    if split_by == 'perc':
        for csv_name, i in zip(CSV_NAMES, SPLIT_NUM):
            split_int = int(total_idx*i)
            sub_df = data.iloc[idx_list[start:start+split_int]]
            sub_df.to_csv(data_path+csv_name, index=False)
            print(f"{len(sub_df)} images saved in {csv_name}")
            start += split_int


    return None



if __name__ == "__main__":
    main()
