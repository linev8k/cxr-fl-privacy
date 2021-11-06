
"""Create CSV files of random subset from Mendeley data. Patient overlap cannot occur.
May also be applied to CheXpert data, if file contains only unique patients."""


import json
import argparse
import pandas as pd
import random

from utils import check_path

client_str = 'client46'
#ADJUST THIS
ORIG_CSV = f'CheXpert-v1.0-small/new_clients/{client_str}/client_train.csv' #name of csv to split
SUB_DIR = f'CheXpert-v1.0-small/new_clients/{client_str}/' #subdirectory within data path in which to save csv files
CSV_NAMES = ['client_train.csv', 'client_val.csv', 'client_test.csv'] #filenames for saving
CSV_NAMES = [SUB_DIR + csv_name for csv_name in CSV_NAMES]
SPLIT_NUM = [10,10,10] #fraction of original data, or number of images
random_seed = 208
split_by = 'int' # 'perc' or 'int'; whether number or percentage is specified

filter_by = 'frontal' # only relevant for CheXpert data

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

    # exclude_df_1 = pd.read_csv(SUB_DIR+'client37.csv')
    # exclude_df_2 = pd.read_csv(SUB_DIR+'client38.csv')
    # exclude_df_3 = pd.read_csv(SUB_DIR+'client39.csv')
    # exclude_df_4 = pd.read_csv(SUB_DIR+'client40.csv')
    #
    # data = pd.concat([data,exclude_df_1]).drop_duplicates(keep=False)
    # data = pd.concat([data,exclude_df_2]).drop_duplicates(keep=False)
    # data = pd.concat([data,exclude_df_3]).drop_duplicates(keep=False)
    # data = pd.concat([data,exclude_df_4]).drop_duplicates(keep=False)

    if filter_by is not None:
        for i in range(len(data)):
            data = data[data['Path'].str.contains(filter_by)]
    # data = data[data['Age'] >70]

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
