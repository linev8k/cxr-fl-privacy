#https://github.com/Stomper10/CheXpert/blob/master/run_preprocessing.py

"""Create CSV files of specified splits from some CheXpert CSV.
There is no patient overlap created."""

# specify options in config.json file
# front_lat: can be frontal, lateral, both

import json
import argparse
import pandas as pd
import random

from utils import check_path

#ADJUST THIS
ORIG_CSV = 'train.csv' #name of csv to split
SUB_DIR = './' #subdirectory within data path in which to save csv files
CSV_NAMES = ['train_mod.csv', 'test_mod.csv'] #filenames for saving
CSV_NAMES = [SUB_DIR + csv_name for csv_name in CSV_NAMES]
SPLIT_PERC = [0.8,0.2] #fractions for splitting original data


def main():

    assert len(CSV_NAMES)==len(SPLIT_PERC), "Different number of file names provided than was splitted"

    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('cfg_path', type = str, help = 'Path to the config file in json format.')
    #set path to chexpert data
    parser.add_argument('--chexpert', '-d', dest='chexpert_path', help='Path to CheXpert data.', default='./')
    args = parser.parse_args()
    with open(args.cfg_path) as f:
        cfg = json.load(f)

    data_path = check_path(args.chexpert_path, warn_exists=False, require_exists=True)
    random_seed = cfg['random_seed']

    # Read from CSV files
    chex_data = pd.read_csv(data_path+'CheXpert-v1.0-small/' + ORIG_CSV)
    split_data = split_df(chex_data, random_seed=random_seed, split_perc=SPLIT_PERC)


    # Testdata = Traindata.head(500) # use first 500 training data as test data (obs ratio is almost same!)
    # Traindata = Traindata[500:2000]
    # Traindata = Traindata[1:4] #toy example for testing

    # Validdata = pd.read_csv(data_path+'CheXpert-v1.0-small/valid.csv')
    # Validdata = Validdata[1:3] #toy example for testing

    # Testdata = Validdata #use validation data for testing in this toy example, just to check the processing

    if cfg['front_lat'] != 'both':
        #use either only frontal or lateral images
        print(f"Only using {cfg['front_lat']} images")

        for i in range(len(split_data)):
            split_data[i] = split_data[i][split_data[i]['Path'].str.contains(cfg['front_lat'])] # use only frontal or lateral images
    else:
        print("Using both frontal and lateral images")

    #create CSVs
    for i in range(len(SPLIT_PERC)):
        split_data[i].to_csv(data_path+'CheXpert-v1.0-small/'+CSV_NAMES[i], index = False)

    # print(f"Train data length:", len(Traindata))
    # Validdata.to_csv(data_path+'CheXpert-v1.0-small/valid_mod.csv', index = False)
    # print(f"Valid data length:", len(Validdata))
    # # Testdata = Validdata #for testing
    # Testdata.to_csv(data_path+'CheXpert-v1.0-small/test_mod.csv', index = False)
    # print("Test data length:", len(Testdata))

    print("Modified CSVs saved")


def unique_patients_list(chex_df, random_seed):

    """Return a list of paths leading to unique patients.
    Takes a pandas dataframe as input, as read from an original CheXpert data CSV file.
    Set random seed of random module for reproducibility."""

    #process paths to cut after patient numbers
    filenames = chex_df['Path']
    patient_paths = filenames.str.split('/')
    patient_paths = patient_paths.apply(lambda x: '/'.join(x[:3]))

    #get unique patients list and shuffle
    unique_patients = list(set(patient_paths))
    unique_patients.sort() # because set inserts unwanted randomness
    random.seed(random_seed)
    random.shuffle(unique_patients)

    return unique_patients

def split_df(chex_df, random_seed, split_perc):

    """Take original CheXpert dataframe read from CSV, split it into a number of subsets.
    split_perc is a list specifying the fraction of each split and should sum to one.
    Returns a list with corresponding number of dataframes containing all original information.
    Currently, patients are shuffled with a seed and then chunked into respective splits in the given order.
    """
    assert round(sum(split_perc),3) == 1, "Split fractions don't sum to one"

    #get list with unique patient paths, shuffled
    unique_patients = unique_patients_list(chex_df, random_seed=random_seed)
    num_patients = len(unique_patients)

    num_split = []
    patient_splits = []
    start = 0
    end = 0
    for perc in split_perc:
        #calculate number of samples per patient split
        cur_sample_size = int(num_patients*perc)
        num_split.append(cur_sample_size)
        end += cur_sample_size
        patient_splits.append(unique_patients[start:end]) #split patients
        start += cur_sample_size

    assert sum([len(split) for split in patient_splits]) == len(unique_patients)
    print(f"Number of patients in splits: {num_split}")

    #create dataframe subsets of original dataframe
    split_dfs = []
    for split in patient_splits:
        split_dfs.append(chex_df[chex_df['Path'].str.split('/').apply(lambda x: '/'.join(x[:3])).isin(split)])
    print(f"Number of images in splits: {[len(df) for df in split_dfs]}")

    return split_dfs



if __name__ == "__main__":
    main()
