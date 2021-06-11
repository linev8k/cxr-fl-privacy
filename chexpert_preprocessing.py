#https://github.com/Stomper10/CheXpert/blob/master/run_preprocessing.py

"""Create CSV files of train, validation, test split"""

# specify options in config.json file
# front_lat: can be frontal, lateral, both

import json
import argparse
import pandas as pd

from utils import check_path


def main():

    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('cfg_path', type = str, help = 'Path to the config file in json format.')
    #set path to chexpert data
    parser.add_argument('--chexpert', '-d', dest='chexpert_path', help='Path to CheXpert data.', default='./')
    args = parser.parse_args()
    with open(args.cfg_path) as f:
        cfg = json.load(f)

    data_path = check_path(args.chexpert_path, warn_exists=False, require_exists=True)

    # Read from CSV files
    Traindata = pd.read_csv(data_path+'CheXpert-v1.0-small/train.csv')
    Testdata = Traindata.head(500) # use first 500 training data as test data (obs ratio is almost same!)
    Traindata = Traindata[500:2000]
    # Traindata = Traindata[1:4] #toy example for testing

    Validdata = pd.read_csv(data_path+'CheXpert-v1.0-small/valid.csv')
    # Validdata = Validdata[1:3] #toy example for testing

    # Testdata = Validdata #use validation data for testing in this toy example, just to check the processing

    if cfg['front_lat'] != 'both':
        #use either only frontal or lateral images
        print(f"Only using {cfg['front_lat']} images")

        Traindata = Traindata[Traindata['Path'].str.contains(cfg['front_lat'])] # use only frontal or lateral images
        Validdata = Validdata[Validdata['Path'].str.contains(cfg['front_lat'])]

    else:
        print("Using both frontal and lateral images")

    #create CSVs
    Traindata.to_csv(data_path+'CheXpert-v1.0-small/train_mod.csv', index = False)
    print(f"Train data length:", len(Traindata))
    Validdata.to_csv(data_path+'CheXpert-v1.0-small/valid_mod.csv', index = False)
    print(f"Valid data length:", len(Validdata))
    Testdata = Validdata #for testing
    Testdata.to_csv(data_path+'CheXpert-v1.0-small/test_mod.csv', index = False)
    print("Test data length:", len(Testdata))




if __name__ == "__main__":
    main()
