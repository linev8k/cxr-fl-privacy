# Privacy-Preserving Classification of X-Ray Images in a Federated Learning Setting

The project explores the application of mechanisms to defend a federated learning model against reconstruction attacks and enhance privacy.

## Script Usage Examples

**Split Data**  

Split CheXpert data without patient overlap. Currently, this splits the data randomly without any sorting mechanism.  
Insert the name of the CSV with data that should be split (can be original or subset of CheXpert data), the names of the CSVs for the resulting splits, and the percentages of the splits in the script:  
```python
ORIG_CSV = 'client4.csv' #name of csv to split
SUB_DIR = 'client4/' #subdirectory within data path in which to save csv files
CSV_NAMES = ['client_train.csv', 'client_test.csv'] #filenames for saving
SPLIT_PERC = [0.8,0.2] #fractions for splitting original data
```
Run the script:  
```sh
python3 split_chexpert.py config.json
```
Specify the path where data lives and where the CSVs are saved:  
```sh
python3 split_chexpert.py config.json -d  path_to_data/
```
Similarly, split other data which does not include multiple images per patient (here, this applies to Mendeley data):  
```sh
python3 split_mendeley.py config.json -d path_to_data/ 
```
Adjust this part in the script:
```python
ORIG_CSV = 'train_mendeley.csv' #name of csv to split
SUB_DIR = 'client5/' #subdirectory within data path in which to save csv files
CSV_NAMES = ['client_test.csv'] #filenames for saving (at most 2)
CSV_NAMES = [SUB_DIR + csv_name for csv_name in CSV_NAMES]
SPLIT_PERC = 0.002 #fraction of original data
KEEP_REST = False #whether to also save other part of the data
```

**Train a network using federated learning**   
Using a GPU and verifying its availability can be turned off with ```--no_gpu``` flag:   
```sh
python3 train_FL.py config.json [--no_gpu]
```
Otherwise, specify which GPUs to use in the script (currently, using DataParallel is not very efficient and one GPU should suffice):   
```python
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
```
Add the path where results should be stored:   
```sh
python3 train_FL.py config.json -o path_to_results/
```
Specify the path where data lives:
```sh
python3 train_FL.py config.json -d path_to_data/ -df path_to_datafile/
```
For convenience, a shell script can be used (make sure all arguments are specified properly): 
```sh
sh run_fl.sh
```

**Test a model**  
Test a model on individual test data sets of clients. Usage is similar to ```train_FL.py```. Specify the model path:

```sh
python3 test_model.py config.json -m ./results/global_2rounds.pth.tar
```
By default, the test data set of each client is used. If you want to use the validation sets, do:  
```sh
python3 test_model.py config.json -m ./results/global_2rounds.pth.tar --val
```
Again, a shell script can be used for convenience:  
```sh
sh run_test.sh
```
A CSV file with the results is saved in the output path. You can change ```CSV_OUTPUT_NAME``` in the script.  
Make sure to use the correct ```config``` file, pay attention to dataset settings such as ```class_idx```, as well as ```num_clients``` and ```client_dirs```. Those should be the same as used for model training.

## Training Configuration

## Result Output

All models are saved in the directory specified with ```--output_path``` or ```-o```.  At the end of training, this directory will have the following structure:  

```
.
├── round0_client0 						# directory with results from first round, first client
|		└── 1-epoch_FL.pth.tar 		# model checkpoint per epoch
|		└── ...
|		└── round0_client0.csv 		# CSV containing client result metrics per epoch
├── round0_client1 						# first round, second client
|	  └── ...
├──	round1_client0 						# second round, first client
|		└── ...
├── ...
├── global_0rounds.pth.tar 		# checkpoint of global model per round
├── ...
├── global_validation.csv 		# CSV containing AUC for global models
├── train_results.csv 				# merged CSV summarizing individual client's metrics
```


## Acknowledgements

* [CheXpert classification](https://github.com/Stomper10/CheXpert)
* [CheXpert Data](https://stanfordmlgroup.github.io/competitions/chexpert/)
* [Mendeley X-Ray Data](https://data.mendeley.com/datasets/rscbjbr9sj/3)
