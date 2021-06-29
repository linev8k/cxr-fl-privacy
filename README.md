# Privacy-Preserving Classification of X-Ray Images in a Federated Learning Setting

The project explores the application of mechanisms to defend a federated learning model against reconstruction attacks and enhance privacy.

# Script Usage Examples

**Split CheXpert data**  

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
Specify the path where data lives (this path should contain ```CheXpert-v1.0-small/```) and where the CSVs are saved:  
```sh
python3 split_chexpert.py config.json -d  path_to_chexpert/
```

**Train a network using federated learning**   
Using a GPU and verifying its availability can be turned off with ```--no_gpu``` flag:   
```sh
python3 train_FL.py config.json [--no_gpu]
```
Otherwise, specify which GPUs to use in the script:   
```python
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
```
Add the path where results should be stored:   
```sh
python3 train_FL.py config.json -o path_to_results/
```
Specify the path where data lives (this path should contain ```CheXpert-v1.0-small/```)
```sh
python3 train_FL.py config.json -d path_to_chexpert/
```

**Test a model**  
Test a model on individual test data sets of clients. Usage is similar to ```train_FL.py```. Specify path to model to test:  
```sh
python3 test_model.py config.json -m ./results/global_2rounds.pth.tar
```
Make sure to use the correct ```config``` file, pay attention to dataset settings such as ```class_idx```, as well as ```num_clients``` and ```client_dirs```. Those should be the same as used for model training.

# Result Output

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
