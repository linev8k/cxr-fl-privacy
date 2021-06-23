# Privacy-Preserving Classification of X-Ray Images in a Federated Learning Setting

The project explores the application of mechanisms to defend a federated learning model against reconstruction attacks and enhance privacy.

# Script Usage Examples

**Splitting of CheXpert data**  

Split CheXpert data without patient overlap. Currently, this splits the data randomly without any sorting mechanism.  
Insert the name of the CSV with data that should be split (can be original or subset of CheXpert data), the names of the CSVs for the resulting splits, and the percentages of the splits in the script:  
```python
ORIG_CSV = 'train.csv' #name of csv to split
CSV_NAMES = ['train_mod.csv', 'test_mod.csv'] #filenames for saving
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

**Training a network using federated learning**   
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


## Acknowledgements

* [CheXpert classification](https://github.com/Stomper10/CheXpert)
