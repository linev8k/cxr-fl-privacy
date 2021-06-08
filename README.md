# Privacy-Preserving Classification of X-Ray Images in a Federated Learning Setting

The project explores the application of mechanisms to defend a federated learning model against reconstruction attacks and enhance privacy.

# Script Usage Examples

**Preprocessing of CheXpert data** (defining data splits)  

```sh
python3 chexpert_preprocessing.py config.json
```
**Training a network using federated learning**   
For testing, verifying GPU availability can be turned off with ```--no_gpu``` flag:   
```sh
python3 train_FL.py config.json [--no_gpu]
```
Add the path where results should be stored:   
```sh
python3 train_FL.py config.json -o path_to_results/
```

Specify which GPUs to use in the script:   
```python
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
```


## Acknowledgements

* [CheXpert classification](https://github.com/Stomper10/CheXpert)
