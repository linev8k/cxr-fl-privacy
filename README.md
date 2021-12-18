# Privacy-Preserving Classification of X-Ray Images in a Federated Learning Setting

This project explores the application of federated learning to classification of X-ray images and the integration of differential privacy to defend local models against reconstruction attacks and enhance privacy.

## Setup

Create and activate a conda environment with

```sh
conda env create -f environment.yml
conda activate cxr
```

This environment also enables reconstruction experiments from [invert-gradients-cxr](https://github.com/linev8k/invert-gradients-cxr).

## Quick Start

**Prepare Data**  

Split CheXpert or Mendeley data files with ```split_chexpert.py``` (if one patient has several images) and ```split_mendeley.py```(if one patient has one image). Specify file paths and parameters in scripts.

Specify where data files live and where the CSVs are saved:  
```sh
python split_chexpert.py config.json -d  path_to_data/
```
The dataloader can be found in ```chexpert_data.py```.



**Train a Model with Federated Learning**   

The main training script is ```train_FL.py```. It uses modules from ```trainer.py```, ```utils.py```, and ```chexpert_data.py```.  

Example (see the script and```run_fl.sh``` for more options):

```sh
python train_FL.py config.json -o path_to_output/ -d path_to_data/ -df path_to_client_data_files/ 
```


**Train a Model with Local Differentially Privacy**

Set ```private=true``` in ```config.json``` and specify the privacy parameters in ```privacy_config.json```. 



**Test a Model**  
Test a model on client subdatasets. Outputs a CSV with AUC values per client.

The following example uses the clients' validation data files for testing:

```sh
python test_model.py config.json -m model.pth.tar --val
```
See the script and ```run_test.sh``` for more argument options.
## Training Configuration

Specify training parameters in ```config.json```:

| Parameter                    | Options                                                      | Note                                                   |
| ---------------------------- | ------------------------------------------------------------ | ------------------------------------------------------ |
| front_lat                    | 'frontal' \| 'lateral' \| 'both'                             | X-ray image views to include                           |
| class_idx                    | ```list``` of ```int```                                      | Index of class labels to use for classification        |
| policy                       | 'zeros' \| 'ones'                                            | Replace uncertainty labels with zeros or ones          |
| input                        | 'L' \| 'RGB'                                                 | Image as one-channel or three-channel input            |
| net                          | ```str```, e.g., 'DenseNet121'                               | Name of network architecture (```custom_models.py```)  |
| imgtransResize               | ```int```, eg., 224                                          | Images are resized to square size                      |
| augment                      | ```bool```                                                   | Apply training data augmentation                       |
| random_seed                  | ```int```                                                    | Reproducibility                                        |
| pre_trained                  | ```bool```                                                   | Use a pre-trained model (on ImageNet data)             |
| freeze_mode                  | 'none' \| 'batch_norm ' \| 'all_but_last' \| 'middle'        | Layer freezing                                         |
| optimal                      | 'SGD' \| 'Adam'                                              | Optimizer                                              |
| lr, betas, eps, weight_decay | ```float```, ```list``` of ```float```, ```float```, ```float``` | Optimizer parameters                                   |
| earl_stop_rounds             | ```int```                                                    | Early stopping after n rounds without improvement      |
| reduce_lr_rounds             | ```int```                                                    | Reduce lr after n rounds without improvement           |
| num_clients                  | ```int```                                                    |                                                        |
| batch_size                   | ```int```                                                    |                                                        |
| max_epochs                   | ```int```                                                    | Max. local epochs                                      |
| com_rounds                   | ```int```                                                    | Max. communication rounds                              |
| fraction                     | ```float```                                                  | Fraction of clients to subsample each round            |
| sel_max_rounds               | ```int```                                                    | Max. rounds a client can be selected                   |
| private                      | ```bool```                                                   | Apply differential privacy (```privacy_config.json```) |
| track_norm                   | ```bool```                                                   | Track gradient l2-norms during training                |

For private training parameters, check out ```privacy_config.json```:

| Parameter        | Options                  | Note                                                |
| ---------------- | ------------------------ | --------------------------------------------------- |
| epsilon          | ```float``` or ```int``` |                                                     |
| min_delta        | ```float```              |                                                     |
| max_grad_norm    | ```float```              | Gradient clipping value                             |
| noise_multiplier | ```float```              | If epsilon is fixed, this is inferred automatically |



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
├── train_results.csv 				# merged CSV of individual clients' validation metrics
```


## Acknowledgements

* [CheXpert classification](https://github.com/Stomper10/CheXpert)
* [CheXpert Data](https://stanfordmlgroup.github.io/competitions/chexpert/) and [Mendeley X-Ray Data](https://data.mendeley.com/datasets/rscbjbr9sj/3)
* [PyTorch](https://pytorch.org/) 1.9 and [Opacus](https://opacus.ai/) 0.14
