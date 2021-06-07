# GPU Usage SCORE Lab

A basic guide on how the GPU resource provided by the HPI SCORE Lab can be used with Anaconda.  
An official documentation on the hardware and how to reserve it is available [here](https://score.hpi.uni-potsdam.de/). This should always be the first reference. Note that you have to be inside the HPI network (VPN) to be able to access most of the material.

_Last updated: 07. June, 2021_  

## Resources

Currently there are three GPU machines available (see also the [Hardware Specs](https://score.hpi.uni-potsdam.de/wiki/DELab/Specs)). 

* 2x IBM AC922 (4 GPUs each)  
(Note: This is a power PC architecture which needs to be considered when setting up the compute environment.)  
* NVIDIA DGX A100 (8 GPUs)  

The AC922 can be reserved by booking time slots via calendars. The DGX is currently managed using a Google Sheet. Reservation details can be found [here](https://docs.google.com/document/d/1LmHuF8wpyAnzUt3jGBa9132AsE3kz8EInJdPvqPxhEM/edit?usp=sharing).

Before working with the GPUs, you can verify availabilities with ```nvidia-smi```.  

### Access

AC922-01:  
```sh
ssh [firstname.lastname]@ac922-01.delab.i.hpi.de
```
AC922-02:
```sh
ssh [firstname.lastname]@ac922-02.delab.i.hpi.de
```
DXA-A100:
```sh
ssh [firstname.lastname]@dgxa100-01.delab.i.hpi.de
```

## Storage

You have a home directory available on all machines. However, it is **not shared** between machines. Ask where you can store large amounts of data. On DGX-A100 you also have access to ```/scratch```, where you can put temporary files (though keep in mind that no backups are kept for this.)

## Anaconda - DGX-A100

Anaconda is a convenient package manager for Python.  

### Set Up Anaconda

Install Anaconda in your home directory. To do this, follow [these steps](https://docs.anaconda.com/anaconda/install/linux/). Omit the GUI packages. For the first step, instead of downloading Anaconda from a browser, you can use ```wget``` with the respective link, as in   ```wget https://www.anaconda.com/download/#linux```.  
You can also consider to install [miniconda](https://docs.conda.io/en/latest/miniconda.html) for a more light-weight installation. 

A guide on how to manage environments with conda is available [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Keep an eye on memory space taken up by Anaconda packages, it can pile up when working with lots of environments.

#### Environment Setup Example

Create and activate an environment  
```sh
conda create --name [env_name]
conda activate [env_name]
```
Install packages  
```sh
conda install [package]
```
Verify installations  
```sh
conda list
```

### PyTorch

If you want to work with PyTorch, conda might not install the correct CUDA version automatically. You can refer to the official [PyTorch installation recommendation](https://pytorch.org/get-started/locally/) on how to install PyTorch correctly. Currently, the following command installs the correct dependencies:  
```sh
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```

#### Check and Use GPUs in PyTorch Code

Check if PyTorch recognizes the GPUs
```python
import torch
torch.cuda.is_available()
```
Check number of available GPUs 
```python
torch.cuda.device_count()
```
Assign GPU 0 to a variable for convenient moving of objects to this resource, create two tensors and perform an operation on them using the GPU.
```python
cuda_device = torch.device('cuda:0') #specify index of GPU after colon

#first tensor
t1 = torch.ones(1,50)
print(t1.device) #should be CPU
t1 = t1.to(cuda_device)
print(t1.device) #should be specified GPU

#second tensor
t2 = torch.ones(1,50)
t2 = t2.to(cuda_device)

#add them
t_sum = t1 + t2
print(t_sum.device) #should be specified GPU
```
[More examples](https://deeplizard.com/learn/video/Bs1mdHZiAS8) of how to use PyTorch with a GPU.

To limit the resources that PyTorch should use to selected GPUs, you can set the environment variable _before_ importing the torch module.  
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3' #make only GPUs 2 and 3 accessible

import torch
print(torch.cuda.device_count()) #should be only 2 now
```
Note that PyTorch will index them starting from 0 again, independent of their original index.
