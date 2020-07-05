# Shopee Product Detection
## Usage
### Setting config files
***1. specify datapath, preprocess parameters in the config file***

```python=
dataset:
    name: 'ImageDataset'
    kwargs:
        data_dir: '/nfs/nas-5.1/wbcheng/shopee_task2'
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
``` 

***2. specify the model in config file***
```python=
net:
    name: 'EffNet_b3_fc'
    kwargs:
        in_channels: 3
        out_channels: 42
```
>1. EffNet_b3_fc
>2. ResNet
>3. EffNet_b4
>4. .....

***3. Specify model save path and random seed for reproducing same result***
```python=
main:
    random_seed: 8745
    saved_dir:'/nfs/nas5.1/wbcheng/shopee_task2/modules/Effnet_b3fc_modified'
```
>change saved_dir path

<br></br>
## Run Script file
./run.sh <datafile path>
> the whole process will take approximately two days on a single 2080Ti GPU.

