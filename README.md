# image-compression-with-swin-transformer
Learned image compression with transformers

https://spie.org/Publications/Proceedings/Paper/10.1117/12.2656516?SSO=1
## Citation
```
@inproceedings{shen2023learned,
  title={Learned image compression with transformers},
  author={Shen, Tianma and Liu, Ying},
  booktitle={Big Data V: Learning, Analytics, and Applications},
  volume={12522},
  pages={10--20},
  year={2023},
  organization={SPIE}
}
```

## Installation

This method high depend on [CompressAI](https://github.com/InterDigitalInc/CompressAI), If you meet some problems for install compressai, please check their Doc firstly.
```bash
conda create -n compress python=3.7
conda activate compress
pip install compressai
pip install pybind11
git clone https://github.com/stm233/image-compression-with-swin-transformer image-compression
cd image-compression
pip install -e .
pip install -e '.[dev]'
```

## Usage

### Dataset
If you wanna choice the same dataset like our paper, please install fiftyone.

```bash
pip install fiftyone
```
Then use the downloader_openimages.py to download the training images. You need to setup the path for your own dataset at .py

```bash
python downloader_openimages.py
```
#### Data Structure
Please put the training and validation data into the right path, or you need to fix the datasets/utils.py

- rootdir/
    - train/
        - data/   
            - img000.png
            - img001.png
    - test/
        - data/  
            - img000.png
            - img001.png

### Pre-trained Model
Right now, we just share one checkpoints, If you need all points, you can contract Tshen2@scu.edu

[lambda = 0.0035](https://drive.google.com/file/d/1tRsx-ek8O2lXlcLdMnQ9q5sD-V_4nuGQ/view?usp=drive_link) 



### Training

Here are some super-parameters you need to set up.

```
-m stf8 # model name, if you have youe own model, you need setup at /z00/__int__.py and /models/ __int__.py
-d /data/Dataset/openimages/ # the path to store your training and validaiton data
--lambda 0.0035 # the number to adjust the bit rate of your model, the lambda is higher, the bit rate is higher
--batch_size 12 # this is depend on your GPU's memory
--patch_szie 256,256 # this is the input image's size, we prefer to enlarge the size when the model convergenced
--save_path ./save/ # the save path for your model
--checkpoint ./save/23.ckpt # if this is empty, the model will be trained from the stratch
```

If you wanna use the default super parameter to train your model, we can miss some items in your command.

Eg.

```
python train.py -m stf8 -d /data/Dataset/openimages/ --lambda 0.025 --batch_size 24
```


