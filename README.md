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





