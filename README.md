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
