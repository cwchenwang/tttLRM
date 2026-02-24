<h1 align="center">tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction</h1>
<p align="center"><a href="https://arxiv.org/abs/2602.20160"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://cwchenwang.github.io/tttLRM/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>

## 📦 Installation
```bash
python3.10 -m venv tttlrm
source tttlrm/bin/activate
# CAUTION: change it to your CUDA version
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118 xformers
pip install -U setuptools wheel packaging ninja
## Install flash-attn (You can also install prebuild wheels at: https://github.com/mjun0812/flash-attention-prebuild-wheels)
pip install flash_attn==2.5.9.post1 --no-build-isolation
pip install -r requirements.txt
```

## 🤖 Pretrained Models
```bash
bash script/download_ckpts.sh
```

## ⚡ Inference
We use `sp_size` for sequence parallel, which denotes the number of GPUs used for one sequence. Input views and generated Gaussians will be evenly distributed to `sp_size` GPUs.
```bash
# For Full model
bash script/inference_dl3dv.sh
# For AR model (4 views per chunk, set by '-s model.miniupdate_views')
bash script/inference_dl3dv_ar.sh
```
We might not provide training code at this moment, but it can be easily done by combining [LongLRM](https://github.com/arthurhero/Long-LRM/blob/main/main.py) and our inference code (the sequence parallel part).

## 📂 Dataset
Download DL3DV benchmark (i.e., test; not used in training) data at https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark/tree/main using the following command:
```bash
python data/dl3dv_eval_download.py --odir ./data_example/dl3dv_benchmark --subset hash --only_level4 --hash 032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7
```
Use option `--subset full` to download all testing scenes. After downloading, run
```bash
python data/dl3dv_format_converter.py
```
to convert to our dataset format (OpenCV camera). 

## 🤝 Acknowledgements
Our codebase is a replementation of the internal version. Performance is matched under the same model weights. The code is largely built upon open-source projects including [LongLRM](https://github.com/arthurhero/Long-LRM) and [LaCT](https://github.com/a1600012888/LaCT). We thank the authors for their helpful code.

## ⚖️ License
The checkpoints are licensed under Adobe Research License. 

## 📜 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{wang2026tttlrm,
    title   = {tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction},
    author  = {Chen Wang, Hao Tan, Wang Yifan, Zhiqin Chen, Yuheng Liu, Kalyan Sunkavalli, Sai Bi, Lingjie Liu, Yiwei Hu},
    journal = {CVPR},
    year    = {2026}
}
```

