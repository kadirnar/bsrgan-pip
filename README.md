<div align="center">
<h2>
  BSRGAN-Pip: Packaged version of the BSRGAN repository  
</h2>
<h4>
    <img width="400" alt="teaser" src="docs/results.png">
</h4>
<div>
    <a href="https://pepy.tech/project/bsrgan"><img src="https://pepy.tech/badge/bsrgan" alt="downloads"></a>
    <a href="https://badge.fury.io/py/bsrgan"><img src="https://badge.fury.io/py/bsrgan.svg" alt="pypi version"></a>
    <a href="https://huggingface.co/spaces/kadirnar/bsrgan"><img src="https://img.shields.io/badge/%20HuggingFace%20-Demo-blue.svg" alt="HuggingFace Spaces"></a>
</div>
</div>

## <div align="center">Overview</div>

This repo is a packaged version of the [BSRGAN](https://github.com/cszn/BSRGAN) model.
### Installation
```
pip install bsrgan
```

### BSRGAN Usage
```python
from bsrgan import BSRGAN

model = BSRGAN(weights='kadirnar/bsrgan', device='cuda:0')
pred = model.predict(img_path='data/image/test.png')
```
### Citation
```bibtex
@article{li2022yolov6,
  title={YOLOv6: A single-stage object detection framework for industrial applications},
  author={Li, Chuyi and Li, Lulu and Jiang, Hongliang and Weng, Kaiheng and Geng, Yifei and Li, Liang and Ke, Zaidan and Li, Qingyuan and Cheng, Meng and Nie, Weiqiang and others},
  journal={arXiv preprint arXiv:2209.02976},
  year={2022}
}
```
