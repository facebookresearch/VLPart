## Installation

### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.9 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- `pip install -r requirements.txt`


### Example conda environment setup
```bash
conda create --name vlpart python=3.9 -y
conda activate vlpart
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia

# under your working directory
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ..

git clone https://github.com/facebookresearch/VLPart.git
cd VLPart
pip install -r requirements.txt
```
