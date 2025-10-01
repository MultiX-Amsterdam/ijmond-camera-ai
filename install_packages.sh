#!/bin/sh

pip install --upgrade opencv-python~=4.11.0
pip install --upgrade scikit-learn==1.6.1
pip install --upgrade requests==2.32.4
pip install --upgrade pillow==11.2.1
pip install --upgrade matplotlib==3.10.3
pip install --upgrade tqdm==4.67.1
pip install git+https://github.com/facebookresearch/segment-anything.git
wget -P bbox_learn https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth