#!/bin/sh

# Download DINOv3 pretrained weights from HuggingFace (timm pre-converted, correct key names).
# Must be run from the repo root directory.
if [ ! -f "unimatch_v2/pretrained/dinov3_small.safetensors" ]; then
    hf download timm/vit_small_patch16_dinov3.lvd1689m model.safetensors \
        --local-dir unimatch_v2/pretrained
    mv unimatch_v2/pretrained/model.safetensors unimatch_v2/pretrained/dinov3_small.safetensors
fi
if [ ! -f "unimatch_v2/pretrained/dinov3_base.safetensors" ]; then
    hf download timm/vit_base_patch16_dinov3.lvd1689m model.safetensors \
        --local-dir unimatch_v2/pretrained
    mv unimatch_v2/pretrained/model.safetensors unimatch_v2/pretrained/dinov3_base.safetensors
fi
