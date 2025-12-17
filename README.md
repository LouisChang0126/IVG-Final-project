# SuperGaussian

## Environment install

```bash
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip

pip uninstall torch torchvision functorch tinycudann
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

pip install nerfstudio
```

* 注意：每次重開termanal後要重新執行
```bash
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
```


## nerfstudio指令
```bash
# 訓練3DGS模型
# ns-train splatfacto --data ./bear

# 訓練3DGS模型(低解析度)
ns-train splatfacto --data ./bear nerfstudio-data --downscale-factor 4

# 觀看訓練完的3DGS模型
ns-viewer --load-config outputs/bear/splatfacto/2025-12-17_012229/config.yml
# 可以在http://localhost:7007/ 觀看並且調整render的軌跡(調整完會自動覆蓋舊的)

# 渲染訓練完的3DGS模型成影片
ns-render camera-path --load-config outputs/bear/splatfacto/2025-12-09_153213/config.yml \
    --camera-path-filename bear/camera_paths/2025-11-04-10-53-10.json \
    --output-path renders/bear/2025-11-04-10-53-12.mp4

# 渲染訓練完的3DGS模型成圖片
ns-render camera-path --load-config outputs/bear/splatfacto/2025-12-17_012229/config.yml \
    --camera-path-filename bear/camera_paths/old.json \
    --output-path renders/bear/low_resolution64x64 \
    --output-format images
```

# [CVPR 2024 Highlight] Enhancing Video Super-Resolution via Implicit Resampling-based Alignment

## Installation

目前使用 python3.9可以順利運行

```bash
python3.9 -m venv .venv
# for powershell
.venv/bin/Activate.ps1 
# for bash
source .venv/bin/activate

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
需手動更改 `.venv\Lib\site-packages\basicsr\data\degradations.py`
Change line 8:

```Python
# OLD (Delete this)
from torchvision.transforms.functional_tensor import rgb_to_grayscale
```
To this:
```Python
# NEW (Replace with this)
from torchvision.transforms.functional import rgb_to_grayscale
```

## Run Demo

Download models from [this link](https://drive.google.com/drive/folders/1MIUK37Izc4IcA_a3eSH-21EXOZO5G5qU?usp=sharing) and put them under `IVG-Final-project/IART/experiments/`.

```bash
# Run demo on frames under `demo/Vid4_BI`:
python demo.py
```

## Data

目前檔案位置 
- low resolution: `IVG-Final-project/IART/demo/bear`
- 4x resolution: `IVG-Final-project/IART/demo/bear_results`

