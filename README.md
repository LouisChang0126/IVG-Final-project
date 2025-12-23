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
ns-viewer --load-config outputs/bear/splatfacto/2025-12-23_154150/config.yml
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

## pipeline
```bash
mkdir SR_bear

# training trajectory to render camera path
python transforms_to_camerapath.py \
    --transforms bear/transforms.json \
    --output bear/camera_paths/20251223.json \
    --fps 24 \
    --render-width 400 \
    --render-height 400

# render from low resolution 3DGS
ns-render camera-path --load-config outputs/bear/splatfacto/2025-12-23_154150/config.yml \
    --camera-path-filename bear/camera_paths/20251223.json \
    --output-path SR_bear/images \
    --output-format images

# render camera path to training trajectory
python camerapath_to_transforms.py \
    --camera_path bear/camera_paths/20251223.json \
    --output_dir SR_bear \
    --ext jpg

######################
##      SR          ##
######################

# train SR 3DGS
ns-train splatfacto --data ./SR_bear
```

