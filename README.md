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

## quick start
```bash
bash ./demo.sh
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

# 渲染訓練完的3DGS模型成圖片
ns-render camera-path --load-config outputs/bear/splatfacto/2025-12-17_012229/config.yml \
    --camera-path-filename bear/camera_paths/old.json \
    --output-path renders/bear/low_resolution64x64 \
    --output-format images
```

## pipeline
```bash
# training trajectory to render camera path
python transforms_to_camerapath.py \
    --transforms bear/transforms.json \
    --output bear/camera_paths/248x184.json \
    --fps 24 \
    --render-width 248 \
    --render-height 184

# render from low resolution 3DGS
ns-render camera-path --load-config outputs/bear/splatfacto/2025-12-23_154150/config.yml \
    --camera-path-filename bear/camera_paths/248x184.json \
    --output-path SR_bear/images_low \
    --output-format images

# super resolution
python MIA-VSR/inference_miavsr.py \
    --test_name miavsr \
    --lr_folder SR_bear/images_low \
    --output_folder SR_bear/images \
    --save_imgs \
    --no_tile

# render camera path to training trajectory
python camerapath_to_transforms.py \
    --camera_path bear/camera_paths/248x184.json \
    --output_dir SR_bear \
    --ext png \
    --scale 4.0

# train SR 3DGS
ns-train splatfacto --data ./SR_bear

# render from SR 3DGS
ns-render camera-path --load-config outputs/SR_bear/splatfacto/2025-12-24_015100/config.yml \
    --camera-path-filename bear/camera_paths/992x736.json \
    --output-path SR_bear/SR_3DGS \
    --output-format images
```

## MIA-VSR
```bash
python MIA-VSR/inference_miavsr.py \
    --test_name miavsr \
    --lr_folder SR_bear/images_low \
    --output_folder SR_bear/images \
    --save_imgs \
    --no_tile
```

## IART
Download models from [this link](https://drive.google.com/drive/folders/1MIUK37Izc4IcA_a3eSH-21EXOZO5G5qU?usp=sharing) and put them under `IVG-Final-project/IART/experiments/pretrained_models/`.

```bash
python IART/inference_iart.py \
--lr_folder SR_bear/images_low \
    --output_folder SR_bear/images_IART \
    --no_tile
```

## InvSR
```
./InvSR/run_batch.sh SR_bear/images_low SR_bear/images_InvSR
```

## JPG/PNG to GIF
```bash
# cd to img folder
ffmpeg -f image2 -framerate 10 -i %05d.jpg output.gif
# ffmpeg -f image2 -framerate 10 -i %05d.png output.gif
```