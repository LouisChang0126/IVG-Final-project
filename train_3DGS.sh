export CUDA_VISIBLE_DEVICES=1
conda activate nerfstudio
export CC=gcc-11
export CXX=g++-11
# 處理圖片數據
# ns-process-data images --data ./bear --output-dir ./processed_data
# 開始訓練
# ns-train splatfacto --data ./bear
ns-train splatfacto --data ./bear nerfstudio-data --downscale-factor 4 # low resolution
# 觀看
ns-viewer --load-config outputs/bear/splatfacto/2025-12-09_153213/config.yml
# 渲染影片
ns-render camera-path --load-config outputs/bear/splatfacto/2025-12-09_153213/config.yml --camera-path-filename bear/camera_paths/2025-11-04-10-53-10.json --output-path renders/bear/2025-11-04-10-53-12.mp4
# 渲染圖片
ns-render camera-path --load-config outputs/bear/splatfacto/2025-12-09_154508/config.yml --camera-path-filename bear/camera_paths/2025-11-04-10-53-10.json \
    --output-path renders/bear/low_resolution \
    --output-format images