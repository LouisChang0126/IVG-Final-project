conda activate nerfstudio
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

# ---------------------------------------------------------
# Step 1: Generate Camera Path
# ---------------------------------------------------------
python transforms_to_camerapath.py \
    --transforms bear/transforms.json \
    --output bear/camera_paths/248x184.json \
    --fps 24 \
    --render-width 248 \
    --render-height 184

# ---------------------------------------------------------
# Step 2: Render from low resolution 3DGS
# ---------------------------------------------------------
LATEST_BEAR=$(ls outputs/bear/splatfacto/ | sort | tail -n 1)
BEAR_CONFIG="outputs/bear/splatfacto/$LATEST_BEAR/config.yml"

echo "Using latest Bear config: $BEAR_CONFIG"

ns-render camera-path --load-config "$BEAR_CONFIG" \
    --camera-path-filename bear/camera_paths/248x184.json \
    --output-path SR_bear/images_low \
    --output-format images

# ---------------------------------------------------------
# Step 3: Super Resolution (MIA-VSR)
#           Can adjust to IART, InvSR, etc.
# ---------------------------------------------------------
python MIA-VSR/inference_miavsr.py \
    --test_name miavsr \
    --lr_folder SR_bear/images_low \
    --output_folder SR_bear/images \
    --save_imgs \
    --no_tile

# ---------------------------------------------------------
# Step 4: Render camera path to training trajectory
# ---------------------------------------------------------
python camerapath_to_transforms.py \
    --camera_path bear/camera_paths/248x184.json \
    --output_dir SR_bear \
    --ext png \
    --scale 4.0

# ---------------------------------------------------------
# Step 5: Train SR 3DGS
# ---------------------------------------------------------
ns-train splatfacto --data ./SR_bear

# ---------------------------------------------------------
# Step 6: Render from SR 3DGS
# ---------------------------------------------------------
LATEST_SR_BEAR=$(ls outputs/SR_bear/splatfacto/ | sort | tail -n 1)
SR_BEAR_CONFIG="outputs/SR_bear/splatfacto/$LATEST_SR_BEAR/config.yml"

echo "Using latest SR_Bear config: $SR_BEAR_CONFIG"

ns-render camera-path --load-config "$SR_BEAR_CONFIG" \
    --camera-path-filename bear/camera_paths/992x736.json \
    --output-path SR_bear/SR_3DGS \
    --output-format images