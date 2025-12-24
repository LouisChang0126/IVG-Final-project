import argparse
import cv2
import glob
import logging
import os
import os.path as osp
import torch
import torch.nn.functional as F
import math

# 引入 IART 模型
from archs.iart_arch import IART
from basicsr.data.data_util import read_img_seq
from basicsr.utils import get_root_logger, get_time_str, imwrite, tensor2img

def tile_process(model, lq, scale=4, tile_size=192, tile_pad=32):
    """
    將 5D Tensor (B, T, C, H, W) 進行空間切塊推論，適用於 IART
    """
    b, t, c, h, w = lq.size()
    img_h, img_w = h * scale, w * scale
    
    output = torch.zeros((b, t, c, img_h, img_w), device=lq.device, dtype=lq.dtype)
    
    h_idx_list = list(range(0, h, tile_size))
    w_idx_list = list(range(0, w, tile_size))
    
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            h_start = max(h_idx - tile_pad, 0)
            h_end = min(h_idx + tile_size + tile_pad, h)
            w_start = max(w_idx - tile_pad, 0)
            w_end = min(w_idx + tile_size + tile_pad, w)
            
            in_patch = lq[..., h_start:h_end, w_start:w_end]
            
            with torch.no_grad():
                # 修改點：IART 直接回傳 output，不像 MIAVSR 回傳 tuple
                out_patch = model(in_patch)
            
            h_start_in = h_idx - h_start
            w_start_in = w_idx - w_start
            h_end_in = h_start_in + min(tile_size, h - h_idx)
            w_end_in = w_start_in + min(tile_size, w - w_idx)
            
            h_start_out = h_idx * scale
            w_start_out = w_idx * scale
            h_end_out = h_start_out + (h_end_in - h_start_in) * scale
            w_end_out = w_start_out + (w_end_in - w_start_in) * scale
            
            output[..., h_start_out:h_end_out, w_start_out:w_end_out] = \
                out_patch[..., h_start_in*scale:h_end_in*scale, w_start_in*scale:w_end_in*scale]
            
            del in_patch, out_patch
    
    return output

def main(args):
    # -------------------- Configurations -------------------- #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_imgs = args.save_imgs
    
    # 這裡請確保路徑正確，或者透過 args 傳入
    # 注意：需修改為你實際存放 IART 權重的路徑
    model_path = args.model_path 
    spynet_path = args.spynet_path

    lr_folder = args.lr_folder
    save_folder = args.output_folder
    os.makedirs(save_folder, exist_ok=True)

    # logger
    log_file = osp.join(save_folder, f'inference_iart_{get_time_str()}.log')
    logger = get_root_logger(logger_name='iart_inference', log_level=logging.INFO, log_file=log_file)
    logger.info(f'Data: {lr_folder}')
    logger.info(f'Model path: {model_path}')

    # -------------------- Model Initialization -------------------- #
    # 參數參照 demo.py 的設定
    model = IART(mid_channels=64,
                 embed_dim=120,
                 depths=[6, 6, 6],      # IART 設定
                 num_heads=[6, 6, 6],   # IART 設定
                 window_size=[3, 8, 8],
                 num_frames=3,
                 cpu_cache_length=100,
                 is_low_res_input=True,
                 spynet_path=spynet_path)

    # 載入權重
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        if 'params' in checkpoint:
            model.load_state_dict(checkpoint['params'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        logger.warning(f"Model path {model_path} does not exist. Please check arguments.")

    model.eval()
    model = model.to(device)

    # -------------------- Inference Loop -------------------- #
    # 支援讀取資料夾內的子資料夾 (Sequence) 或是直接讀取該資料夾內的圖片
    # 邏輯判斷：如果 lr_folder 內全是圖片，則視為單一 sequence；若有子資料夾，則視為多個 sequences
    
    first_level_files = sorted(glob.glob(osp.join(lr_folder, '*')))
    is_single_sequence = all([osp.isfile(x) for x in first_level_files])

    if is_single_sequence:
        subfolder_l = [lr_folder]
    else:
        subfolder_l = sorted(glob.glob(osp.join(lr_folder, '*')))

    for subfolder in subfolder_l:
        subfolder_name = osp.basename(subfolder)
        if subfolder_name == '': # Handle trailing slash
             subfolder_name = osp.basename(osp.dirname(subfolder))
             
        logger.info(f'Processing sequence: {subfolder_name}')

        # 讀取圖片序列
        imgs_lq, imgnames = read_img_seq(subfolder, return_imgname=True)
        
        # 處理 Temporal Batch (避免一次塞入太多 Frame 爆顯存)
        length = len(imgs_lq)
        interval = args.interval
        iters = length // interval
        if length % interval > 0:
            iters += 1
        
        imgs_lq = imgs_lq.unsqueeze(0) # Add Batch Dimension: (1, T, C, H, W)
        
        # 建立該序列的輸出資料夾
        if is_single_sequence:
            seq_save_folder = save_folder
        else:
            seq_save_folder = osp.join(save_folder, subfolder_name)
        os.makedirs(seq_save_folder, exist_ok=True)

        name_idx = 0
        for i in range(iters):
            min_id = min((i + 1) * interval, length)
            lq = imgs_lq[:, i * interval:min_id, :, :, :]
            lq = lq.to(device)
            
            with torch.no_grad():
                if args.no_tile:
                    outputs = model(lq)
                else:
                    outputs = tile_process(model, lq, scale=4, tile_size=args.tile_size, tile_pad=args.tile_pad)
                
                # IART demo code logic for squeeze
                if outputs.dim() == 5:
                    outputs = outputs.squeeze(0)
                
                for idx in range(outputs.shape[0]):
                    if name_idx < len(imgnames):
                        img_name = imgnames[name_idx] + '.png'
                        # tensor2img: rgb2bgr=True 適用於 imwrite (OpenCV based)
                        output_img = tensor2img(outputs[idx], rgb2bgr=True, min_max=(0, 1))
                        imwrite(output_img, osp.join(seq_save_folder, img_name))
                        name_idx += 1
            
            # 釋放記憶體
            del lq
            del outputs
            torch.cuda.empty_cache()
            
        logger.info(f'Finished {subfolder_name}, processed {name_idx} frames.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 必要路徑
    parser.add_argument('--lr_folder', type=str, required=True, help='Path to the low-resolution images folder.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output images folder.')
    
    # 模型路徑 (建議設定預設值或由指令傳入)
    parser.add_argument('--model_path', type=str, default='IART/experiments/pretrained_models/IART_REDS_BI_N16.pth', help='Path to IART checkpoint.')
    parser.add_argument('--spynet_path', type=str, default='IART/experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth', help='Path to Spynet checkpoint.')

    # Inference 設定
    parser.add_argument('--save_imgs', action='store_true', default=True, help='Flag to save output images.')
    parser.add_argument('--interval', type=int, default=10, help='Interval for frame chunking (temporal batch size).')
    
    # Tiling 設定 (解決顯存不足)
    parser.add_argument('--no_tile', action='store_true', help='Flag to disable tile-based processing.')
    parser.add_argument('--tile_size', type=int, default=192, help='Tile size for patch-based processing.')
    parser.add_argument('--tile_pad', type=int, default=32, help='Padding size for each tile.')
    
    args = parser.parse_args()
    main(args)