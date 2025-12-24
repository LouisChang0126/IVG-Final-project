import cv2
import glob
import logging
import os
import os.path as osp
import torch
import torch.nn.functional as F
from archs.mia_vsr_arch import MIAVSR
from basicsr.data.data_util import read_img_seq
from basicsr.metrics import psnr_ssim
from basicsr.utils import get_root_logger, get_time_str, imwrite, tensor2img

import math

def tile_process(model, lq, scale=4, tile_size=192, tile_pad=32):
    """
    將 5D Tensor (B, T, C, H, W) 進行空間切塊推論
    tile_size: 切塊大小 (越小越省 VRAM，但速度變慢)
    tile_pad: 重疊區域 (避免拼貼邊界出現線條)
    """
    b, t, c, h, w = lq.size()
    img_h, img_w = h * scale, w * scale
    
    # 準備一個空的輸出來裝結果 (注意：這裡還是會佔用 VRAM，如果這裡就爆，要改存到 CPU)
    # output shape: (B, T, C, H*scale, W*scale)
    output = torch.zeros((b, t, c, img_h, img_w), device=lq.device, dtype=lq.dtype)
    
    # 計算切塊座標
    h_idx_list = list(range(0, h, tile_size))
    w_idx_list = list(range(0, w, tile_size))
    
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            # 1. 定義輸入的切塊範圍 (包含 padding)
            h_start = max(h_idx - tile_pad, 0)
            h_end = min(h_idx + tile_size + tile_pad, h)
            w_start = max(w_idx - tile_pad, 0)
            w_end = min(w_idx + tile_size + tile_pad, w)
            
            # 2. 切出這一塊 (Input Crop)
            in_patch = lq[..., h_start:h_end, w_start:w_end]
            
            # 3. 進模型推論
            with torch.no_grad():
                # MIA-VSR 回傳 (output, flow)，我們只要 output
                out_patch, _ = model(in_patch)
            
            # 4. 計算輸出的有效區域 (扣除 padding)
            h_start_in = h_idx - h_start
            w_start_in = w_idx - w_start
            h_end_in = h_start_in + min(tile_size, h - h_idx)
            w_end_in = w_start_in + min(tile_size, w - w_idx)
            
            # 放大座標到 Output Scale
            h_start_out = h_idx * scale
            w_start_out = w_idx * scale
            h_end_out = h_start_out + (h_end_in - h_start_in) * scale
            w_end_out = w_start_out + (w_end_in - w_start_in) * scale
            
            # 5. 填回大張 Output
            output[..., h_start_out:h_end_out, w_start_out:w_end_out] = \
                out_patch[..., h_start_in*scale:h_end_in*scale, w_start_in*scale:w_end_in*scale]
            
            # 清理
            del in_patch, out_patch
    
    return output
def main():
    # -------------------- Configurations -------------------- #
    device = torch.device('cuda:0')
    save_imgs = True
    test_y_channel = False
    crop_border = 0
    # set suitable value to make sure cuda not out of memory
    interval = 5
    # model
    model_path = '/home/louis/IVG_final/MIA-VSR/experiments/pretrained_models/MIAVSR_REDS_x4.pth'
    # test data
    test_name = f'sotareds'

    # lr_folder = 'datasets/REDS4/sharp_bicubic'
    # gt_folder = 'datasets/REDS4/GT'
    lr_folder = '/home/louis/IVG_final/bear_SR'
    gt_folder = '/home/louis/IVG_final/bear_SR'
    save_folder = f'results/{test_name}'
    os.makedirs(save_folder, exist_ok=True)

    # logger
    log_file = osp.join(save_folder, f'psnr_ssim_test_{get_time_str()}.log')
    logger = get_root_logger(logger_name='recurrent', log_level=logging.INFO, log_file=log_file)
    logger.info(f'Data: {test_name} - {lr_folder}')
    logger.info(f'Model path: {model_path}')

    # set up the models
    model = MIAVSR( mid_channels=64,
                 embed_dim=120,
                 depths=[6,6,6,6],
                 num_heads=[6,6,6,6],
                 window_size=[3, 8, 8],
                 num_frames=3,
                 cpu_cache_length=100,
                 is_low_res_input=True,
                 use_mask=True,
                 spynet_path='/home/louis/IVG_final/MIA-VSR/experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth')
    model.load_state_dict(torch.load(model_path)['params'], strict=False)
    model.eval()
    model = model.to(device)

    avg_psnr_l = []
    avg_ssim_l = []
    FLOPs = []
    subfolder_l = sorted(glob.glob(osp.join(lr_folder, '*')))
    subfolder_gt_l = sorted(glob.glob(osp.join(gt_folder, '*')))

    # for each subfolder
    subfolder_names = []
    # for subfolder, subfolder_gt in zip(subfolder_l, subfolder_gt_l):
    for subfolder in subfolder_l:
        subfolder_name = osp.basename(subfolder)
        subfolder_names.append(subfolder_name)

        # read lq and gt images
        imgs_lq, imgnames = read_img_seq(subfolder, return_imgname=True)

        # calculate the iter numbers
        length = len(imgs_lq)
        iters = length // interval

        # cluster the excluded file into another group
        if length % interval > 1:
            iters += 1

        avg_psnr = 0
        avg_ssim = 0
        # inference
        name_idx = 0
        imgs_lq = imgs_lq.unsqueeze(0)
        for i in range(iters):
            min_id = min((i + 1) * interval, length)
            lq = imgs_lq[:, i * interval:min_id, :, :, :]
            lq = lq.to(device)
            with torch.no_grad():
                # outputs, _ = model(lq)
                outputs = tile_process(model, lq, scale=4, tile_size=192, tile_pad=32)
                outputs = outputs.squeeze(0)
                for idx in range(outputs.shape[0]):
                    img_name = imgnames[name_idx] + '.png'
                    output = tensor2img(outputs[idx], rgb2bgr=True, min_max=(0, 1))
                    if save_imgs:
                        imwrite(output, osp.join(save_folder, subfolder_name, f'{img_name}'))
                    name_idx += 1
            del lq
            del outputs
            torch.cuda.empty_cache()
        '''
        with torch.no_grad():
            outputs, _ = model(imgs_lq)
            outputs = outputs.squeeze(0)
        # convert to numpy image
        for idx in range(outputs.shape[0]):
            img_name = imgnames[name_idx] + '.png'
            output = tensor2img(outputs[idx], rgb2bgr=True, min_max=(0, 1))
            # read GT image
            # img_gt = cv2.imread(osp.join(subfolder_gt, img_name), cv2.IMREAD_UNCHANGED)
            # crt_psnr = psnr_ssim.calculate_psnr(
            #     output, img_gt, crop_border=crop_border, test_y_channel=test_y_channel)
            # crt_ssim = psnr_ssim.calculate_ssim(
            # output, img_gt, crop_border=crop_border, test_y_channel=test_y_channel)
            # save
            if save_imgs:
                imwrite(output, osp.join(save_folder, subfolder_name, f'{img_name}'))
            # avg_psnr += crt_psnr
            # avg_ssim += crt_ssim
            # logger.info(f'{subfolder_name}--{img_name} - PSNR: {crt_psnr:.6f} dB. SSIM: {crt_ssim:.6f}')
            name_idx += 1
        '''
        #avg_flops = sum(flops)/len(flops) 
        avg_psnr /= name_idx
        logger.info(f'name_idx:{name_idx}')
        avg_ssim /= name_idx
        avg_psnr_l.append(avg_psnr)
        avg_ssim_l.append(avg_ssim)
        #FLOPs.append(avg_flops)

    for folder_idx, subfolder_name in enumerate(subfolder_names):
        logger.info(f'Folder {subfolder_name} - Average PSNR: {avg_psnr_l[folder_idx]:.6f} dB. Average SSIM: {avg_ssim_l[folder_idx]:.6f}. ')#Average FLOPS: {FLOPs[folder_idx]:.6f}.')

    # logger.info(f'Average PSNR: {sum(avg_psnr_l) / len(avg_psnr_l):.6f} dB ' f'for {len(subfolder_names)} clips. ')
    # logger.info(f'Average SSIM: {sum(avg_ssim_l) / len(avg_ssim_l):.6f}  '
    # f'for {len(subfolder_names)} clips. ')
    # logger.info(f'Average FLOPS: {sum(FLOPs) / len(FLOPs):.6f}  '
    # f'for {len(subfolder_names)} clips. ')


if __name__ == '__main__':

    main()
