import psutil
import torch
import sys

def get_size(bytes, suffix="GB"):
    """輔助函式：將 Byte 轉換為 GB"""
    factor = 1024 ** 3
    return f"{bytes / factor:.2f} {suffix}"

print("="*40)
print("     硬體資源分析報告 (Hardware Info)")
print("="*40)

# 1. 分析系統記憶體 (RAM)
svmem = psutil.virtual_memory()
print(f"【系統記憶體 (RAM)】")
print(f"  - 總容量 (Total)    : {get_size(svmem.total)}")
print(f"  - 目前可用 (Available): {get_size(svmem.available)}  <-- 設定參考值")
print(f"  - 已使用 (Used)     : {get_size(svmem.used)} ({svmem.percent}%)")
print("-" * 40)

# 2. 分析顯卡記憶體 (VRAM)
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"【顯卡記憶體 (VRAM)】 - 偵測到 {device_count} 張顯卡")
    
    for i in range(device_count):
        # 取得顯卡名稱
        gpu_name = torch.cuda.get_device_name(i)
        # 取得記憶體資訊 (free_memory, total_memory)
        # 注意：這是 PyTorch 看到的可用量，通常比 nvidia-smi 顯示的略少一點點
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        used_mem = total_mem - free_mem
        
        print(f"  [GPU {i}]: {gpu_name}")
        print(f"  - 總容量 (Total)    : {get_size(total_mem)}")
        print(f"  - 目前可用 (Free)     : {get_size(free_mem)}   <-- 設定參考值")
        print(f"  - 已使用 (Used)     : {get_size(used_mem)}")
else:
    print("【顯卡記憶體 (VRAM)】")
    print("  無法偵測到 CUDA 裝置 (No GPU found)")

print("="*40)

# 給予建議
if torch.cuda.is_available():
    free_mem_gb = torch.cuda.mem_get_info(0)[0] / (1024**3)
    rec_vram = max(0, free_mem_gb - 2.0) # 預留 2GB 給系統畫面輸出或其他背景程式
    print(f"建議設定值：")
    print(f"MAX_RAM_GB  = {psutil.virtual_memory().available / (1024**3) - 2.0:.1f} (建議保留一點緩衝)")
    print(f"MAX_VRAM_GB = {rec_vram:.1f}")