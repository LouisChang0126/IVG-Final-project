import json
import numpy as np
import argparse
from pathlib import Path

def create_exact_transforms(camera_path_filename, output_dir, image_dir="images", extension="jpg", scale=1.0):
    """
    直接讀取 camera_path.json 的 keyframes，不經過任何插值運算，
    將矩陣原封不動地轉存為 transforms.json。
    """
    camera_path_file = Path(camera_path_filename)
    output_dir = Path(output_dir)
    
    if not camera_path_file.exists():
        print(f"錯誤: 找不到檔案 {camera_path_file}")
        return

    print(f"正在讀取: {camera_path_filename} (Direct Mode)")
    with open(camera_path_file, "r") as f:
        data = json.load(f)

    # 1. 直接讀取 keyframes 列表
    # 這是解決偏移的關鍵：我們只相信 keyframes 裡紀錄的數值
    keyframes = data.get("keyframes", [])
    
    if not keyframes:
        print("錯誤: JSON 中找不到 'keyframes' 列表")
        return

    num_frames = len(keyframes)
    print(f"偵測到 {num_frames} 個關鍵影格，將直接轉換。")

    # 2. 取得全域解析度 (若 keyframe 內無個別設定)
    render_w = int(data.get("render_width", 1920) * scale)
    render_h = int(data.get("render_height", 1080) * scale)
    
    # 嘗試計算全域焦距 (從 FOV 反推)，作為備用
    # Nerfstudio 存的是 Vertical FOV
    def fov2focal(fov, pixels):
        return pixels / (2 * np.tan((fov / 2) * np.pi / 180))
    
    global_fov = data.get("default_fov", 75.0) # 預設值
    
    frames_data = []

    # 3. 遍歷每一個 Keyframe
    for i, kf in enumerate(keyframes):
        # 取得矩陣 (camera_path 通常存成 1D list [16])
        matrix_raw = kf.get("matrix") or kf.get("camera_to_world")
        
        if matrix_raw is None:
            print(f"警告: 第 {i} 幀沒有矩陣數據，跳過。")
            continue
            
        # 轉成 4x4 格式
        c2w = np.array(matrix_raw).reshape(4, 4).tolist()

        # 處理相機內參
        # 優先用 keyframe 自己的設定，沒有則用全域
        fov = kf.get("fov", global_fov)
        aspect = kf.get("aspect", float(render_w) / float(render_h))
        
        # 如果 aspect 不是 1.0 (正方形)，需要反推真實寬高
        # 但通常訓練用的 transforms.json 我們直接用 render_width/height 即可
        w = render_w
        h = render_h
        
        # 計算焦距
        fl_y = fov2focal(fov, h)
        fl_x = fl_y # 假設像素是正方形
        
        cx = w / 2.0
        cy = h / 2.0

        # 檔名對應 (frame_00001.jpg)
        file_path = f"./{image_dir}/{i:05d}.{extension}"

        frame_entry = {
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "file_path": file_path,
            "transform_matrix": c2w
        }
        frames_data.append(frame_entry)

    # 4. 輸出 JSON
    json_output = {
        "camera_model": "OPENCV",
        "orientation_override": "none",
        # 建議: 如果您有從 raw.glb 轉出的點雲，這裡記得加上 "ply_file_path": "points3D.ply",
        "frames": frames_data
    }

    output_file = output_dir / "transforms.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(json_output, f, indent=4)
        
    print("-" * 30)
    print(f"成功！已生成無誤差的 transforms.json: {output_file}")
    print(f"解析度: {render_w}x{render_h}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert camera_path.json to transforms.json directly without interpolation")
    parser.add_argument("--camera_path", required=True, help="Path to your camera_path.json")
    parser.add_argument("--output_dir", required=True, help="Directory to save transforms.json")
    parser.add_argument("--ext", default="jpg", help="Image extension (e.g. jpg, png)")
    parser.add_argument("--scale", type=float, default=1.0, help="Resolution scaling factor (e.g., 4.0 for 4x SR)")
    
    args = parser.parse_args()

    create_exact_transforms(args.camera_path, args.output_dir, extension=args.ext, scale=args.scale)