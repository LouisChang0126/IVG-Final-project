# SuperGaussian 實作 Pipeline

**給 AI 助手的專案背景 (Context):**
本專案實作了論文 *SuperGaussian: Repurposing Video Models for 3D Super Resolution* 中描述的工作流程。目標是將 **Nerfstudio** (用於 3DGS 渲染/訓練) 與 **視訊超解析度 (Video Super-Resolution, VSR)** 模型結合。

整個自動化流程由 `super_gaussian.py` 腳本統一調度，共分為 4 個步驟。

-----

## 🏗 流程架構 (Pipeline Architecture)

此工作流程透過「2D 投影」與「放大」的過程，將低解析度的 3DGS 模型轉換為高解析度版本。

1.  **Step 1: 資料生成 (3D $\to$ 2D)**

      * **輸入:** 已訓練好的 Nerfstudio config (`config.yml`) + 使用者自訂的相機路徑 (`camera_path.json`)。
      * **動作:** 使用 `ns-render` 渲染出一系列低解析度圖片。
      * **關鍵點:** 程式會直接從相機路徑數據生成同步的 `transforms.json`，確保相機位姿 (Pose) 的絕對精確。

2.  **Step 2: 視訊超解析度 (2D $\to$ 2D SR)**

      * **輸入:** 低解析度 PNG 序列。
      * **動作:** 應用視訊超解析度模型 (例如: MIA-VSR, IART)。
      * **輸出:** 高解析度 PNG 序列。
      * **⚠️ 待辦事項 (TODO):** 目前腳本中使用的是 *Mock Resize (模擬縮放)* 函式。你需要協助使用者將其替換為實際呼叫 VSR 模型推論 (Inference) 的指令。

3.  **Step 3: 資料整合 (SR $\to$ 3D Ready)**

      * **動作:** 讀取 Step 1 產生的 `transforms.json`。
      * **邏輯:** 根據放大倍率 (預設 x4) 等比放大相機內參 (`fl_x`, `fl_y`, `cx`, `cy`, `w`, `h`)。
      * **輸出:** 符合 Nerfstudio 標準格式的訓練資料夾。

4.  **Step 4: 3D 重建 (High-Res Training)**

      * **動作:** 觸發 `ns-train splatfacto` 指令。
      * **設定:** 強制關閉相機優化 (`--pipeline.datamanager.camera-optimizer.mode off`)，完全信任由 SR 生成的路徑與影像。

-----

## 🛠 依賴與環境 (Dependencies)

  * **主要環境:** 安裝了 **Nerfstudio** 的 `conda` 環境。
  * **次要環境 (選用):** 如果 VSR 模型需要不同的 CUDA 版本或依賴庫，應在 Step 2 中透過 `subprocess` 呼叫外部環境執行。
  * **核心腳本:** `super_gaussian.py`

## 📂 檔案結構 (File Structure)

```text
.
├── README.md               <-- 你現在在這裡
├── super_gaussian.py       <-- 主控自動化腳本
├── camera_path.json        <-- 從 Nerfstudio Viewer 匯出的相機路徑
└── outputs/                <-- 生成產物目錄
    └── my_project/
        ├── 01_low_res_images/      <-- Step 1 渲染出的低解析圖片
        ├── 02_high_res_images/     <-- Step 2 放大後的高解析圖片
        ├── 03_training_data/       <-- Step 3 整理好給 Nerfstudio 用的資料
        ├── transforms_lowres.json  <-- 原始相機位姿
        └── transforms.json         <-- 放大後的相機位姿
```

-----

## 🚀 AI 助手的執行任務 (Action Items)

如果使用者要求你協助執行或修改此 Pipeline，請依循以下檢查點：

### 1\. 確認 VSR 整合 (Step 2)

檢查 `super_gaussian.py` 中的 `step2_super_resolution` 方法：

  * **現狀:** 它正在使用 `PIL.Image.resize` (Mock/模擬)。
  * **目標:** 將該區塊替換為 `subprocess.run([...])` 指令，以呼叫外部 VSR 模型腳本。
  * **檢查:** 確保輸入/輸出目錄參數正確對應 `self.low_res_dir` 與 `self.high_res_dir`。

### 2\. 檢查座標轉換邏輯 (Step 3)

  * 確保 JSON 的縮放邏輯與放大倍率 (Scale Factor) 一致。
  * **關鍵變數:** `w`, `h`, `fl_x`, `fl_y`, `cx`, `cy` 都必須乘以 Scale。

### 3\. 執行指令

若要執行完整流程，請根據使用者輸入產生如下指令：

```bash
python super_gaussian.py \
  --config {低解析模型CONFIG路徑} \
  --camera_path {相機路徑JSON} \
  --output {輸出專案名稱} \
  --scale 4
```

-----

## 🐛 疑難排解知識庫 (Troubleshooting)

  * **問題:** *重建出的高解析度 3D 模型看起來模糊或錯位。*

      * **原因:** `transforms.json` 中的相機內參 (Intrinsics) 可能沒有正確匹配放大後的解析度。
      * **解法:** 檢查 Step 3 的數學運算：`新焦距 = 舊焦距 * 放大倍率`。

  * **問題:** *Step 4 失敗，顯示 "No images found"。*

      * **原因:** `transforms.json` 裡的檔名與 `03_training_data/images` 裡的實際檔名不符。
      * **解法:** 確保 Step 2 的 VSR 模型在輸出圖片時保留了完全一致的檔名 (例如 `00001.png`)，沒有掉幀或改名。

  * **問題:** *3D 畫面出現鬼影 (Ghosting artifacts)。*

      * **原因:** 視訊超解析度模型產生了時間不一致性 (Temporal Inconsistency)。
      * **解法:** 3DGS 訓練 (Step 4) 本身具有過濾功能，但如果鬼影嚴重，請嘗試更換 VSR 模型，或增加 3DGS 的訓練迭代次數。