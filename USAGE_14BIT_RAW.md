# 如何輸入 14-bit .raw 檔案

本文檔說明如何在 RCAN 中使用 14-bit RAW 圖像文件。

## 快速開始

### 方法 1：檔名包含尺寸（推薦）

將您的 raw 檔案命名為包含尺寸的格式：
- 格式：`檔名_寬度x高度.raw` （例如：`image_1920x1080.raw`）
- 替代格式：`檔名_寬度_高度.raw` （例如：`image_1920_1080.raw`）

```bash
python code/test.py --dir_demo /path/to/raw/images --quant_mode float
```

### 方法 2：使用命令行參數

如果您的 raw 檔案名稱中沒有包含尺寸，可以通過命令行參數指定：

```bash
python code/test.py --dir_demo /path/to/raw/images \
    --raw_width 1920 \
    --raw_height 1080 \
    --raw_bit_depth 14 \
    --quant_mode float
```

## 參數說明

- `--raw_bit_depth`：RAW 圖像的位深度（預設值：14）
- `--raw_width`：RAW 圖像的寬度（像素為單位，如果檔名中有則可選）
- `--raw_height`：RAW 圖像的高度（像素為單位，如果檔名中有則可選）

## RAW 檔案格式要求

- **副檔名**：`.raw`
- **資料類型**：16-bit 無符號整數 (uint16) 用於 14-bit 資料
- **字節序**：Little-endian
- **排列方式**：Row-major order（寬度 × 高度 像素）
- **色彩空間**：灰度圖（如需要會自動轉換為 RGB）

## 使用範例

### 範例 1：單張圖像測試

假設您有一張 14-bit 的 RAW 圖像，尺寸為 640x480：

```bash
# 方法 1：檔名包含尺寸
# 檔案名稱：image_640x480.raw
python code/test.py --dir_demo ./raw_images --quant_mode float

# 方法 2：使用命令行參數
# 檔案名稱：image.raw
python code/test.py --dir_demo ./raw_images \
    --raw_width 640 \
    --raw_height 480 \
    --raw_bit_depth 14 \
    --quant_mode float
```

### 範例 2：批次處理多張圖像

如果您有多張相同尺寸的 RAW 圖像：

```bash
# 將所有 raw 檔案放在同一個資料夾
# 使用命名約定：image1_1920x1080.raw, image2_1920x1080.raw, etc.
python code/test.py --dir_demo ./raw_images --quant_mode float
```

### 範例 3：訓練模式

對於訓練數據集，按照以下結構組織您的檔案：

```
data/
  YourDataset/
    HR/
      image1_1920x1080.raw
      image2_1920x1080.raw
      ...
    LR_bicubic/
      X2/
        image1_960x540.raw
        image2_960x540.raw
        ...
```

然後運行：

```bash
python code/train.py \
    --data_train YourDataset \
    --raw_bit_depth 14 \
    --scale 2
```

## 技術細節

### 資料轉換

- 14-bit RAW 圖像會自動從 14-bit 範圍（0-16383）縮放到 8-bit 範圍（0-255）進行處理
- 縮放公式：`scaled_value = (raw_value / 16383) * 255`

### 色彩處理

- 灰度 RAW 圖像會根據 `--n_colors` 參數自動轉換
- 如果 `--n_colors 3`，灰度圖會被轉換為 RGB
- 如果 `--n_colors 1`，圖像保持灰度

### 輸出

- 處理後的圖像會保存為 PNG 格式
- 輸出目錄：`quantize_result/output/`
- 輸出檔名格式：`原檔名_SR.png`

## 測試您的設置

我們提供了一個測試腳本來驗證 RAW 檔案支援功能：

```bash
python test_raw_support.py
```

這個腳本會：
1. 測試檔名解析功能
2. 創建樣本 RAW 檔案
3. 測試讀取和轉換功能
4. 在 `/tmp/rcan_raw_test/` 創建測試檔案

## 支援的圖像格式

現在支援以下圖像格式：
- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)
- TIFF (.tif, .tiff)
- **RAW (.raw)** ← 新增！

## 常見問題

### Q: 我的 RAW 檔案是 12-bit 或 16-bit，可以使用嗎？

A: 可以！只需調整 `--raw_bit_depth` 參數：
```bash
# 12-bit RAW
python code/test.py --dir_demo ./raw_images --raw_bit_depth 12

# 16-bit RAW
python code/test.py --dir_demo ./raw_images --raw_bit_depth 16
```

### Q: 如何知道我的 RAW 檔案的正確尺寸？

A: 您可以通過檔案大小計算：
- 檔案大小（bytes）= 寬度 × 高度 × 2（因為是 16-bit）
- 例如：如果檔案大小是 1,228,800 bytes
  - 1,228,800 / 2 = 614,400 像素
  - 可能是 640 × 960 或 800 × 768 等

### Q: 處理後的圖像質量如何？

A: 程式會：
1. 保留原始的動態範圍資訊
2. 適當縮放到處理範圍
3. 輸出高質量的 PNG 檔案

### Q: 可以混合使用 RAW 和其他格式的圖像嗎？

A: 不建議。如果目錄中同時有 RAW 和其他格式的圖像，建議分開處理以避免尺寸不匹配的問題。

## 注意事項

1. **檔案命名**：強烈建議在檔名中包含尺寸資訊，這樣可以避免手動指定參數
2. **尺寸一致性**：訓練時，確保 LR 檔案的尺寸是 HR 檔案尺寸按比例縮放的結果
3. **記憶體使用**：高解析度的 RAW 檔案會佔用較多記憶體，建議根據您的硬體配置調整 `--patch_size` 和 `--batch_size`
4. **字節序**：目前僅支援 little-endian 格式，如果您的檔案是 big-endian，請先轉換

## 進階選項

### 自訂 RGB 範圍

如果您需要不同的處理範圍：

```bash
python code/test.py --dir_demo ./raw_images \
    --raw_bit_depth 14 \
    --rgb_range 255 \
    --quant_mode float
```

### 處理彩色 RAW（Bayer 格式）

目前版本支援灰度 RAW，如需處理 Bayer 格式的彩色 RAW，建議先使用專門的工具進行 demosaicing，然後將結果保存為標準格式或分離的 R/G/B 通道。

## 相關資源

- [RCAN 論文](https://arxiv.org/abs/1807.02758)
- [PyTorch 文檔](https://pytorch.org/docs/)
- [NumPy 文檔](https://numpy.org/doc/)

## 問題回報

如果您遇到問題或有建議，請在 GitHub Issues 中回報。
