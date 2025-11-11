# from pytorch_nndct.apis import torch_quantizer
import os
import glob
import torch
from tqdm import tqdm

import utility
import data
import model as model_module   # 避免與變數同名衝突
import loss
from option import args

from PIL import Image
import numpy as np
import cv2
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def get_model_flops(model_name, model, input_W, input_H):
    from torchprofiler import Profiler

    # Initialize your profiler
    profiler = Profiler(model)

    # Run for specified input shape
    profiler.run((1,3,input_H, input_W)) # Include batch_size. e.g. (1, 3, 224, 224)

    # Print summary
    profiler.print_summary()

    # You can also view the overall statistics respectively
    profiler.total_input
    profiler.total_output
    profiler.total_params
    profiler.total_flops()
    profiler.trainable_params

    print('Model name:', model_name)

def prepare(a, b, device):
    def _prepare(t):
        if t is not None:
            return t.to(device)
        return None
    return _prepare(a), _prepare(b)


def fast_finetune_model(mdl, loader, loss, device):
    mdl.eval()
    loss.start_log()
    with torch.no_grad():
        for batch, (lr, hr, _) in enumerate(loader.loader_train):
            lr, hr = prepare(lr, hr, device)
            sr = mdl(lr, 0)
            Loss = loss(sr, hr)
    loss.end_log(len(loader.loader_train))


def calib_model_train(mdl, loader, device):
    mdl.eval()
    with torch.no_grad():
        for batch, (lr, hr, _) in enumerate(loader.loader_train):
            lr, hr = prepare(lr, hr, device)
            sr = mdl(lr, 0)


def _tensor_to_uint8_hwc(t):
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return arr.astype(np.uint8)


def _safe_out_name(name):
    base = os.path.basename(str(name))
    root, _ = os.path.splitext(base)
    if len(root) == 0:
        root = "out"
    return f"{root}_SR.png"


def _ensure_outdir():
    save_dir = os.path.join('quantize_result', 'output')
    os.makedirs(save_dir, exist_ok=True)
    print(f"結果圖像將保存至: {save_dir}")
    return save_dir


# -------------------------
# A) 成對 dataloader（若 HR 存在）
# -------------------------
def test_model_paired(mdl, loader, device):
    """使用現有的 loader（需要 HR 檔案能被對上）。回傳處理到的張數。"""
    torch.set_grad_enabled(False)
    mdl.eval()
    self_scale = [2]
    save_dir = _ensure_outdir()
    count = 0

    with torch.no_grad():
        for _, d in enumerate(loader.loader_test):
            for idx_scale, _ in enumerate(self_scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, _ = prepare(lr, hr, device)  # 忽略 hr
                    sr = mdl(lr, idx_scale)
                    sr = utility.quantize(sr, 255)
                    out_img = _tensor_to_uint8_hwc(sr)

                    raw_name = filename[0] if isinstance(filename, (list, tuple)) else filename
                    out_path = os.path.join(save_dir, _safe_out_name(raw_name))
                    Image.fromarray(out_img).save(out_path)
                    count += 1
    return count


# -------------------------
# B) 無 GT：直接讀資料夾
# -------------------------
    """
    從資料夾直接讀 LR 影像，不依賴 dataloader / HR 檔。
    影像路徑：input_dir/*.png|jpg|jpeg
    """
def test_model_nogt_dir(mdl, device, input_dir, scale=2,
                        use_sharpen=True, sigma=1.2, amount=0.8, thresh=0):
    """
    從資料夾直接讀 LR 影像（無 GT），做推論並輸出到 quantize_result/inference_output。
    - 前處理：BGR、uint8、0~255（與專案一致）
    - 後處理：可選 Unsharp Mask 銳化
    """
    torch.set_grad_enabled(False)
    mdl.eval()
    save_dir = _ensure_outdir()

    # 收集影像
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
    img_paths = []
    for e in exts:
        img_paths += glob.glob(os.path.join(input_dir, e))
    img_paths = sorted(img_paths)

    if len(img_paths) == 0:
        print(f"[WARN] No-GT 模式：在資料夾找不到影像：{input_dir}")
        return 0

    print(f"[INFO] No-GT 模式：將推論 {len(img_paths)} 張影像，來源：{input_dir}")

    # ---- Unsharp Mask 定義 ----
    def unsharp(img_bgr, sigma=1.2, amount=1.0, thresh=0):
        blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
        sharp = cv2.addWeighted(img_bgr, 1 + amount, blur, -amount, 0)
        if thresh > 0:
            mask = (cv2.absdiff(img_bgr, blur) < thresh)
            sharp[mask] = img_bgr[mask]
        return sharp
    # ---------------------------

    with torch.no_grad():
        for p in tqdm(img_paths, ncols=80):
            # 讀圖：BGR、uint8、0~255
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[WARN] cannot read: {p}")
                continue

            # [H,W,3] → [1,3,H,W]、float32、0~255
            arr = bgr.transpose(2, 0, 1).astype(np.float32)
            lr = torch.from_numpy(arr).unsqueeze(0).to(device)

            # 前向（RCAN wrapper 的 idx_scale 通常用 0）
            sr = mdl(lr, 0)
            sr = utility.quantize(sr, 255)  # 對齊到 0~255

            # 轉為 BGR uint8
            out = sr.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            # 可選：細節強化
            if use_sharpen:
                out = unsharp(out, sigma=sigma, amount=amount, thresh=thresh)

            # 存圖：BGR → RGB
            out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            out_path = os.path.join(save_dir, _safe_out_name(p))
            Image.fromarray(out_rgb).save(out_path)

    return len(img_paths)



def main():
    if not checkpoint.ok:
        return

    # 1) 建立模型並載入浮點權重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model_module.Model(args, checkpoint)

    float_pt = os.path.join(args.float_model_path, 'model_float.pt')
    pretrain_state_dict = torch.load(float_pt, map_location='cpu')
    new_dict = {'model.' + k: v for k, v in pretrain_state_dict.items()}
    model_state_dict = net.state_dict()
    model_state_dict.update(new_dict)
    net.load_state_dict(model_state_dict)
    net = net.to(device)

    dummy_input = torch.randn([64, 3, 48, 48])

    # 2) 依 quant_mode 分支
    if args.quant_mode == 'float':
        # 優先嘗試成對 dataloader；若長度為 0，自動切到資料夾模式
        loader = data.Data(args)

        processed = test_model_paired(net, loader, device)
        if processed == 0:
            # 自動改為資料夾直讀（無需改 option.py）
            # 預設資料夾：../data/test_images，可自行改路徑
            default_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'test_images'))
            # 若有自定義資料夾，可放在 args.dir_demo 或 args.dir_data/nogt
            dir_nogt = getattr(args, 'dir_demo', None)
            if dir_nogt is None or not os.path.isdir(dir_nogt):
                candidate = os.path.join(getattr(args, 'dir_data', '../data'), 'nogt')
                dir_nogt = candidate if os.path.isdir(candidate) else default_dir

            print(f"[INFO] 成對測試集中沒有可用樣本，切換到 No-GT 資料夾模式：{dir_nogt}")
            _ = test_model_nogt_dir(net, device, dir_nogt, scale=getattr(args, 'scale', 2))

    elif args.quant_mode == 'calib':
        from pytorch_nndct.apis import torch_quantizer
        loader = data.Data(args)
        quantizer = torch_quantizer(args.quant_mode, net, (dummy_input,), device=device)
        quant_model = quantizer.quant_model
        if args.fast_finetune:
            _loss = loss.Loss(args, checkpoint).to(device)
            quantizer.fast_finetune(fast_finetune_model, (quant_model, loader, _loss, device))
        calib_model_train(quant_model, loader, device)
        quantizer.export_quant_config()

    else:
        from pytorch_nndct.apis import torch_quantizer
        if getattr(args, 'dump_xmodel', False):
            cpu_device = torch.device("cpu")
            net_cpu = net.to(cpu_device)
            input_cpu = torch.randn([1, 3, 360, 640])
            quantizer = torch_quantizer(args.quant_mode, net_cpu, (input_cpu,), device=cpu_device)
            quant_model = quantizer.quant_model
            if args.fast_finetune:
                quantizer.load_ft_param()
            output = quant_model(input_cpu, 2)
            quantizer.export_xmodel(output_dir='quantize_result/', deploy_check=True)
        else:
            loader = data.Data(args)
            quantizer = torch_quantizer(args.quant_mode, net, (dummy_input,), device=device)
            quant_model = quantizer.quant_model
            if args.fast_finetune:
                quantizer.load_ft_param()
            processed = test_model_paired(quant_model, loader, device)
            if processed == 0:
                default_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'test_images'))
                dir_nogt = getattr(args, 'dir_demo', None)
                if dir_nogt is None or not os.path.isdir(dir_nogt):
                    candidate = os.path.join(getattr(args, 'dir_data', '../data'), 'nogt')
                    dir_nogt = candidate if os.path.isdir(candidate) else default_dir
                print(f"[INFO] 成對測試集中沒有可用樣本，切換到 No-GT 資料夾模式：{dir_nogt}")
                _ = test_model_nogt_dir(quant_model, device, dir_nogt, scale=getattr(args, 'scale', 2))


if __name__ == '__main__':
    main()
