# -*- coding: utf-8 -*-
"""
SegFormer 实物验证脚本（Jetson Orin / 服务器通用）

前提提醒：
    当前权重是用 --crop（按 mask 外接框裁剪）训出来的，训练时物体在画面里
    占很大比例。所以验证时请让【物体尽量占满画面】；如果物体很小、离得很远，
    效果会明显下降（这属于训练/推理尺度不一致，不是模型坏了）。

用法：
    # 摄像头实时验证（整图输入）
    python3 infer_live.py --model ~/dataset_seg/segformer_final --source 0

    # 让物体占不满画面时：手动框选 ROI，只识别框内
    python3 infer_live.py --model ~/dataset_seg/segformer_final --source 0 --mode roi

    # 单张图片
    python3 infer_live.py --model ~/dataset_seg/segformer_final --source test.jpg --save out.png

    # SSH 无显示器：不弹窗，只跑 N 帧打印结果
    python3 infer_live.py --model ~/dataset_seg/segformer_final --source 0 --headless --max-frames 100

按键（非 headless 时）：q 退出 / s 保存快照 / r 重新框选 ROI
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (SegformerForSemanticSegmentation,
                          SegformerImageProcessor)

ID2LABEL = {0: 'background', 1: 'apple', 2: 'tuna_fish_can',
            3: 'plastic_bottle', 4: 'power_drill', 5: 'wood_block'}

# BGR（OpenCV 顺序），只给前景 5 类配色，背景不涂色
PALETTE_BGR = {
    1: (40, 40, 220),      # apple       红
    2: (220, 120, 40),     # tuna_can    蓝
    3: (220, 220, 60),     # bottle      青
    4: (40, 180, 250),     # drill       橙黄
    5: (70, 120, 180),     # wood        棕
}


def load_model(model_dir, size):
    proc = SegformerImageProcessor.from_pretrained(str(model_dir), do_reduce_labels=False)
    try:
        proc.size = {'height': size, 'width': size}
        proc.do_resize = True
    except Exception as e:                       # 不同 transformers 版本属性名可能不同
        print('[警告] 设置输入尺寸失败，将使用权重自带的尺寸:', e)
    print('processor 输入尺寸:', getattr(proc, 'size', None))

    model = SegformerForSemanticSegmentation.from_pretrained(str(model_dir))
    use_cuda = torch.cuda.is_available()
    model.to('cuda' if use_cuda else 'cpu').eval()
    print('设备:', 'cuda' if use_cuda else 'cpu')
    return proc, model, use_cuda


@torch.no_grad()
def predict(proc, model, pil_img, half):
    """返回 (pred numpy[H,W] 值为类id, 每类像素数字典)"""
    out = proc(images=pil_img, return_tensors='pt')
    pv = out['pixel_values'].to(next(model.parameters()).device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=half):
        logits = model(pixel_values=pv).logits
    h, w = pv.shape[-2:]
    lg = F.interpolate(logits.float(), size=(h, w), mode='bilinear', align_corners=False)
    pred = lg.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    total = pred.size
    counts = {c: int((pred == c).sum()) for c in range(len(ID2LABEL))}
    return pred, counts, total


def summarize(counts, total):
    fg = sum(counts[c] for c in range(1, len(ID2LABEL)))
    ratio = fg / max(total, 1)
    if fg == 0:
        return '未检出物体', ratio, None
    order = sorted(range(1, len(ID2LABEL)), key=lambda c: -counts[c])
    top = order[0]
    conf = counts[top] / max(fg, 1)
    txt = '%s (%.0f%%)' % (ID2LABEL[top], 100 * conf)
    if len(order) > 1 and counts[order[1]] > 0:
        txt += '  次优:%s %.0f%%' % (ID2LABEL[order[1]], 100 * counts[order[1]] / fg)
    return txt, ratio, top


def overlay(bgr, pred, alpha=0.55):
    """把预测 mask 半透明叠到原图上"""
    color = np.zeros_like(bgr)
    for c, col in PALETTE_BGR.items():
        color[pred == c] = col
    m = pred > 0
    out = bgr.copy()
    out[m] = (bgr[m] * (1 - alpha) + color[m] * alpha).astype(np.uint8)
    return out


def open_camera(src):
    import cv2
    if str(src).isdigit():
        idx = int(src)
        for backend in (getattr(cv2, 'CAP_V4L2', 0), cv2.CAP_ANY):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                return cap
            cap.release()
        # JetPack 上常见：需要走 gstreamer
        pipe = ('v4l2src device=/dev/video%d ! video/x-raw,width=1280,height=720 '
                '! videoconvert ! appsink' % idx)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap
        return None
    cap = cv2.VideoCapture(str(src))
    return cap if cap.isOpened() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='~/dataset_seg/segformer_final')
    ap.add_argument('--source', default='0', help='摄像头编号(0/1) 或图片路径')
    ap.add_argument('--mode', default='full', choices=['full', 'roi', 'center'],
                    help='full=整图 / roi=手动框选 / center=中心方形裁剪')
    ap.add_argument('--size', type=int, default=512, help='送进网络的尺寸，需与训练一致')
    ap.add_argument('--save', default=None, help='单图模式下保存结果图路径')
    ap.add_argument('--headless', action='store_true', help='不弹窗，只打印结果')
    ap.add_argument('--max-frames', type=int, default=0, help='>0 时跑这么多帧就退出')
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--warmup', type=int, default=30,
                    help='开局丢弃前 N 帧（相机自动曝光需要时间收敛，'
                         '首帧常是白屏/黑屏）。固定曝光相机可设 0')
    ap.add_argument('--min-std', type=float, default=10.0,
                    help='帧标准差低于此值视为空白帧并跳过（默认 10）')
    ap.add_argument('--roi', default=None,
                    help='固定 ROI，格式 x,y,w,h（如 80,40,480,480）。'
                         '指定后不再弹窗选择，可绕开 OpenCV 5.x 的 selectROI bug')
    args = ap.parse_args()

    try:
        import cv2
    except ImportError:
        print('[错误] 缺少 opencv，先装: pip3 install opencv-python')
        raise SystemExit(1)

    model_dir = Path(args.model).expanduser().resolve()
    if not (model_dir / 'config.json').exists():
        print('[错误] %s 下没有 config.json，检查路径' % model_dir)
        raise SystemExit(1)

    proc, model, use_cuda = load_model(model_dir, args.size)
    half = bool(args.fp16 and use_cuda)

    src = Path(args.source).expanduser()
    is_image = str(args.source).lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))

    roi = None
    if args.roi:
        try:
            roi = tuple(int(v) for v in args.roi.split(','))
            assert len(roi) == 4 and roi[2] > 0 and roi[3] > 0
            print('使用固定 ROI: x=%d y=%d w=%d h=%d' % roi)
        except Exception:
            print('[错误] --roi 格式应为 x,y,w,h，例如 --roi 80,40,480,480')
            raise SystemExit(1)
    frame_count = 0
    t0 = time.time()

    def fallback_center(bgr):
        H, W = bgr.shape[:2]
        s = min(H, W)
        return bgr[(H - s) // 2:(H - s) // 2 + s, (W - s) // 2:(W - s) // 2 + s]

    def handle(bgr):
        nonlocal roi
        if args.mode == 'roi':
            if roi is None:
                # 命令行给了固定 ROI 就直接用，避免弹窗
                if args.roi:
                    roi = tuple(args.roi)
                else:
                    try:
                        r = cv2.selectROI('选择ROI后按回车/空格确认', bgr,
                                          fromCenter=False, showCrosshair=True)
                        cv2.destroyWindow('选择ROI后按回车/空格确认')
                    except cv2.error as e:
                        # opencv 5.x 的 Qt 后端常在这里抛 "NULL window handler"
                        print('\n[警告] selectROI 弹窗失败（%s）' % str(e)[:80])
                        print('       自动改用【中心方形裁剪】。')
                        print('       如需指定固定区域，用 --roi x,y,w,h（如 --roi 80,40,480,480）\n')
                        roi = 'CENTER'
                    if roi != 'CENTER' and not (roi[2] > 0 and roi[3] > 0):
                        roi = (0, 0, bgr.shape[1], bgr.shape[0])
            if roi == 'CENTER':
                crop = fallback_center(bgr)
            else:
                x, y, w, h = [int(v) for v in roi]
                crop = bgr[y:y + h, x:x + w]
        elif args.mode == 'center':
            H, W = bgr.shape[:2]
            s = min(H, W)
            crop = bgr[(H - s) // 2:(H - s) // 2 + s, (W - s) // 2:(W - s) // 2 + s]
        else:
            crop = bgr

        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pred, counts, total = predict(proc, model, pil, half)
        # 把预测结果放大回 crop 尺寸再叠加
        pred_big = np.array(Image.fromarray(pred).resize(
            (crop.shape[1], crop.shape[0]), resample=Image.NEAREST))
        vis = overlay(crop, pred_big)
        txt, ratio, top = summarize(counts, total)
        return vis, txt, ratio, pred_big

    if is_image:
        bgr = cv2.imread(str(src))
        if bgr is None:
            print('[错误] 读不到图片:', src)
            raise SystemExit(1)
        vis, txt, ratio, pred = handle(bgr)
        print('识别结果: %s' % txt)
        print('前景占比: %.2f%%' % (100 * ratio))
        if ratio < 0.02:
            print('  [提示] 前景太少，基本没检到物体。让物体占满画面、或改用 --mode roi 框住它')
        if args.save:
            cv2.imwrite(args.save, vis)
            print('已保存:', args.save)
        if not args.headless:
            cv2.imshow('result', vis)
            cv2.waitKey(0)
        return

    cap = open_camera(args.source)
    if cap is None:
        print('[错误] 打不开摄像头:', args.source)
        print('  确认设备: ls /dev/video*   或改用图片模式 --source test.jpg')
        raise SystemExit(1)

    print('开始实时推理，按 q 退出 / s 保存快照 / r 重新框选')
    if args.warmup > 0:
        print('预热中，丢弃前 %d 帧（相机自动曝光收敛）...' % args.warmup)
    try:
        skipped = 0
        done = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                print('读帧失败')
                break

            # 预热：丢掉曝光未收敛的前若干帧
            if frame_count < args.warmup:
                frame_count += 1
                if frame_count == args.warmup:
                    print('预热完成，开始推理')
                continue

            # 空白帧（白屏/黑屏）保护：标准差过低说明画面没内容
            if args.min_std > 0 and float(bgr.std()) < args.min_std:
                skipped += 1
                if skipped % 30 == 1:
                    print('  [跳过] 空白帧 std=%.1f（累计 %d 帧）' % (bgr.std(), skipped))
                continue

            vis, txt, ratio, pred = handle(bgr)

            if args.headless:
                done += 1
                if done % 10 == 0:
                    fps = done / max(time.time() - t0, 1e-6)
                    print('frame %d | %s | fg %.2f%% | %.1f fps'
                          % (done, txt, 100 * ratio, fps))
                if args.max_frames and done >= args.max_frames:
                    break
                continue

            cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
            cv2.putText(vis, 'fg %.1f%%' % (100 * ratio), (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('SegFormer live', vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('s'):
                p = 'snap_%d.png' % int(time.time())
                cv2.imwrite(p, vis)
                print('已保存', p)
            if k == ord('r'):
                roi = None
            done += 1
            if args.max_frames and done >= args.max_frames:
                break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
    print('结束，有效帧 %d，跳过空白帧 %d' % (done, skipped))


if __name__ == '__main__':
    main()

