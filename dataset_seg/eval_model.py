# -*- coding: utf-8 -*-
"""
模型评估 + 训练曲线绘制（服务器上跑，4090 几十秒出结果）

做两件事：
  A. 评估：在验证集(180张，未参与训练)上算
       - 分割指标：每类 IoU / Dice / 像素准确率、前景 mIoU、总体 mIoU
       - 分类指标：用预测 mask 的像素占比判定物体类别，输出
         整体正确率、每类正确率、5x5 混淆矩阵
  B. 绘图：读 train_log.csv，画出 loss 曲线 / miou_fg 曲线 / 5类 IoU 曲线

用法:
    python3 eval_model.py                          # 默认评估 ckpt_v2/best
    python3 eval_model.py --ckpt ckpt_v2/best --split val
    python3 eval_model.py --plot-only              # 只画图，不评估
    python3 eval_model.py --split all              # 全部 900 张都评（含训练集，仅参考）

注意:
    训练用了 --crop，评估必须同样 crop，否则尺度不一致、指标会虚低。
    本脚本默认 --crop，与训练保持一致。
"""
import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (SegformerForSemanticSegmentation,
                          SegformerImageProcessor)

DEFAULT_MAP = "apple=1,tuna_fish_can=2,plastic_bottle=3,power_drill=4,wood_block=5"
ID2LABEL = {0: "background"}
for _it in DEFAULT_MAP.split(','):
    _k, _v = _it.split('=')
    ID2LABEL[int(_v)] = _k.strip()
NUM_CLASSES = len(ID2LABEL)
CLASS_NAMES = [ID2LABEL[c] for c in range(1, NUM_CLASSES)]
SEED = 42          # 必须与训练时一致，才能复现同样的 720/180 划分


# ----------------------------- 工具 -----------------------------
def stratified_split(stems, val_ratio=0.2, seed=SEED):
    """与 train_v2.py 完全一致的分层划分"""
    rng = random.Random(seed)
    by = defaultdict(list)
    for s in stems:
        by[s.rsplit('_', 1)[0]].append(s)
    train, val = [], []
    for c in sorted(by):
        it = by[c][:]
        rng.shuffle(it)
        k = max(1, int(round(len(it) * val_ratio)))
        val += it[:k]
        train += it[k:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def crop_by_mask(img, lbl, margin=0.15):
    """按前景外接框扩边裁剪（确定性版本，不做随机外扩）"""
    a = np.array(lbl)
    ys, xs = np.where(a > 0)
    if len(ys) == 0:
        return img, lbl
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = a.shape
    dh, dw = int((y1 - y0) * margin) + 8, int((x1 - x0) * margin) + 8
    y0, y1 = max(0, y0 - dh), min(h, y1 + dh)
    x0, x1 = max(0, x0 - dw), min(w, x1 + dw)
    return img.crop((x0, y0, x1, y1)), lbl.crop((x0, y0, x1, y1))


def compute_metrics_from_hist(hist):
    """hist[i][j] = 真实i被预测为j的像素数"""
    hist = np.asarray(hist, dtype=np.float64)
    inter = np.diag(hist)
    union = hist.sum(0) + hist.sum(1) - inter
    iou = inter / np.maximum(union, 1)
    dice = 2 * inter / np.maximum(hist.sum(0) + hist.sum(1), 1)
    acc = inter / np.maximum(hist.sum(1), 1)      # 每类像素准确率
    return iou, dice, acc


# ----------------------------- 评估 -----------------------------
@torch.no_grad()
def evaluate(ckpt_dir, root, split='val', crop=True, size=512, bs=8, workers=4):
    ckpt_dir = Path(ckpt_dir).expanduser().resolve()
    if not (ckpt_dir / 'config.json').exists():
        print('[错误] %s 下没有 config.json' % ckpt_dir)
        raise SystemExit(1)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    proc = SegformerImageProcessor.from_pretrained(str(ckpt_dir), do_reduce_labels=False)
    try:
        proc.size = {'height': size, 'width': size} if crop else {'height': 544, 'width': 960}
        proc.do_resize = True
    except Exception:
        pass
    model = SegformerForSemanticSegmentation.from_pretrained(str(ckpt_dir))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    print('设备:', device, '| 输入尺寸:', getattr(proc, 'size', None))

    stems = sorted(p.stem for p in (root / 'labels').glob('*.png'))
    tr, va = stratified_split(stems, 0.2, SEED)
    use = {'val': va, 'train': tr, 'all': stems}[split]
    print('评估样本: %d (%s)' % (len(use), split))

    img_paths = {p.stem: p for p in (root / 'images').glob('*')}
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    cls_correct = defaultdict(int)
    cls_total = defaultdict(int)
    cls_conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)   # 分类混淆矩阵(1..5)
    undetected = 0

    for i, stem in enumerate(use):
        img = Image.open(img_paths[stem]).convert('RGB')
        lbl = Image.open(root / 'labels' / (stem + '.png')).convert('L')
        if crop:
            img, lbl = crop_by_mask(img, lbl)

        out = proc(images=img, segmentation_maps=lbl, return_tensors='pt')
        pv = out['pixel_values'].to(device)
        lb = out['labels']
        if lb.dim() == 3 and lb.shape[0] == 1:
            lb = lb.squeeze(0)
        lb = lb.long()

        with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
            logits = model(pixel_values=pv).logits
        lg = F.interpolate(logits.float(), size=lb.shape[-2:],
                           mode='bilinear', align_corners=False)
        pred = lg.argmax(dim=1)[0].cpu().numpy().astype(np.int64)
        gt = lb.numpy().astype(np.int64)

        k = (gt >= 0) & (gt < NUM_CLASSES)
        hist += np.bincount((NUM_CLASSES * gt[k] + pred[k]).ravel(),
                            minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)

        # 分类判定：预测 mask 里各类像素占比最大的那个类
        true_cls = None
        base = stem.rsplit('_', 1)[0]
        for name, cid in ((ID2LABEL[c], c) for c in range(1, NUM_CLASSES)):
            if base == name:
                true_cls = cid
                break
        fg_counts = {c: int((pred == c).sum()) for c in range(1, NUM_CLASSES)}
        if sum(fg_counts.values()) == 0:
            pred_cls = 0                     # 未检出
            undetected += 1
        else:
            pred_cls = max(fg_counts, key=fg_counts.get)
        if true_cls is not None:
            cls_total[true_cls] += 1
            if pred_cls == true_cls:
                cls_correct[true_cls] += 1
            cls_conf[true_cls][pred_cls] += 1

        if (i + 1) % 100 == 0:
            print('  已处理 %d/%d' % (i + 1, len(use)))

    iou, dice, acc = compute_metrics_from_hist(hist)
    res = {
        'split': split, 'num_samples': len(use), 'crop': crop,
        'miou_all': float(iou.mean()), 'miou_fg': float(iou[1:].mean()),
        'mean_dice_fg': float(dice[1:].mean()),
        'per_class': {
            ID2LABEL[c]: {'iou': float(iou[c]), 'dice': float(dice[c]),
                          'pixel_acc': float(acc[c])}
            for c in range(NUM_CLASSES)
        },
        'classification': {
            'overall_acc': float(sum(cls_correct.values()) / max(sum(cls_total.values()), 1)),
            'per_class_acc': {ID2LABEL[c]: float(cls_correct[c] / max(cls_total[c], 1))
                              for c in sorted(cls_total)},
            'per_class_correct': {ID2LABEL[c]: int(cls_correct[c])
                                  for c in sorted(cls_total)},
            'per_class_total': {ID2LABEL[c]: int(cls_total[c]) for c in sorted(cls_total)},
            'undetected': int(undetected),
            'confusion_matrix': {          # 行=真实, 列=预测
                ID2LABEL[t]: {('none' if p == 0 else ID2LABEL[p]): int(cls_conf[t][p])
                              for p in range(NUM_CLASSES) if cls_conf[t][p] > 0}
                for t in sorted(cls_total)
            },
        },
    }
    return res


def print_report(res):
    print('\n' + '=' * 68)
    print('评估结果  (%s集, %d 张, crop=%s)' % (res['split'], res['num_samples'], res['crop']))
    print('=' * 68)
    print('\n--- 分割指标 ---')
    print('%-16s %8s %8s %10s' % ('类别', 'IoU', 'Dice', '像素准确率'))
    for c in range(NUM_CLASSES):
        m = res['per_class'][ID2LABEL[c]]
        print('%-16s %8.4f %8.4f %10.4f' % (ID2LABEL[c], m['iou'], m['dice'], m['pixel_acc']))
    print('\n前景 mIoU  : %.4f      总体 mIoU: %.4f' % (res['miou_fg'], res['miou_all']))
    print('前景 Dice  : %.4f' % res['mean_dice_fg'])

    cl = res['classification']
    print('\n--- 分类正确率（用预测 mask 的像素占比判定类别）---')
    print('整体正确率: %.2f%%   (%d/%d)' % (
        100 * cl['overall_acc'],
        sum(cl['per_class_correct'].values()), sum(cl['per_class_total'].values())))
    if cl['undetected']:
        print('未检出(预测全背景): %d 张' % cl['undetected'])
    print('\n%-16s %10s   %s' % ('类别', '正确率', '正确/总数'))
    for c in range(1, NUM_CLASSES):
        n = ID2LABEL[c]
        print('%-16s %9.1f%%   %d/%d' % (
            n, 100 * cl['per_class_acc'][n],
            cl['per_class_correct'][n], cl['per_class_total'][n]))

    print('\n--- 分类混淆矩阵（行=真实, 列=预测）---')
    hdr = ['none'] + CLASS_NAMES
    print('%-16s' % '' + ''.join('%10s' % h[:9] for h in hdr))
    for c in range(1, NUM_CLASSES):
        row = cl['confusion_matrix'].get(ID2LABEL[c], {})
        print('%-16s' % ID2LABEL[c] + ''.join('%10d' % row.get(h, 0) for h in hdr))
    print('=' * 68)


# ----------------------------- 绘图 -----------------------------
def ascii_plot(ys, title, width=60, height=10):
    """matplotlib 不可用时的兜底：用字符画折线"""
    ys = [y for y in ys if y is not None]
    if not ys:
        return '%s: (无数据)' % title
    lo, hi = min(ys), max(ys)
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    grid = [[' '] * width for _ in range(height)]
    for xi in range(width):
        idx = int(xi * (len(ys) - 1) / max(width - 1, 1))
        yi = int((ys[idx] - lo) / (hi - lo) * (height - 1))
        grid[height - 1 - yi][xi] = '*'
    lines = ['%s   max=%.4f' % (title, hi)]
    for r in grid:
        lines.append('   |' + ''.join(r))
    lines.append('   +' + '-' * width)
    lines.append('    min=%.4f' % lo)
    return '\n'.join(lines)


def plot_curves(csv_path, out_png):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print('[警告] 找不到 %s，跳过绘图' % csv_path)
        return None
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        print('[警告] %s 是空的' % csv_path)
        return None

    def col(name):
        return [float(r[name]) for r in rows if r.get(name) not in (None, '')]

    ep = [int(r['epoch']) for r in rows]
    series = {
        'train_loss': col('train_loss'), 'val_loss': col('val_loss'),
        'miou_fg': col('miou_fg'), 'miou_all': col('miou_all'),
    }
    for c in range(1, NUM_CLASSES):
        series['iou_' + ID2LABEL[c]] = col('iou_' + ID2LABEL[c])

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print('[警告] matplotlib 不可用 (%s)，改用字符图' % e)
        for k in ['train_loss', 'val_loss', 'miou_fg']:
            if series[k]:
                print(ascii_plot(series[k], k))
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0][0]
    if series['train_loss']:
        ax.plot(ep[:len(series['train_loss'])], series['train_loss'], label='train_loss')
    if series['val_loss']:
        ax.plot(ep[:len(series['val_loss'])], series['val_loss'], label='val_loss')
    ax.set_title('Loss'); ax.set_xlabel('epoch'); ax.legend(); ax.grid(alpha=.3)

    ax = axes[0][1]
    if series['miou_fg']:
        ax.plot(ep[:len(series['miou_fg'])], series['miou_fg'], label='miou_fg (foreground)', lw=2)
    if series['miou_all']:
        ax.plot(ep[:len(series['miou_all'])], series['miou_all'], label='miou_all', ls='--')
    ax.set_title('mIoU'); ax.set_xlabel('epoch'); ax.legend(); ax.grid(alpha=.3)

    ax = axes[1][0]
    for c in range(1, NUM_CLASSES):
        v = series['iou_' + ID2LABEL[c]]
        if v:
            ax.plot(ep[:len(v)], v, label=ID2LABEL[c])
    ax.set_title('Per-class IoU'); ax.set_xlabel('epoch'); ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = axes[1][1]
    best = max(rows, key=lambda r: float(r['miou_fg']))
    txt = ['Best epoch: %s' % best['epoch'], 'miou_fg : %.4f' % float(best['miou_fg']),
           'miou_all: %.4f' % float(best['miou_all']), '', 'Per-class IoU:']
    for c in range(1, NUM_CLASSES):
        txt.append('  %-16s %.4f' % (ID2LABEL[c], float(best['iou_' + ID2LABEL[c]])))
    ax.axis('off')
    ax.text(0.05, 0.95, '\n'.join(txt), va='top', family='monospace', fontsize=11)

    plt.tight_layout()
    plt.savefig(str(out_png), dpi=130)
    print('曲线图已保存:', out_png)
    return str(out_png)


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=None)
    ap.add_argument('--ckpt', default=None, help='默认 根目录下的 ckpt_v2/best')
    ap.add_argument('--split', default='val', choices=['val', 'train', 'all'])
    ap.add_argument('--crop', dest='crop', action='store_true', default=True)
    ap.add_argument('--no-crop', dest='crop', action='store_false')
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--plot-only', action='store_true')
    ap.add_argument('--csv', default=None, help='默认 根目录下 ckpt_v2/train_log.csv')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from find_root import resolve_root
    root, why = resolve_root(args.root)
    print('数据集根目录: %s   (%s)' % (root, why))

    out_dir = Path(args.out) if args.out else root / 'eval_out'
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = args.ckpt or str(root / 'ckpt_v2' / 'best')
    csv_path = args.csv or str(root / 'ckpt_v2' / 'train_log.csv')

    if not args.plot_only:
        res = evaluate(ckpt, root, split=args.split, crop=args.crop)
        print_report(res)
        jp = out_dir / ('eval_%s.json' % args.split)
        json.dump(res, open(jp, 'w'), indent=2, ensure_ascii=False)
        print('\n结果已保存:', jp)

    png = out_dir / 'training_curves.png'
    plot_curves(csv_path, png)


if __name__ == '__main__':
    main()
