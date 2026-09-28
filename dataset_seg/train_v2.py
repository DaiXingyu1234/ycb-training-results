# -*- coding: utf-8 -*-
"""
SegFormer-B0 语义分割训练（手写训练循环版）

与 Trainer 版的区别：不依赖 accelerate / sklearn / evaluate，
只用 torch + transformers 的模型与处理器，避免在 Jetson(aarch64, py3.8)
上反复踩依赖版本坑。

功能：
  - 加权 CE + Dice（背景像素占绝对多数时防止模型躺平预测全背景）
  - 每 epoch 计算 6 类混淆矩阵，输出每类 IoU，按【前景 mIoU】保存最优模型
  - AMP(fp16) 自动按 CUDA 可用性开关
  - 可选 --crop：按 mask 外接框裁剪，把小目标变成大目标
  - 训练日志写 train_log.csv

用法：
  python3 train_v2.py --epochs 1 --bs 2 --lr 3e-5 --workers 2 --crop --fp16   # 冒烟测试
  python3 train_v2.py --epochs 100 --bs 2 --lr 3e-5 --workers 2 --crop --fp16 # 正式训练
  python3 train_v2.py --limit 40 --epochs 2                                   # 极小样本快速验证
"""
import argparse
import csv
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformers import (SegformerImageProcessor,
                          SegformerForSemanticSegmentation)

# ------------------------- 配置区 -------------------------
DEFAULT_MAP = "apple=1,tuna_fish_can=2,plastic_bottle=3,power_drill=4,wood_block=5"
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
IMG_H, IMG_W = 544, 960        # 不裁剪时用；必须都是 32 的倍数
BG_WEIGHT = 0.1                # 背景类 loss 权重
DICE_WEIGHT = 1.0
IGNORE_INDEX = 255
# ---------------------------------------------------------

ID2LABEL = {0: "background"}
for _it in DEFAULT_MAP.split(','):
    _k, _v = _it.split('=')
    ID2LABEL[int(_v)] = _k.strip()
NUM_CLASSES = len(ID2LABEL)


# ----------------------------- 数据集 -----------------------------
def _jitter(img):
    """轻量光度增强，不引入额外依赖"""
    from torchvision.transforms import functional as TF
    img = TF.adjust_brightness(img, 1 + random.uniform(-0.25, 0.25))
    img = TF.adjust_contrast(img, 1 + random.uniform(-0.25, 0.25))
    img = TF.adjust_saturation(img, 1 + random.uniform(-0.25, 0.25))
    return img


class SegDataset(Dataset):
    def __init__(self, stems, root, processor, train=False, crop=False):
        self.stems = stems
        self.root = Path(root)
        self.processor = processor
        self.train = train
        self.crop = crop
        self.img_paths = {p.stem: p for p in (self.root / 'images').glob('*')}

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem = self.stems[i]
        img = Image.open(self.img_paths[stem]).convert('RGB')   # 防 RGBA 四通道
        lbl = Image.open(self.root / 'labels' / (stem + '.png')).convert('L')

        if self.crop:
            # 小目标救星：按前景外接框扩边裁剪，再交给 processor 统一 resize
            a = np.array(lbl)
            ys, xs = np.where(a > 0)
            if len(ys):
                y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
                m = 0.15
                h, w = a.shape
                dh, dw = int((y1 - y0) * m) + 8, int((x1 - x0) * m) + 8
                y0, y1 = max(0, y0 - dh), min(h, y1 + dh)
                x0, x1 = max(0, x0 - dw), min(w, x1 + dw)
                if random.random() < 0.5:                       # 随机再外扩，做尺度扰动
                    ey, ex = (y1 - y0) // 4, (x1 - x0) // 4
                    y0, y1 = max(0, y0 - ey), min(h, y1 + ey)
                    x0, x1 = max(0, x0 - ex), min(w, x1 + ex)
                img = img.crop((x0, y0, x1, y1))
                lbl = lbl.crop((x0, y0, x1, y1))

        if self.train:
            # 几何变换必须图与标签同步；标签一律 NEAREST，禁止插值
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.3:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                lbl = lbl.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() < 0.4:
                ang = random.uniform(-10, 10)
                img = img.rotate(ang, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
                lbl = lbl.rotate(ang, resample=Image.NEAREST, fillcolor=0)
            if random.random() < 0.5:                           # 只作用于图像
                img = _jitter(img)

        out = self.processor(images=img, segmentation_maps=lbl, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in out.items()}
        lb = item['labels'].long()
        if lb.dim() == 3 and lb.shape[0] == 1:                  # 兼容 (1,H,W)
            lb = lb.squeeze(0)
        item['labels'] = lb
        return item


# ----------------------------- 损失 -----------------------------
class CEDiceLoss(nn.Module):
    """加权 CE + Dice。背景像素占压倒多数时，纯 CE 会让模型直接躺平预测全背景。"""

    def __init__(self, num_classes, bg_weight=BG_WEIGHT, dice_weight=DICE_WEIGHT):
        super().__init__()
        w = torch.ones(num_classes)
        w[0] = bg_weight
        self.register_buffer('cls_w', w)
        self.dice_weight = dice_weight

    def forward(self, logits, labels):
        # SegFormer 输出是输入的 1/4，必须先上采样到标签尺寸再算 loss
        logits = F.interpolate(logits, size=labels.shape[-2:],
                               mode='bilinear', align_corners=False)
        w = self.cls_w.to(logits.device)
        ce = F.cross_entropy(logits, labels, weight=w, ignore_index=IGNORE_INDEX)

        p = torch.softmax(logits, dim=1)
        y = labels.clamp(min=0)
        oh = torch.zeros_like(p).scatter_(1, y.unsqueeze(1), 1.0)
        wv = w.view(1, -1, 1, 1)
        inter = (p * oh * wv).sum(dim=(0, 2, 3))
        union = ((p + oh) * wv).sum(dim=(0, 2, 3))
        dice = 1 - (2 * inter / union.clamp(min=1e-6)).mean()
        return ce + self.dice_weight * dice


# ----------------------------- 指标 -----------------------------
@torch.no_grad()
def evaluate(model, loader, device, loss_fn, amp=False):
    model.eval()
    hist = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.float64)
    total_loss, nb = 0.0, 0
    for batch in loader:
        labels = batch.pop('labels').to(device)
        pv = batch['pixel_values'].to(device)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(pixel_values=pv).logits
            loss = loss_fn(logits, labels)
        total_loss += float(loss) * pv.size(0)
        nb += pv.size(0)

        lg = F.interpolate(logits.float(), size=labels.shape[-2:],
                           mode='bilinear', align_corners=False)
        preds = lg.argmax(dim=1).cpu()
        lbs = labels.cpu().long()
        k = (lbs >= 0) & (lbs < NUM_CLASSES)
        h = torch.bincount((NUM_CLASSES * lbs[k] + preds[k]).view(-1),
                           minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
        hist += h.double()

    inter = torch.diag(hist)
    union = hist.sum(0) + hist.sum(1) - inter
    iou = inter / union.clamp(min=1)
    res = {'loss': total_loss / max(nb, 1),
           'miou_all': float(iou.mean()),
           'miou_fg': float(iou[1:].mean())}
    for c in range(1, NUM_CLASSES):
        res['iou_' + ID2LABEL[c]] = float(iou[c])
    return res


# ----------------------------- 学习率 -----------------------------
def lr_at(step, total_steps, base_lr, warmup_steps):
    """warmup 线性上升 + 之后多项式衰减(0.9)，SegFormer 官方用的就是 poly"""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (1.0 - min(prog, 1.0)) ** 0.9


# ----------------------------- 模型来源 -----------------------------
# 训练机常常访问不了 huggingface.co / hf-mirror.com。这里做三件事：
#   1) --model 直接是本地目录 -> 直接用
#   2) 数据集根目录下的 pretrained/ 里有同名目录 -> 直接用
#   3) 都没有 -> 依次尝试各镜像（huggingface.co -> hf-mirror.com），
#      全挂则给出"怎么离线拷贝"的明确指引，而不是抛一堆看不懂的堆栈。
MODEL_ENDPOINTS = ["https://huggingface.co", "https://hf-mirror.com"]


def _is_local_model_dir(p: Path) -> bool:
    try:
        return p.is_dir() and (p / 'config.json').exists()
    except OSError:
        return False


def _find_local_model(root):
    """在 pretrained/ 下找可用的本地权重目录（递归两层，兼容 pretrained/segformer-b0-ade）"""
    bases = [Path(root) / 'pretrained',
             Path(__file__).resolve().parent / 'pretrained']
    for base in bases:
        if not base.is_dir():
            continue
        # pretrained/ 本身就是模型目录
        if _is_local_model_dir(base):
            return base
        # pretrained/<xxx>/ 是模型目录
        for d in sorted(base.iterdir()):
            if d.is_dir() and _is_local_model_dir(d):
                return d
            # 再深一层，兼容 pretrained/hub/xxx 这类结构
            if d.is_dir():
                for e in sorted(d.iterdir()):
                    if e.is_dir() and _is_local_model_dir(e):
                        return e
    return None


def resolve_model(model_arg, root):
    """本地优先。注意：--model 的默认值就是 hub id，若不做区分会直接跳过本地目录
    去联网（历史 bug）。所以这里先无条件找本地，找不到再考虑联网。"""
    # 1) 显式指定了本地路径
    if model_arg and model_arg.strip() not in ('', MODEL_NAME):
        for cand in (Path(model_arg).expanduser(), Path(root) / model_arg):
            if _is_local_model_dir(cand):
                print('  (使用 --model 指定的本地目录 %s)' % cand.resolve())
                return str(cand.resolve())

    # 2) 找 pretrained/ 下的本地权重（无论 --model 是否传了 hub id）
    local = _find_local_model(root)
    if local is not None:
        print('  (自动使用本地权重目录 %s)' % local)
        return str(local)

    # 3) 都没有 -> 联网；用本地时会自动开启离线模式
    return _try_endpoints(model_arg or MODEL_NAME)


def _try_endpoints(model_id):
    """依次尝试各镜像能否连通，返回第一个能用的 endpoint 下的 model_id"""
    import socket
    import urllib.parse

    ok = []
    for ep in MODEL_ENDPOINTS:
        host = urllib.parse.urlparse(ep).netloc
        try:
            socket.setdefaulttimeout(5)
            socket.create_connection((host, 443), timeout=5).close()
            ok.append(ep)
            print('  镜像 %s 连通' % ep)
        except Exception as e:
            print('  镜像 %s 不可达 (%s)' % (ep, type(e).__name__))

    if not ok:
        print('\n' + '=' * 66)
        print('[网络不通] 所有预训练权重镜像都无法连接。')
        print('请在【能上网的机器】上执行：')
        print('    python3 export_model.py pretrained/segformer-b0-ade')
        print('然后把整个 pretrained/segformer-b0-ade 目录拷到训练机，例如：')
        print('    rsync -avP pretrained/segformer-b0-ade/ \\')
        print('        dxy@服务器:~/projects/ycb/dataset_seg/pretrained/segformer-b0-ade/')
        print('再训练：python3 train_v2.py --model pretrained/segformer-b0-ade ...')
        print('=' * 66)
        raise SystemExit(1)

    os.environ['HF_ENDPOINT'] = ok[0]
    return model_id


# ----------------------------- 划分 -----------------------------
def stratified_split(stems, val_ratio=0.2, seed=42):
    """按类别分层划分，纯 Python，不依赖 sklearn"""
    rng = random.Random(seed)
    by = defaultdict(list)
    for s in stems:
        by[s.rsplit('_', 1)[0]].append(s)      # wood_block_00178 -> wood_block
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=None,
                    help='数据集根目录；不填则自动查找（当前目录/脚本目录/~/dataset_seg）')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--bs', type=int, default=2)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--accum', type=int, default=1, help='梯度累积步数')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=None)
    ap.add_argument('--model', default=MODEL_NAME)
    ap.add_argument('--crop', action='store_true')
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='>0 时只用前 N 个样本，用于冒烟测试')
    ap.add_argument('--log-every', type=int, default=20)
    args = ap.parse_args()

    from find_root import resolve_root
    root, why = resolve_root(args.root)
    print('数据集根目录: %s   (%s)' % (root, why))
    out_dir = Path(args.out) if args.out else root / 'ckpt_v2'
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    amp = bool(args.fp16 and use_cuda)

    print('环境: python=%s torch=%s cuda=%s'
          % (os.sys.version.split()[0], torch.__version__, use_cuda))
    if args.fp16 and not use_cuda:
        print('[警告] 没有可用 CUDA，--fp16 自动关闭')

    stems = sorted(p.stem for p in (root / 'labels').glob('*.png'))
    if not stems:
        print('[错误] %s 下 labels/ 是空的。' % (root / 'labels'))
        print('       先跑: python3 make_labels.py %s' % root)
        raise SystemExit(1)
    if args.limit > 0:
        stems = stems[:args.limit]

    size = {'height': 512, 'width': 512} if args.crop else {'height': IMG_H, 'width': IMG_W}

    # ---- 模型来源：优先本地目录，其次按镜像顺序联网下载 ----
    model_src = resolve_model(args.model, root)
    print('模型来源: %s' % model_src)
    # 本地目录：强制离线，避免 transformers 在加载前先去 hub 打一次 HEAD 请求
    if Path(model_src).is_dir():
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        print('  (已开启离线模式，不会访问网络)')

    processor = SegformerImageProcessor.from_pretrained(
        model_src, do_reduce_labels=False, size=size, do_resize=True)

    train_stems, val_stems = stratified_split(stems, 0.2, args.seed)
    print('总样本=%d 训练=%d 验证=%d 尺寸=%s 裁剪=%s bs=%d accum=%d fp16=%s'
          % (len(stems), len(train_stems), len(val_stems), size, args.crop,
             args.bs, args.accum, amp))
    print('类别表:', ID2LABEL)

    train_ds = SegDataset(train_stems, root, processor, train=True, crop=args.crop)
    val_ds = SegDataset(val_stems, root, processor, train=False, crop=args.crop)

    kw = dict(batch_size=args.bs, num_workers=args.workers,
              pin_memory=use_cuda and args.workers == 0)
    if args.workers > 0:
        kw['persistent_workers'] = True
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_src, num_labels=NUM_CLASSES, id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()},
        ignore_mismatched_sizes=True)      # 把预训练的 150 类解码头换成 6 类
    model.to(device)

    loss_fn = CEDiceLoss(NUM_CLASSES)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim == 1 or n.endswith('.bias')) else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{'params': decay, 'weight_decay': 0.01},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=args.lr, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    steps_per_epoch = max(1, len(train_loader) // args.accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * 0.1))
    print('每 epoch 步数=%d 总步数=%d warmup=%d' % (steps_per_epoch, total_steps, warmup_steps))

    log_path = out_dir / 'train_log.csv'
    fcsv = open(log_path, 'w', newline='')
    writer = csv.writer(fcsv)
    writer.writerow(['epoch', 'lr', 'train_loss', 'val_loss', 'miou_all', 'miou_fg']
                    + ['iou_' + ID2LABEL[c] for c in range(1, NUM_CLASSES)])

    best_fg, best_ep, global_step = -1.0, -1, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for it, batch in enumerate(train_loader):
            labels = batch.pop('labels').to(device, non_blocking=True)
            pv = batch['pixel_values'].to(device, non_blocking=True)

            lr_now = lr_at(global_step, total_steps, args.lr, warmup_steps)
            for g in optimizer.param_groups:
                g['lr'] = lr_now

            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(pixel_values=pv).logits
                loss = loss_fn(logits, labels) / args.accum
            scaler.scale(loss).backward()

            run_loss += float(loss) * args.accum * pv.size(0)
            seen += pv.size(0)

            if (it + 1) % args.accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every == 0:
                    print('  ep%d step %d/%d lr=%.2e loss=%.4f'
                          % (ep, global_step, total_steps, lr_now, run_loss / max(seen, 1)))

        train_loss = run_loss / max(seen, 1)
        m = evaluate(model, val_loader, device, loss_fn, amp=amp)
        dt = time.time() - t0

        line = ('ep%03d lr=%.2e | train %.4f val %.4f | miou_all %.4f 【miou_fg %.4f】 | %s | %.0fs'
                % (ep, lr_now, train_loss, m['loss'], m['miou_all'], m['miou_fg'],
                   ' '.join('%s %.3f' % (ID2LABEL[c], m['iou_' + ID2LABEL[c]])
                            for c in range(1, NUM_CLASSES)), dt))
        print(line, flush=True)
        writer.writerow([ep, lr_now, train_loss, m['loss'], m['miou_all'], m['miou_fg']]
                        + [m['iou_' + ID2LABEL[c]] for c in range(1, NUM_CLASSES)])
        fcsv.flush()

        if m['miou_fg'] > best_fg:
            best_fg, best_ep = m['miou_fg'], ep
            model.save_pretrained(str(out_dir / 'best'))
            processor.save_pretrained(str(out_dir / 'best'))
            print('  -> 新高，已保存 best (miou_fg=%.4f)' % best_fg)

    print('\n训练结束。最优 miou_fg=%.4f (ep%d)，模型在 %s' % (best_fg, best_ep, out_dir / 'best'))
    fcsv.close()


if __name__ == '__main__':
    main()
