# -*- coding: utf-8 -*-
"""
SegFormer-B0 语义分割训练脚本（背景 + 5 个 YCB 物体 = 6 类）

依赖:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install "transformers==4.36.2" accelerate numpy pillow
国内服务器建议先执行:
    export HF_ENDPOINT=https://hf-mirror.com

用法:
    python3 train_segformer.py                          # 默认 ~/dataset_seg
    python3 train_segformer.py --root /home/xqq/dataset_seg --epochs 100 --bs 8
    python3 train_segformer.py --crop                   # 小目标建议加：按 mask 外接框裁剪后再训
"""
import argparse
import os
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from transformers import (SegformerImageProcessor,
                          SegformerForSemanticSegmentation,
                          TrainingArguments, Trainer)

# ------------------------- 配置区（按需修改这里就够了） -------------------------
DEFAULT_MAP = "apple=1,tuna_fish_can=2,plastic_bottle=3,power_drill=4,wood_block=5"
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"   # 小数据只用 B0
IMG_H, IMG_W = 544, 960        # 训练尺寸，必须都是 32 的倍数；544x960 ≈ 720x1280 的比例
BG_WEIGHT = 0.1                # 背景类在 loss 里的权重（前景太小时压低背景）
DICE_WEIGHT = 1.0              # Dice 项权重（小目标很关键）
# -----------------------------------------------------------------------------

ID2LABEL = {0: "background"}
for _item in DEFAULT_MAP.split(','):
    _k, _v = _item.split('=')
    ID2LABEL[int(_v)] = _k.strip()
NUM_CLASSES = len(ID2LABEL)          # 6


# ----------------------------- 数据集 -----------------------------
class SegDataset(Dataset):
    def __init__(self, stems, root, processor, train=False, crop=False):
        self.stems = stems
        self.root = Path(root)
        self.processor = processor
        self.train = train
        self.crop = crop
        self.img_paths = {}
        for p in (self.root / 'images').glob('*'):
            self.img_paths[p.stem] = p

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem = self.stems[i]
        img = Image.open(self.img_paths[stem]).convert('RGB')   # 防止 RGBA 四通道
        lbl = Image.open(self.root / 'labels' / (stem + '.png')).convert('L')

        if self.crop:
            # 小目标救星：按前景外接框扩边裁成方形，再交给 processor resize
            a = np.array(lbl)
            ys, xs = np.where(a > 0)
            if len(ys):
                y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
                m = 0.15
                h, w = a.shape
                dh, dw = int((y1 - y0) * m) + 8, int((x1 - x0) * m) + 8
                y0, y1 = max(0, y0 - dh), min(h, y1 + dh)
                x0, x1 = max(0, x0 - dw), min(w, x1 + dw)
                if random.random() < 0.5:                       # 随机再外扩，增加尺度扰动
                    ey, ex = (y1 - y0) // 4, (x1 - x0) // 4
                    y0, y1 = max(0, y0 - ey), min(h, y1 + ey)
                    x0, x1 = max(0, x0 - ex), min(w, x1 + ex)
                img = img.crop((x0, y0, x1, y1))
                lbl = lbl.crop((x0, y0, x1, y1))

        if self.train:
            # 图像与标签必须做完全相同的几何变换（标签用 NEAREST，禁止插值）
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
            if random.random() < 0.5:                            # 只作用于图像的光度扰动
                img = _jitter(img)

        out = self.processor(images=img, segmentation_maps=lbl, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in out.items()}
        item['labels'] = item['labels'].long()
        return item


def _jitter(img):
    """轻量光度增强，不需要装 albumentations"""
    import torchvision.transforms.functional as TF
    img = TF.adjust_brightness(img, 1 + random.uniform(-0.25, 0.25))
    img = TF.adjust_contrast(img, 1 + random.uniform(-0.25, 0.25))
    img = TF.adjust_saturation(img, 1 + random.uniform(-0.25, 0.25))
    return img


# ----------------------------- 损失 -----------------------------
class CEDiceLoss(nn.Module):
    """加权 CE + Dice：背景像素占绝大多数时，纯 CE 会让模型直接躺平预测全背景"""

    def __init__(self, num_classes, bg_weight=BG_WEIGHT, dice_weight=DICE_WEIGHT):
        super().__init__()
        w = torch.ones(num_classes)
        w[0] = bg_weight
        self.register_buffer('cls_w', w)
        self.dice_weight = dice_weight

    def forward(self, logits, labels):
        # HF 的 SegFormer 输出是输入的 1/4，必须先上采样到标签尺寸再算 loss
        logits = F.interpolate(logits, size=labels.shape[-2:],
                               mode='bilinear', align_corners=False)
        w = self.cls_w.to(logits.device)
        ce = F.cross_entropy(logits, labels, weight=w, ignore_index=255)

        p = torch.softmax(logits, dim=1)
        y = labels.clamp(min=0)
        oh = torch.zeros_like(p).scatter_(1, y.unsqueeze(1), 1.0)
        wv = w.view(1, -1, 1, 1)
        inter = (p * oh * wv).sum(dim=(0, 2, 3))
        union = ((p + oh) * wv).sum(dim=(0, 2, 3))
        dice = 1 - (2 * inter / union.clamp(min=1e-6)).mean()
        return ce + self.dice_weight * dice


# ----------------------------- 指标 -----------------------------
def compute_metrics(eval_pred):
    """自己算混淆矩阵，避免 evaluate / 版本 API 差异；重点看 foreground mIoU"""
    logits, labels = eval_pred
    logits = torch.from_numpy(np.asarray(logits)).float()
    labels = torch.from_numpy(np.asarray(labels)).long()
    logits = F.interpolate(logits, size=labels.shape[-2:],
                           mode='bilinear', align_corners=False)
    preds = logits.argmax(dim=1)

    k = (labels >= 0) & (labels < NUM_CLASSES)
    hist = torch.bincount((NUM_CLASSES * labels[k] + preds[k]).view(-1),
                          minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    inter = torch.diag(hist)
    union = hist.sum(0) + hist.sum(1) - inter
    iou = inter / union.clamp(min=1)

    res = {'miou_all': float(iou.mean()), 'miou_fg': float(iou[1:].mean())}
    for c in range(1, NUM_CLASSES):
        res['iou_' + ID2LABEL[c]] = float(iou[c])
    return res


# ----------------------------- Trainer -----------------------------
class SegTrainer(Trainer):
    def __init__(self, *a, loss_fn=None, **kw):
        super().__init__(*a, **kw)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop('labels', None)
        if labels is None:
            labels = kw.get('labels')
        out = model(**inputs)
        loss = self.loss_fn(out.logits, labels.long())
        return (loss, out) if return_outputs else loss


# ----------------------------- 划分 -----------------------------
def stratified_split(stems, val_ratio=0.2, seed=42):
    """按类别分层，不依赖 sklearn"""
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    for s in stems:
        by_cls[s.rsplit('_', 1)[0]].append(s)
    train, val = [], []
    for c in sorted(by_cls):
        items = by_cls[c][:]
        rng.shuffle(items)
        k = max(1, int(round(len(items) * val_ratio)))
        val += items[:k]
        train += items[k:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(Path.home() / 'dataset_seg'))
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=6e-5)
    ap.add_argument('--out', default=None)
    ap.add_argument('--model', default=MODEL_NAME)
    ap.add_argument('--crop', action='store_true', help='按 mask 外接框裁剪后再训练（小目标推荐）')
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root / 'ckpt_segformer_b0'
    print('环境: python=%s torch=%s cuda=%s transformers=%s'
          % (__import__('sys').version.split()[0], torch.__version__,
             torch.cuda.is_available(),
             __import__('transformers').__version__))
    if not torch.cuda.is_available():
        print('[警告] 没检测到 GPU，900 张 100 epoch 会非常慢，确认是否在正确的 conda 环境里')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    lbl_dir = root / 'labels'
    stems = sorted(p.stem for p in lbl_dir.glob('*.png'))
    assert stems, 'labels/ 是空的，请先跑 make_labels.py'

    # 训练集尺寸：裁剪模式用方形，否则保持接近 1280x720 的比例
    size = {'height': 512, 'width': 512} if args.crop else {'height': IMG_H, 'width': IMG_W}
    processor = SegformerImageProcessor.from_pretrained(
        args.model, do_reduce_labels=False, size=size, do_resize=True)

    train_stems, val_stems = stratified_split(stems, 0.2, args.seed)
    print('总样本=%d  训练=%d  验证=%d  输入尺寸=%s  裁剪=%s'
          % (len(stems), len(train_stems), len(val_stems), size, args.crop))
    print('类别表:', ID2LABEL)

    train_ds = SegDataset(train_stems, root, processor, train=True, crop=args.crop)
    val_ds = SegDataset(val_stems, root, processor, train=False, crop=args.crop)

    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model, num_labels=NUM_CLASSES, id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()},
        ignore_mismatched_sizes=True)          # 自动把 150 类解码头换成 6 类

    # transformers 版本差异：
    #   <=4.40 用 evaluation_strategy / evaluation_strategy 老名字
    #   >=4.41 改成 eval_strategy
    # 注意：TrainingArguments 是 dataclass，直接 setattr 一个不存在的字段名不会报错，
    # 只是默默失效 -> 验证集一次都不跑 -> load_best_model_at_end 直接崩。
    # 所以必须靠字段白名单判断，不能用 try/except。
    import inspect
    _fields = set(inspect.signature(TrainingArguments.__init__).parameters)
    eval_kw = 'eval_strategy' if 'eval_strategy' in _fields else 'evaluation_strategy'
    print('使用评估参数名:', eval_kw)

    base_kw = dict(
        output_dir=str(out_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        num_train_epochs=args.epochs,          # 靠最优模型，不会真跑满
        lr_scheduler_type='polynomial',
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=20,
        save_total_limit=3,
        seed=args.seed,
        dataloader_num_workers=args.workers,
        remove_unused_columns=False,
        report_to='none',
    )
    # 关键：用前景 mIoU 选模型，别被背景拉高的分数骗了
    best_kw = dict(load_best_model_at_end=True,
                   metric_for_best_model='eval_miou_fg',
                   greater_is_better=True)

    if args.fp16 and torch.cuda.is_available():
        base_kw['fp16'] = True
    elif args.fp16:
        print('[警告] --fp16 已指定但没有可用 CUDA，已自动关闭')

    targs = TrainingArguments(
        **base_kw,
        **{eval_kw: 'epoch', 'save_strategy': 'epoch'},
        **best_kw,
    )

    trainer = SegTrainer(model=model, args=targs, train_dataset=train_ds,
                         eval_dataset=val_ds, compute_metrics=compute_metrics,
                         loss_fn=CEDiceLoss(NUM_CLASSES))
    trainer.train()

    final = root / 'segformer_final'
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    print('训练完成，模型已保存到', final)


if __name__ == '__main__':
    main()

