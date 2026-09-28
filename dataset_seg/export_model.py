# -*- coding: utf-8 -*-
"""
把 SegFormer-B0 预训练权重导出成一个干净的本地目录。

用途：当训练机（服务器）访问不了 huggingface.co / hf-mirror.com 时，
在【能上网的那台机器】（比如 Orin，或你自己的笔记本）上跑这个脚本，
得到一个纯文件目录，再用 rsync / U 盘拷到训练机，训练时 --model 指向它即可。

用法（在有网的机器上）:
    python3 export_model.py                       # 默认输出到 ./pretrained/segformer-b0-ade
    python3 export_model.py /home/xqq/segformer_b0_local

完成后把整个目录拷到训练机，例如拷到 ~/projects/ycb/dataset_seg/pretrained/segformer-b0-ade
然后训练:
    python3 train_v2.py --model pretrained/segformer-b0-ade ...
"""
import os
import sys
from pathlib import Path

MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"

# 按优先级尝试的镜像；脚本会依次试，谁通就用谁
ENDPOINTS = [
    os.environ.get('HF_ENDPOINT', '').strip() or None,
    "https://huggingface.co",
    "https://hf-mirror.com",
]


def try_export(out_dir: Path):
    from transformers import (SegformerForSemanticSegmentation,
                              SegformerImageProcessor)

    out_dir.mkdir(parents=True, exist_ok=True)
    print('下载/加载处理器 ...')
    proc = SegformerImageProcessor.from_pretrained(MODEL_ID)
    print('下载/加载模型 ...')
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)

    proc.save_pretrained(str(out_dir))
    model.save_pretrained(str(out_dir))
    return True


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path.cwd() / 'pretrained' / 'segformer-b0-ade'
    out_dir = out_dir.expanduser().resolve()

    if (out_dir / 'config.json').exists() and \
       any(out_dir.glob('*.safetensors')) and \
       (out_dir / 'preprocessor_config.json').exists():
        print('目录已存在且完整，跳过下载: %s' % out_dir)
        for f in sorted(out_dir.iterdir()):
            print('   ', f.name, f.stat().st_size, 'bytes')
        return

    tried = []
    for ep in ENDPOINTS:
        if not ep:
            continue
        os.environ['HF_ENDPOINT'] = ep
        print('\n=== 尝试镜像: %s ===' % ep)
        try:
            try_export(out_dir)
            print('\n[成功] 已导出到 %s' % out_dir)
            for f in sorted(out_dir.iterdir()):
                print('   ', f.name, f.stat().st_size, 'bytes')
            return
        except Exception as e:
            tried.append((ep, type(e).__name__, str(e)[:120]))
            print('  失败: %s: %s' % (type(e).__name__, str(e)[:120]))

    print('\n[全部失败] 试过的镜像:')
    for ep, en, em in tried:
        print('  - %s -> %s: %s' % (ep, en, em))
    print('\n请检查网络/代理；或换一台能上网的机器跑本脚本，再把目录拷过来。')
    raise SystemExit(1)


if __name__ == '__main__':
    main()

