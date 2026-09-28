# -*- coding: utf-8 -*-
"""
转换前的体检脚本：确认每张掩码的"前景像素值"是否等于文件名所属类别的 id
跑完看到 全对=900 / 不一致=0 再执行 make_labels.py

用法:
    python3 check_masks.py
    python3 check_masks.py /home/xqq/dataset_seg
"""
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

DEFAULT_MAP = "apple=1,tuna_fish_can=2,plastic_bottle=3,power_drill=4,wood_block=5"
name2id = {}
for _i in DEFAULT_MAP.split(','):
    _k, _v = _i.split('=')
    name2id[_k.strip()] = int(_v)
id2name = {v: k for k, v in name2id.items()}

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'dataset_seg'
MASK_DIR = root / 'masks'

per_cls = defaultdict(Counter)      # 类名 -> {前景值: 张数}
mismatch = []
empty = []
multi = []
n = 0

for p in sorted(MASK_DIR.glob('*')):
    stem = p.stem
    cls = None
    base = stem.rsplit('_', 1)[0]
    if base in name2id:
        cls = base
    else:
        for k in name2id:
            if stem.startswith(k + '_'):
                cls = k
                break
    if cls is None:
        mismatch.append((p.name, '文件名无法解析类别'))
        continue

    a = np.array(Image.open(p))
    if a.ndim == 3:
        a = a[..., 0]
    nz = sorted(int(v) for v in np.unique(a) if v != 0)
    n += 1

    if len(nz) == 0:
        empty.append(p.name)
        per_cls[cls]['<空>'] += 1
        continue
    if len(nz) > 1:
        multi.append((p.name, nz))
    per_cls[cls][nz[0]] += 1
    if nz[0] != name2id[cls]:
        mismatch.append((p.name, '前景值=%d 但文件名类别=%s(id=%d)' % (nz[0], cls, name2id[cls])))

print('扫描掩码数:', n)
print('\n--- 每个类名的前景像素值分布 ---')
ok = 0
for cls in sorted(per_cls):
    want = name2id[cls]
    dist = per_cls[cls]
    good = dist.get(want, 0)
    ok += good
    flag = 'OK' if good == sum(dist.values()) else '!! 有问题'
    print('%-16s 期望值=%d  实际分布=%s  %s' % (cls, want, dict(dist), flag))

print('\n全对:', ok, '/', n)
print('空掩码:', len(empty), empty[:5])
print('多前景值:', len(multi), multi[:5])
print('不一致:', len(mismatch), mismatch[:5])
print('\n结论:', '可以直接跑 make_labels.py' if (ok == n and not mismatch) else '先处理上面的异常再转换')

