# -*- coding: utf-8 -*-
"""
把 dataset_seg/masks/ 的掩码转成 SegFormer 要的单通道标签图 labels/

你的掩码已经是语义标签图：
    背景 = 0，前景像素值 = 类别 id
    apple=1, tuna_fish_can=2, plastic_bottle=3, power_drill=4, wood_block=5

所以本脚本【不做任何重新编号】，只做三件事：
    1) 校验每张掩码的唯一值必须 ⊆ {0,1,2,3,4,5}，且只有一个非零值
    2) 校验该非零值与文件名所属类的 id 一致（抓标注错误的图）
    3) 统一存成单通道 mode='L' 的 PNG（杜绝调色板 / 有损压缩）

用法:
    python3 make_labels.py                        # 默认 ~/dataset_seg
    python3 make_labels.py /home/xqq/dataset_seg
"""
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

DEFAULT_MAP = "apple=1,tuna_fish_can=2,plastic_bottle=3,power_drill=4,wood_block=5"


def parse_map(s):
    m = {}
    for item in s.split(','):
        k, v = item.split('=')
        m[k.strip()] = int(v)
    return m


def main():
    from find_root import resolve_root
    root, why = resolve_root(sys.argv[1] if len(sys.argv) > 1 else None)
    print('数据集根目录: %s   (%s)' % (root, why))
    if not (root / 'masks').is_dir():
        print('[错误] %s 下没有 masks/，请显式指定路径：python3 make_labels.py /你的/dataset_seg' % root)
        return
    name2id = parse_map(DEFAULT_MAP)
    id2name = {v: k for k, v in name2id.items()}

    print('类别映射（0 = background，与掩码像素值一致）:')
    for cid in sorted(id2name):
        print('   id=%d -> %s' % (cid, id2name[cid]))

    IMG_DIR = root / 'images'
    MASK_DIR = root / 'masks'
    LBL_DIR = root / 'labels'
    LBL_DIR.mkdir(exist_ok=True)

    def stem2cid(stem):
        """wood_block_00178 -> wood_block -> 5
        从右边只切一次去掉末尾编号，避免类名自带下划线被切坏。"""
        base = stem.rsplit('_', 1)[0]
        if base in name2id:
            return name2id[base]
        for n, cid in name2id.items():
            if stem.startswith(n + '_'):
                return cid
        raise KeyError('无法从文件名解析类别: %s' % stem)

    stats = defaultdict(lambda: [0, 0.0, 1.0])   # cid -> [张数, 占比和, 最小占比]
    n_empty = 0
    n_ok = 0
    n_bad = 0
    bad_list = []

    for mask_path in sorted(MASK_DIR.glob('*')):
        stem = mask_path.stem
        if not list(IMG_DIR.glob(stem + '.*')):
            print('skip (无对应原图):', mask_path.name)
            continue

        m = np.array(Image.open(mask_path))
        if m.ndim == 3:
            if (m[..., 0] == m[..., 1]).all() and (m[..., 1] == m[..., 2]).all():
                m = m[..., 0]                      # 三通道完全相同 -> 当灰度
            else:
                raise RuntimeError('%s 是彩色掩码，请先确认颜色映射' % mask_path)

        vals = sorted(int(v) for v in np.unique(m))
        nonzero = [v for v in vals if v != 0]

        # ---- 校验 1：前景值必须恰好一个，且在 1..5 ----
        if len(nonzero) > 1:
            print('!! 掩码含多个非零值 %s（一张图多个类别？）: %s' % (nonzero, mask_path.name))
            n_bad += 1
            bad_list.append(mask_path.name)
        if any(v not in id2name for v in nonzero):
            raise RuntimeError('%s 掩码含非法类别值 %s，合法为 1~5' % (mask_path, nonzero))

        # ---- 校验 2：掩码值与文件名类别是否一致 ----
        cid_file = stem2cid(stem)
        if nonzero and nonzero[0] != cid_file:
            print('!! 掩码值(%d) 与文件名类别 %s(%d) 不一致: %s'
                  % (nonzero[0], id2name[cid_file], cid_file, mask_path.name))
            n_bad += 1
            bad_list.append(mask_path.name)

        # ---- 核心：像素值原样保留，不重新编号 ----
        lbl = m.astype(np.uint8)
        cid = cid_file

        ratio = float((lbl > 0).mean())
        s = stats[cid]
        s[0] += 1
        s[1] += ratio
        s[2] = min(s[2], ratio)
        if ratio == 0.0:
            n_empty += 1
            print('!! 空掩码（全背景）:', mask_path.name)

        # mode='L' 单通道 PNG：绝不存 JPG（有损会把类别边界插成不存在的 id）
        Image.fromarray(lbl, mode='L').save(LBL_DIR / (stem + '.png'))
        n_ok += 1

    print('\n--- 每类统计 ---')
    for cid in sorted(stats):
        n, tot, mn = stats[cid]
        print('id=%d %-16s 张数=%d 平均前景占比=%.2f%% 最小=%.2f%%'
              % (cid, id2name[cid], n, 100 * tot / n, 100 * mn))
    print('空掩码数:', n_empty)
    print('异常掩码数:', n_bad, bad_list[:10])
    print('共生成 %d 张标签 -> %s' % (n_ok, LBL_DIR))

    if len(stats) != len(name2id):
        print('\n[错误] 只有 %d 个类被匹配到，期望 %d 个，请检查！' % (len(stats), len(name2id)))
    else:
        print('\n[OK] 5 个类全部正确匹配，可以直接开训。')


if __name__ == '__main__':
    main()
