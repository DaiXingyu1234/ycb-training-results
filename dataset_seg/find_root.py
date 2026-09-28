# -*- coding: utf-8 -*-
"""
自动定位 dataset_seg 根目录。

背景：早期版本写死 ~/dataset_seg，一旦数据放在别处（例如
~/projects/ycb/dataset_seg）就会静默创建一个空目录并报"labels/ 为空"。

现在的查找顺序：
    1) 命令行显式传入
    2) 当前工作目录（如果它本身就是 dataset_seg，或它的子目录里有）
    3) 脚本所在目录
    4) ~/dataset_seg（兜底）

判定"这就是数据集根目录"的依据：该目录下同时存在 images/ 和 masks/
（或 images/ 和 labels/）。
"""
from pathlib import Path


def _looks_like_root(p: Path) -> bool:
    try:
        if not p.is_dir():
            return False
        has_img = (p / 'images').is_dir()
        has_mask = (p / 'masks').is_dir()
        has_lbl = (p / 'labels').is_dir()
        return has_img and (has_mask or has_lbl)
    except OSError:
        return False


def resolve_root(explicit=None):
    """返回 (root, 来源说明字符串)"""
    if explicit:
        return Path(explicit).expanduser().resolve(), '命令行指定'

    here = Path.cwd().resolve()

    # 当前目录本身就是 dataset_seg
    if _looks_like_root(here):
        return here, '当前目录'

    # 当前目录的下一级里找（例如 ~/projects/ycb 下有 dataset_seg/）
    for sub in sorted(here.iterdir()) if here.is_dir() else []:
        if sub.is_dir() and _looks_like_root(sub):
            return sub, '当前目录的子目录 %s' % sub.name

    # 脚本所在目录
    script_dir = Path(__file__).resolve().parent
    if _looks_like_root(script_dir):
        return script_dir, '脚本所在目录'
    for sub in sorted(script_dir.iterdir()) if script_dir.is_dir() else []:
        if sub.is_dir() and _looks_like_root(sub):
            return sub, '脚本目录下的 %s' % sub.name

    # 兜底
    home_default = Path.home() / 'dataset_seg'
    return home_default, '兜底 ~/dataset_seg（未找到数据集，可能路径不对）'
