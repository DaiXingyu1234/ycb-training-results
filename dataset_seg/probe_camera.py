# -*- coding: utf-8 -*-
"""
摄像头体检：逐个试 /dev/video*，找出真正能出图的那个，并保存抓到的帧。

Jetson(Orin NX) 上常见坑：
  - /dev/video0 往往不是采集节点（是 metadata/legacy 节点），能 ls 到但打不开
  - CSI 摄像头(IMX708等) 必须走 gstreamer + nvarguscamerasrc，普通 V4L2 打不开
  - opencv-python 的 pip 版通常【不带 gstreamer 支持】，需要系统 apt 版的
    libopencv 或自行编译

用法:
    python3 probe_camera.py              # 试 0..9
    python3 probe_camera.py 0 1 2 3 4 5  # 只试指定的
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

OUT = Path.home() / 'camera_probe'
OUT.mkdir(exist_ok=True)


def report_backend():
    print('cv2 版本:', cv2.__version__)
    try:
        info = cv2.getBuildInformation()
    except Exception:
        info = ''
    has_gs = 'GStreamer' in info and 'YES' in info
    print('是否编译进 GStreamer 支持:', '是' if has_gs else '否')
    if not has_gs:
        print('  [注意] pip 安装的 opencv 通常不带 gstreamer。')
        print('        JetPack 自带的是 /usr/lib/python3/dist-packages/cv2 版本，')
        print('        或者用: pip3 install opencv-python 后再试 v4l2 索引。')
    return has_gs


def gst_pipe(idx, w=1280, h=720, fps=30):
    """CSI 摄像头走 nvarguscamerasrc；USB 摄像头走 v4l2src"""
    return ('nvarguscamerasrc sensor-id=%d ! '
            'video/x-raw(memory:NVMM),width=%d,height=%d,framerate=%d/1 ! '
            'nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! '
            'video/x-raw,format=BGR ! appsink' % (idx, w, h, fps))


def try_open(idx):
    """返回 (成功与否, 描述, 帧)"""
    # 1) 普通 V4L2
    for backend in (cv2.CAP_V4L2, cv2.CAP_ANY):
        try:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None and frame.size > 0:
                    return True, 'V4L2 index=%d backend=%s' % (idx, backend), frame
        except Exception:
            pass

    # 2) GStreamer (CSI)
    try:
        cap = cv2.VideoCapture(gst_pipe(idx), cv2.CAP_GSTREAMER)
        if cap.isOpened():
            # CSI 相机首帧慢，多试几次
            for _ in range(30):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    cap.release()
                    return True, 'GStreamer nvargus sensor-id=%d' % idx, frame
            cap.release()
    except Exception:
        pass
    return False, '', None


def main():
    has_gs = report_backend()
    args = sys.argv[1:]
    indices = [int(a) for a in args] if args else list(range(10))
    print('\n开始探测索引:', indices)
    print('-' * 60)

    found = []
    for idx in indices:
        ok, desc, frame = try_open(idx)
        if ok:
            p = OUT / ('cam_%d.jpg' % idx)
            cv2.imwrite(str(p), frame)
            mean = float(np.mean(frame))
            print('[可用] %s  分辨率=%s  平均亮度=%.1f  -> 已保存 %s'
                  % (desc, frame.shape[1::-1], mean, p))
            found.append((idx, desc, mean))
        else:
            print('[打不开] index=%d' % idx)

    print('-' * 60)
    if not found:
        print('没有找到可用摄像头。下一步建议：')
        print('  1) 看系统日志确认摄像头被识别:')
        print('       dmesg | tail -40 | grep -i -E "video|imx|camera|argus"')
        print('  2) CSI 摄像头自检(需装 v4l-utils):')
        print('       v4l2-ctl --list-devices')
        print('       v4l2-ctl -d /dev/video0 --all')
        print('  3) 确认 argus 守护进程在跑:')
        print('       systemctl status nvargus-daemon')
        print('  4) 先用拍照命令验证（JetPack 自带）:')
        print('       nvgstcapture-1.0 --sensor-id=0')
        return

    # 排除"能打开但是全黑"的坏帧
    good = [f for f in found if f[2] > 5.0]
    if good:
        best = good[0]
        print('\n推荐使用: --source %d   (%s)' % (best[0], best[1]))
    else:
        print('\n所有能打开的相机都是黑帧（平均亮度<5），检查镜头盖/曝光/权限。')
        print('先把抓到的图看一眼:', OUT)


if __name__ == '__main__':
    main()

