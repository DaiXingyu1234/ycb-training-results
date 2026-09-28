# -*- coding: utf-8 -*-
"""
Jetson 掉电/关机诊断工具

作用：用【纯 torch 矩阵乘法】把 GPU 拉满（完全不碰你的数据集和训练脚本），
同时把内存、温度、电压、功耗追加写入日志文件。
机器若再次关机，重启后读日志最后几行即可知道死前发生了什么。

用法:
    python3 power_check.py 60        # 满载 60 秒（默认 60）
    python3 power_check.py 30 --cpu  # 同时压 CPU，模拟 dataloader 负载

判定:
    纯 GPU 压力就关机 -> 硬件/电源问题（与训练脚本无关）
    能撑过去        -> 回去调小训练参数再试
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

LOG = Path.home() / 'power_check.log'


def log(line):
    ts = time.strftime('%H:%M:%S')
    s = '[%s] %s' % (ts, line)
    print(s, flush=True)
    try:
        with open(LOG, 'a') as f:
            f.write(s + '\n')
    except Exception:
        pass


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception as e:
        return '(失败: %s)' % e


def read_temp():
    vals = []
    for p in sorted(Path('/sys/devices/virtual/thermal').glob('thermal_zone*/temp')):
        try:
            v = int(p.read_text().strip())
            if v > 0:
                vals.append('%s=%.1fC' % (p.parent.name, v / 1000.0))
        except Exception:
            pass
    return ' '.join(vals[:8])


def read_mem():
    out = sh('free -m | head -2 | tail -1')
    parts = out.split()
    if len(parts) >= 3:
        return 'mem used=%sMB/%sMB' % (parts[2], parts[1])
    return out


def read_tegra():
    """tegrastats 采样 1 秒，抽 VDD_IN / VDD_CPU_GPU_CV / 功耗"""
    out = sh('timeout 3 sudo tegrastats --interval 200 --count 2 2>/dev/null | tail -1')
    if not out or out.startswith('(失败'):
        return None
    keys = []
    for tok in out.split():
        if tok.startswith(('VDD_IN', 'VDD_CPU_GPU_CV', 'VDD_SOC', 'POM_5V')):
            keys.append(tok)
    return ' '.join(keys) if keys else out[:160]


def monitor(stop_ev, interval=1.0):
    """后台采样线程"""
    tegra_ok = [True]
    while not stop_ev.is_set():
        line = read_mem() + ' | ' + read_temp()
        if tegra_ok[0]:
            t = read_tegra()
            if t is None:
                tegra_ok[0] = False
                log('(tegrastats 不可用，仅记录内存与温度；sudo 无权限或 jetson-stats 未装)')
            else:
                line += ' | ' + t
        log(line)
        stop_ev.wait(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('seconds', nargs='?', type=int, default=60)
    ap.add_argument('--cpu', action='store_true', help='同时压 CPU，模拟 dataloader')
    ap.add_argument('--size', type=int, default=4096, help='矩阵边长，4096 约吃 1~2GB 显存')
    args = ap.parse_args()

    import torch
    log('=' * 70)
    log('诊断开始 | torch=%s cuda=%s' % (torch.__version__, torch.cuda.is_available()))
    log('设备: %s' % sh('cat /proc/device-tree/model 2>/dev/null | tr -d "\\0"'))
    log('nvpmodel: %s' % sh('sudo nvpmodel -q 2>/dev/null | head -3'))
    log('内核: %s' % sh('uname -r'))
    log('磁盘: %s' % sh('df -h / | tail -1'))
    log('swap: %s' % sh('swapon --show=NAME,SIZE --noheadings 2>/dev/null || echo "(无 swap)"'))

    if not torch.cuda.is_available():
        log('CUDA 不可用，无法做 GPU 压力测试')
        return

    stop_ev = threading.Event()
    th = threading.Thread(target=monitor, args=(stop_ev,), daemon=True)
    th.start()

    dev = torch.device('cuda')
    n = args.size
    a = torch.randn(n, n, device=dev, dtype=torch.float16)
    b = torch.randn(n, n, device=dev, dtype=torch.float16)
    log('已分配显存 %.2f GB，开始满载 %d 秒（fp16 matmul %dx%d）'
        % (torch.cuda.memory_allocated() / 1024 ** 3, args.seconds, n, n))

    stop_cpu = threading.Event()

    def cpu_burn():
        x = 0.0
        while not stop_cpu.is_set():
            x += sum(i * i for i in range(200000)) * 1e-12
        return x

    cpus = []
    if args.cpu:
        cnt = min(4, os.cpu_count() or 1)
        for _ in range(cnt):
            t = threading.Thread(target=cpu_burn, daemon=True)
            t.start()
            cpus.append(t)
        log('同时启用 %d 个 CPU 满载线程' % cnt)

    t0 = time.time()
    it = 0
    try:
        while time.time() - t0 < args.seconds:
            c = torch.mm(a, b)
            torch.cuda.synchronize()
            it += 1
            if it % 50 == 0:
                log('已迭代 %d 次，用时 %.0fs，显存 %.2fGB'
                    % (it, time.time() - t0, torch.cuda.memory_allocated() / 1024 ** 3))
    except Exception as e:
        log('!! GPU 压力测试中异常: %r' % e)
    finally:
        stop_cpu.set()
        stop_ev.set()

    dt = time.time() - t0
    log('压力测试完成: %d 次迭代 / %.1f 秒，未关机' % (it, dt))
    log('结论: 纯 GPU 满载能撑住 -> 电源大概率够，回去把训练参数调小（--bs 1 --workers 0，先不开 --fp16）')
    log('=' * 70)


if __name__ == '__main__':
    main()

