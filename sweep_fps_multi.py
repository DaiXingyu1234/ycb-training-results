import time, os, glob, cv2, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from ultralytics import YOLO

IMG_DIR = '/home/xqq/dxy/images'
MODEL   = '/home/xqq/dxy/best.pt'
SIZES   = list(range(320, 832, 32))

# ---------- 读图 ----------
files = sorted(sum([glob.glob(os.path.join(IMG_DIR, e))
                    for e in ('*.jpg','*.jpeg','*.png','*.bmp')], []))
imgs = [cv2.imread(f) for f in files]
imgs = [im for im in imgs if im is not None]
print(f'共加载 {len(imgs)} 张测试图', flush=True)
if not imgs:
    raise SystemExit('没读到图片，检查目录')

# ---------- 每张图重复遍数（图多就少重复，控制总时长） ----------
REPEAT = 5 if len(imgs) <= 60 else (3 if len(imgs) <= 150 else 2)

model = YOLO(MODEL)
for _ in range(20):                       # 全局预热
    model.predict(imgs[0], imgsz=640, device=0, verbose=False)

rows = []
for s in SIZES:
    for _ in range(10):                   # 每档分辨率单独预热
        model.predict(imgs[0], imgsz=s, device=0, verbose=False)
    t0, n = time.time(), 0
    for _ in range(REPEAT):
        for im in imgs:
            model.predict(im, imgsz=s, device=0, verbose=False)
            n += 1
    ms = (time.time() - t0) / n * 1000
    rows.append(dict(imgsz=s, ms=round(ms, 2), fps=round(1000/ms, 1), n=n))
    print(rows[-1], flush=True)

f = pd.DataFrame(rows)
f.to_csv('/home/xqq/dxy/fps_sweep_multi.csv', index=False)
print('\n已保存 fps_sweep_multi.csv')

# ---------- 云端 A10 实测 mAP（测试集 50 张） ----------
m = pd.DataFrame({'imgsz':[320,416,512,640,800],
                  'map50':[0.859,0.9753,0.9836,0.9827,0.9803],
                  'map5095':[0.5828,0.6686,0.7826,0.7871,0.8091]})

# ---------- 中文字体（没有就自动用英文） ----------
zh = False
for p in ['/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
          '/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc',
          '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc']:
    if os.path.exists(p):
        font_manager.fontManager.addfont(p); zh = True; break
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei','DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
T = (lambda a,b: a) if zh else (lambda a,b: b)

fig, ax1 = plt.subplots(figsize=(9.5, 5.6))
ax1.plot(m.imgsz, m.map50,   'o-', color='tab:blue',  lw=2.4, ms=8,
         label=T('成功率 mAP50（A10）','Detection rate mAP50 (A10)'))
ax1.plot(m.imgsz, m.map5095, 's--', color='tab:green', lw=2, ms=7,
         label=T('轮廓精度 mAP50-95（A10）','Mask mAP50-95 (A10)'))
ax1.set_xlabel(T('输入分辨率 imgsz','Input resolution imgsz'), fontsize=12)
ax1.set_ylabel('mAP', fontsize=12, color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue'); ax1.set_ylim(0.5,1.0); ax1.grid(alpha=.3)

ax2 = ax1.twinx()
ax2.plot(f.imgsz, f.fps, '^-', color='tab:red', lw=2.4, ms=8,
         label=T('推理频率 FPS（Jetson 多图平均）','FPS (Jetson, multi-image avg)'))
ax2.set_ylabel('FPS', fontsize=12, color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red'); ax2.set_ylim(0, max(f.fps)*1.35)
ax2.axhline(10, ls=':', color='gray', lw=2, label='10 Hz 门槛')
ax2.text(SIZES[-1]-60, 12.5, '10 Hz', color='gray', fontsize=10)

for s, c, t in [(512,'#2ca02c',T('性价比 512','best-value 512')),
                (800,'#9467bd',T('高精度 800','high-accuracy 800'))]:
    ax1.axvline(s, ls=':', color=c, lw=2)
    ax1.text(s+6, 0.90, t, color=c, fontsize=9.5, rotation=90)

ax1.set_title(T(f'分辨率—成功率—频率权衡（{len(imgs)} 张图平均）',
                f'Resolution vs Accuracy vs FPS ({len(imgs)} images avg)'), fontsize=13)
h1,l1 = ax1.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='center right', fontsize=9)
plt.tight_layout()
plt.savefig('/home/xqq/dxy/分辨率_成功率_频率权衡曲线_多图.png', dpi=150)
print('已保存 分辨率_成功率_频率权衡曲线_多图.png')