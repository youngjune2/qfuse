import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# 스타일 적용 (ratio_qps.py 와 동일)
plt.style.use(['science', 'ieee'])

# ============================================================
# 유형별 비교 (Table 2) — 질문유형 4개 x 시스템 5개, 정확도(%)
#   분모: Bridge 90, Structured 30, DOC 30, Trap 21
# ============================================================
types = [r"Bridge", r"Structured", r"DOC", r"Trap"]
denom = np.array([90, 30, 30, 21], dtype=float)

# 각 시스템의 유형별 raw score (Bridge, Structured, DOC, Trap)
raw = {
    "HippoRAG":  np.array([45, 30, 29, 14]),
    "LightRAG":  np.array([12, 29, 28, 15]),
    "HybridRAG": np.array([ 6, 30, 16, 20]),
    "RouteRAG":  np.array([ 6, 30,  9, 20]),
    "QFuse (Ours)": np.array([61, 30, 22, 21]),
}
# v1 강조형: baseline 은 회색 명도로, QFuse(Ours) 만 진주황 + 굵은 테두리
colors = {
    "HippoRAG":    (0.80, 0.80, 0.82),
    "LightRAG":    (0.66, 0.66, 0.70),
    "HybridRAG":   (0.52, 0.52, 0.57),
    "RouteRAG":    (0.38, 0.38, 0.43),
    "QFuse (Ours)": (255/255, 150/255, 40/255),
}
linewidths = {
    "HippoRAG": 0.6, "LightRAG": 0.6, "HybridRAG": 0.6,
    "RouteRAG": 0.6, "QFuse (Ours)": 1.6,   # Ours 강조
}

systems = list(raw.keys())
n = len(systems)
x = np.arange(len(types))
bar_width = 0.8 / n

fig, ax = plt.subplots(figsize=(4, 3))
for i, s in enumerate(systems):
    acc = raw[s] / denom * 100.0
    off = (i - (n - 1) / 2) * bar_width
    ax.bar(x + off, acc, width=bar_width, color=colors[s],
           edgecolor='black', linewidth=linewidths[s], label=s, zorder=3)

ax.set_ylabel('Accuracy (\%)', fontsize=22)
ax.tick_params(axis='y', labelsize=13)
ax.tick_params(axis='x', labelsize=11, length=0)
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 20))
ax.set_xticks(x)
ax.set_xticklabels(types, fontsize=12, ha='center')
ax.set_xlim(-0.5, len(types) - 0.5)
ax.minorticks_off()
ax.grid(axis='y', linestyle='-', linewidth=0.5, alpha=0.3, color='gray', zorder=0)

# 범례 — 그림 위쪽 바깥에 3열
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3,
          fontsize=9, frameon=False, columnspacing=1.0, handlelength=1.2)

plt.tight_layout()
plt.savefig("figs/fig_types.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figs/fig_types.png", dpi=300, bbox_inches='tight')
print("Saved: figs/fig_types.pdf, figs/fig_types.png")
