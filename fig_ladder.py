import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# 스타일 적용 (ratio_qps.py 와 동일)
plt.style.use(['science', 'ieee'])

# 색상 정의 (RGB)
color_bridge = (170/255, 233/255, 255/255)   # Bridge
color_total  = (255/255, 179/255, 102/255)   # Total

# ============================================================
# Ablation 사다리 — full 모델(맨 우측)에서 장치를 하나씩 제거 (실측 ×3 결정론)
#   맨 우측 = QFuse(full), 왼쪽으로 -Pruning -> -Relation -> -Entity
#   각 라벨 = 오른쪽 이웃 대비 '추가로 제거한' 장치.
#   Bridge 분모 90, Total 분모 171
# ============================================================
labels = [
    r"$-$ Entity",
    r"$-$ Relation",
    r"$-$ Pruning",
    "QFuse",   # = full 모델
]

bridge_score = np.array([25, 31, 40, 61])    # / 90
total_score  = np.array([79, 98, 111, 134])  # / 171

bridge_acc = bridge_score / 90 * 100.0
total_acc  = total_score / 171 * 100.0

# Figure 생성
fig, ax = plt.subplots(figsize=(4, 3))

x = np.arange(len(labels))
bar_width = 0.35

bars1 = ax.bar(x - bar_width/2, bridge_acc, width=bar_width, color=color_bridge,
               edgecolor='black', label='Bridge', zorder=3)
bars2 = ax.bar(x + bar_width/2, total_acc, width=bar_width, color=color_total,
               edgecolor='black', label='Total', zorder=3)

# 막대 위 수치 라벨
for b, v in zip(bars1, bridge_acc):
    ax.text(b.get_x() + b.get_width()/2, v + 1.2, f"{v:.0f}", ha='center', va='bottom', fontsize=9)
for b, v in zip(bars2, total_acc):
    ax.text(b.get_x() + b.get_width()/2, v + 1.2, f"{v:.0f}", ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Accuracy (\%)', fontsize=22)
ax.tick_params(axis='y', labelsize=13)
ax.tick_params(axis='x', labelsize=8, length=0)
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 20))
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12, ha='center')
ax.set_xlim(-0.5, len(labels) - 0.5)
ax.minorticks_off()

# 격자선
ax.grid(axis='y', linestyle='-', linewidth=0.5, alpha=0.3, color='gray', zorder=0)

# 범례
ax.legend(loc='upper left', fontsize=13, frameon=False)

plt.tight_layout()
plt.savefig("figs/fig_ladder.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figs/fig_ladder.png", dpi=300, bbox_inches='tight')
print("Saved: figs/fig_ladder.pdf, figs/fig_ladder.png")
