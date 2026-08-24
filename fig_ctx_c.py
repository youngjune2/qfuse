import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'ieee'])

# Table 3 — 같은 KG, 검색기만 다른 3개 시스템
systems = ["HippoRAG", "LightRAG", "QFuse\n(Ours)"]
recall  = np.array([0.600, 0.144, 0.744])   # Context Recall (좌 패널)
size    = np.array([54.0, 34.0, 2.6])        # Context Size (우 패널)
colors  = [(170/255, 233/255, 255/255),
           (179/255, 224/255, 179/255),
           (255/255, 179/255, 102/255)]

x = np.arange(len(systems))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.7))

# 좌: Context Recall (높을수록 좋음)
b1 = ax1.bar(x, recall, width=0.6, color=colors, edgecolor='black', zorder=3)
for b, r in zip(b1, recall):
    ax1.text(b.get_x() + b.get_width()/2, r + 0.015, f"{r:.3f}",
             ha='center', va='bottom', fontsize=9)
ax1.set_ylabel('Context Recall $\\uparrow$', fontsize=15)
ax1.set_ylim(0, 1.0)
ax1.set_yticks(np.arange(0, 1.01, 0.2))
ax1.set_xticks(x); ax1.set_xticklabels(systems, fontsize=10)
ax1.set_xlim(-0.5, len(systems) - 0.5)
ax1.tick_params(axis='x', length=0); ax1.tick_params(axis='y', labelsize=10)
ax1.grid(axis='y', linestyle='-', linewidth=0.5, alpha=0.3, color='gray', zorder=0)

# 우: Context Size (낮을수록 좋음)
b2 = ax2.bar(x, size, width=0.6, color=colors, edgecolor='black', zorder=3)
for b, s in zip(b2, size):
    ax2.text(b.get_x() + b.get_width()/2, s + 0.8, f"{s:g}",
             ha='center', va='bottom', fontsize=9)
ax2.set_ylabel('Context Size $\\downarrow$', fontsize=15)
ax2.set_ylim(0, 60)
ax2.set_yticks(np.arange(0, 61, 20))
ax2.set_xticks(x); ax2.set_xticklabels(systems, fontsize=10)
ax2.set_xlim(-0.5, len(systems) - 0.5)
ax2.tick_params(axis='x', length=0); ax2.tick_params(axis='y', labelsize=10)
ax2.grid(axis='y', linestyle='-', linewidth=0.5, alpha=0.3, color='gray', zorder=0)

plt.tight_layout()
plt.savefig("figs/fig_ctx_c.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figs/fig_ctx_c.png", dpi=300, bbox_inches='tight')
print("Saved: figs/fig_ctx_c.{pdf,png}")
