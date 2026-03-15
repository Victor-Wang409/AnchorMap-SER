import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. 全局学术字体与样式设置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

def plot_va_space(npz_path="av_results_3d.npz", output_dir="./plots"):
    if not os.path.exists(npz_path):
        print(f"❌ 找不到文件: {npz_path}")
        return

    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)
    
    # pred 的形状为 (N, 3)，第 0 列是 Valence，第 1 列是 Arousal
    pred = data['pred']
    labels = data['labels']
    
    pred_v = pred[:, 0]
    pred_a = pred[:, 1]
    
    # 映射字典（根据您的 IEMOCAP 实际类别数调整）
    # 如果是 4 分类通常为: 0: Ang, 1: Hap, 2: Neu, 3: Sad
    # 这里提供一个通用的 fallback 字典
    emotion_names = {
        0: "Angry",
        1: "Happy", 
        2: "Neutral",
        3: "Sad",
        4: "Disgust",
        5: "Excitement",
        6: "Fear",
        7: "Frustration",
        8: "Surprise"
    }
    
    # 将整数标签转换为字符串标签以显示在图例中
    string_labels = [emotion_names.get(int(lbl), f"Class {int(lbl)}") for lbl in labels]
    
    # ==========================================
    # 2. 绘制散点图
    # ==========================================
    plt.figure(figsize=(9, 7), dpi=300)
    sns.set_theme(style="ticks")
    
    # 使用 seaborn 的散点图，采用高级的调色板
    ax = sns.scatterplot(
        x=pred_v, 
        y=pred_a, 
        hue=string_labels, 
        palette="deep",    # 学术界常用的沉稳配色
        s=10,              # 点的大小
        alpha=0.7,         # 透明度，防止重叠
        edgecolor=None
    )
    
    # 添加十字辅助线 (以 0,0 为中心，划分 V-A 空间的四个象限)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1.2, alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1.2, alpha=0.5)
    
    # 设置标题和坐标轴
    plt.title('2D Valence-Arousal Emotion Manifold', fontweight='bold', pad=15)
    plt.xlabel('Valence (Predicted)', fontweight='bold')
    plt.ylabel('Arousal (Predicted)', fontweight='bold')
    
    # 将图例移到图表外部，防止遮挡数据点
    plt.legend(title='Emotion Classes', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    
    plt.tight_layout()
    
    # ==========================================
    # 3. 保存图表
    # ==========================================
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    pdf_path = os.path.join(output_dir, "va_scatter_plot.pdf")
    png_path = os.path.join(output_dir, "va_scatter_plot.png")
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight')
    
    print(f"✅ 散点图已成功保存至 {pdf_path} 和 {png_path}")

if __name__ == "__main__":
    plot_va_space()