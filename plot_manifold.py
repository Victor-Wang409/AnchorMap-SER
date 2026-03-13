import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.manifold import TSNE

def plot_feature_space_tsne(pickle_path, output_dir):
    print(f"Loading data from {pickle_path}...")
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    embeddings = data['embeddings']      # 高维特征 (样本数, emb_dim)
    emotions = data['emotion']           # 离散情感标签
    status = data['status']              # train, val, test 划分

    # 为了学术严谨性，流形可视化通常只展示 Test 集，证明模型的泛化能力
    # 如果您想展示所有数据，可以将这里的判断去掉
    test_idx = np.where(np.array(status) == 'test')[0]
    
    if len(test_idx) == 0:
        print("Warning: No 'test' samples found. Using all samples instead.")
        test_idx = np.arange(len(emotions))

    test_emb = embeddings[test_idx]
    test_emo = np.array(emotions)[test_idx]

    # IEMOCAP 情感映射字典 (请根据实际情况调整)
    emo_mapping = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad', 4: 'Excited'}
    emo_labels = [emo_mapping[e] for e in test_emo]

    print(f"Running t-SNE on {len(test_emb)} samples. This might take a few seconds...")
    # 运行 t-SNE 降维到 2D
    # perplexity 通常设置在 5 到 50 之间，取决于样本量
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    emb_2d = tsne.fit_transform(test_emb)

    # 构建 DataFrame 方便 Seaborn 绘图
    df = pd.DataFrame({
        't-SNE Dimension 1': emb_2d[:, 0],
        't-SNE Dimension 2': emb_2d[:, 1],
        'Emotion': emo_labels
    })

    # ==========================================
    # 开始高规格学术绘图
    # ==========================================
    plt.figure(figsize=(9, 7), dpi=300)
    sns.set_theme(style="ticks") # ticks 风格比纯白底色多了一些刻度线，更具学术感

    # 为不同情感指定符合直觉的颜色 (例如：愤怒-红色，悲伤-蓝色，快乐-黄色，中性-灰色)
    palette = {
        'Angry': '#d62728',   # 红色
        'Happy': '#ff7f0e',   # 橘黄色
        'Neutral': '#7f7f7f', # 灰色
        'Sad': '#1f77b4',      # 蓝色
        'Excited': '#1f77a3'      # 蓝色
    }

    # 绘制散点图
    # alpha=0.8 增加透明度防止重叠遮挡, edgecolor 增加描边让点更清晰
    ax = sns.scatterplot(
        data=df, 
        x='t-SNE Dimension 1', 
        y='t-SNE Dimension 2',
        hue='Emotion', 
        palette=palette, 
        s=60, 
        alpha=0.85, 
        edgecolor='w',
        linewidth=0.5
    )

    # 优化字体与排版
    plt.title('t-SNE Visualization of High-Dimensional Emotion Representations', 
              fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    plt.ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 优化图例 (Legend)
    plt.legend(title='Emotion Category', title_fontsize=13, fontsize=11, 
               loc='best', frameon=True, shadow=True)

    # 去除右侧和上侧的边框，这是顶刊常用的排版技巧
    sns.despine()
    plt.tight_layout()

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    pdf_path = os.path.join(output_dir, "tsne_manifold.pdf")
    png_path = os.path.join(output_dir, "tsne_manifold.png")
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight')
    
    print(f"Manifold plots successfully saved to {pdf_path} and {png_path}")
    plt.close()

if __name__ == "__main__":
    # 路径配置：请确保指向正确的 embeddings.pickle
    PICKLE_FILE = "./dump/tmp/embeddings.pickle"
    OUTPUT_FOLDER = "./plots"
    
    plot_feature_space_tsne(PICKLE_FILE, OUTPUT_FOLDER)