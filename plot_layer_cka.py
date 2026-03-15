import os
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

# ==========================================
# 1. 核心数学模块：高效 Feature-space Linear CKA
# ==========================================
def feature_space_linear_CKA(X, Y):
    """
    计算两个高维表征 X 和 Y 之间的线性 CKA 相似度。
    采用 Feature-space 公式等效替代 Gram-space，规避 N 过大导致的 OOM。
    X, Y 形状为 (N, D)，N 为总帧数 (可能达到数万)，D 为特征维度 (1024)
    """
    # 1. 沿样本/帧维度去中心化
    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)
    
    # 2. 计算特征协方差/交叉协方差矩阵 (D x D)，而非 Gram 矩阵 (N x N)
    # 利用弗罗贝尼乌斯内积性质: ||X^T Y||_F^2 = trace(X X^T Y Y^T)
    dot_prod = X_centered.T @ Y_centered             # 形状: (D, D)
    cov_X = X_centered.T @ X_centered                # 形状: (D, D)
    cov_Y = Y_centered.T @ Y_centered                # 形状: (D, D)
    
    # 3. 计算 CKA
    numerator = np.linalg.norm(dot_prod, ord='fro') ** 2
    denominator = np.linalg.norm(cov_X, ord='fro') * np.linalg.norm(cov_Y, ord='fro')
    
    return numerator / denominator

# ==========================================
# 2. 帧级特征提取与 CKA 矩阵计算模块
# ==========================================
def extract_unpooled_features_and_compute_cka(dataset_path, num_samples=150, batch_size=8):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading WavLM model on {device}...")
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-large')
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
    wavlm.eval()
    
    print(f"Loading dataset from {dataset_path}...")
    try:
        with open(dataset_path, "rb") as f:
            dataset_dict = pickle.load(f)
            dataset = dataset_dict.get('test', dataset_dict.get('train'))
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None
    
    np.random.seed(42)
    sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    audios = []
    for idx in sample_indices:
        audios.append(dataset[int(idx)]['audio']['array'])
        
    print(f"Processing {len(audios)} audio samples in batches of {batch_size}...")
    
    # 初始化一个存储 25 层特征的列表，每个元素也是一个列表，用于存放该层下所有样本的帧序列
    all_layer_frames = [[] for _ in range(25)]
    downsample_rate = 320 

    # 引入批量循环，防止 GPU OOM
    for i in tqdm(range(0, len(audios), batch_size), desc="Extracting features"):
        batch_audios = audios[i : i + batch_size]
        
        inputs = feature_extractor(batch_audios, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs['input_values'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = wavlm(input_values=input_values, attention_mask=attention_mask, output_hidden_states=True)
            hiddens = outputs.hidden_states 
        
        actual_wav_length = attention_mask.sum(dim=1).cpu().numpy()
        feature_lens = [round(wav_length / downsample_rate) for wav_length in actual_wav_length]
        
        # 逐层、逐样本提取有效帧并移至 CPU 内存
        for layer_idx in range(len(hiddens)):
            layer_tensor = hiddens[layer_idx]
            for b in range(len(feature_lens)):
                valid_len = max(1, feature_lens[b])
                valid_feat = layer_tensor[b, :valid_len, :].cpu().numpy()
                all_layer_frames[layer_idx].append(valid_feat)
        
        # 显式清理 GPU 缓存
        del inputs, input_values, attention_mask, outputs, hiddens
        torch.cuda.empty_cache()
    
    print("Concatenating valid frames across all samples...")
    layer_features = []
    for layer_idx in range(25):
        # 将该层所有样本的帧序列纵向拼接为 (Total_Frames, 1024)
        layer_unpooled_matrix = np.concatenate(all_layer_frames[layer_idx], axis=0)
        layer_features.append(layer_unpooled_matrix)
        
    total_frames = layer_features[0].shape[0]
    print(f"Total valid frames extracted: {total_frames} frames per layer.")
    print("Computing 25x25 Frame-level CKA Similarity Matrix...")
    
    num_layers = len(layer_features)
    cka_matrix = np.zeros((num_layers, num_layers))
    
    for i in tqdm(range(num_layers), desc="CKA Calculation"):
        for j in range(num_layers):
            if i <= j:
                cka_val = feature_space_linear_CKA(layer_features[i], layer_features[j])
                cka_matrix[i, j] = cka_val
                cka_matrix[j, i] = cka_val 
                
    return cka_matrix

# ==========================================
# 3. 高质量学术可视化模块
# ==========================================
def plot_cka_heatmap(cka_matrix, output_dir):
    df = pd.DataFrame(cka_matrix)
    df.index = [f"L{i}" for i in range(cka_matrix.shape[0])]
    df.columns = [f"L{i}" for i in range(cka_matrix.shape[1])]
    
    df = df.iloc[::-1]
    
    plt.figure(figsize=(10, 8), dpi=300)
    sns.set_theme(style="white")
    
    # 采用学术界常用于 CKA 的 "viridis" 或 "mako" 配色
    ax = sns.heatmap(df, 
                     cmap="viridis", 
                     annot=False, 
                     square=True,
                     linewidths=0.5, 
                     cbar_kws={'label': 'Frame-level Linear CKA Similarity'})
    
    plt.title('Frame-level Representational Similarity (CKA) Across WavLM Layers', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Transformer Layer Depth', fontsize=14, fontweight='bold')
    plt.ylabel('Transformer Layer Depth', fontsize=14, fontweight='bold')
    
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10, rotation=0)
    
    plt.tight_layout()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    pdf_path = os.path.join(output_dir, "wavlm_unpooled_cka_matrix.pdf")
    png_path = os.path.join(output_dir, "wavlm_unpooled_cka_matrix.png")
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight')
    
    print(f"Frame-level CKA heatmaps successfully saved to {pdf_path} and {png_path}")

if __name__ == "__main__":
    DATASET_PATH = "./data/audio_partial4_train_dataset.pickle"
    OUTPUT_FOLDER = "./plots"
        
    cka_mat = extract_unpooled_features_and_compute_cka(DATASET_PATH, num_samples=150)
    
    if cka_mat is not None:
        plot_cka_heatmap(cka_mat, OUTPUT_FOLDER)