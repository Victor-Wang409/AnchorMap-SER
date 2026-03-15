import os
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel

# ==========================================
# 🌟 全局学术字体设置：Times New Roman
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

# ==========================================
# 1. 高效 Feature-space Linear CKA
# ==========================================
def feature_space_linear_CKA(X, Y):
    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)
    
    dot_prod = X_centered.T @ Y_centered             
    cov_X = X_centered.T @ X_centered                
    cov_Y = Y_centered.T @ Y_centered                
    
    numerator = np.linalg.norm(dot_prod, ord='fro') ** 2
    denominator = np.linalg.norm(cov_X, ord='fro') * np.linalg.norm(cov_Y, ord='fro')
    
    return numerator / denominator

# ==========================================
# 2. 通用多模型帧级特征提取与 CKA 计算
# ==========================================
def process_single_model_cka(model_id, dataset_path, output_dir, num_samples=150, batch_size=4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"🚀 Processing Model: {model_id}")
    print(f"{'='*50}")
    
    # 使用 Auto 接口，无缝兼容 wav2vec2, hubert, wavlm, emotion2vec 等
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    # 对于 emotion2vec 等带自定义权重的模型，trust_remote_code=True 是好习惯
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()
    
    try:
        with open(dataset_path, "rb") as f:
            dataset_dict = pickle.load(f)
            dataset = dataset_dict.get('test', dataset_dict.get('train'))
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    np.random.seed(42)
    sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    audios = [dataset[int(idx)]['audio']['array'] for idx in sample_indices]
        
    print(f"Extracting features using mini-batches of {batch_size}...")
    
    # 动态探测模型层数（不同模型层数可能不同，Base=13, Large=25）
    dummy_input = feature_extractor(audios[:1], sampling_rate=16000, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        dummy_out = model(dummy_input, output_hidden_states=True)
        num_layers = len(dummy_out.hidden_states)
        
    print(f"Detected {num_layers} hidden states (including embeddings) for {model_id}.")
    
    all_layer_frames = [[] for _ in range(num_layers)]

    for i in tqdm(range(0, len(audios), batch_size), desc="Forward Pass"):
        batch_audios = audios[i : i + batch_size]
        
        inputs = feature_extractor(batch_audios, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device) if 'attention_mask' in inputs else None
        
        with torch.no_grad():
            outputs = model(input_values, attention_mask=attention_mask, output_hidden_states=True)
            hiddens = outputs.hidden_states 
        
        # 动态计算有效帧数比例，规避不同模型下采样率不同的问题
        max_audio_len = input_values.shape[1]
        max_frame_len = hiddens[0].shape[1]
        
        if attention_mask is not None:
            actual_wav_lengths = attention_mask.sum(dim=1).cpu().numpy()
        else:
            actual_wav_lengths = np.array([max_audio_len] * len(batch_audios))
            
        # 比例法计算帧级长度：(当前语音采样点数 / 最大采样点数) * 最大帧数
        feature_lens = [max(1, int((wav_len / max_audio_len) * max_frame_len)) for wav_len in actual_wav_lengths]
        
        for layer_idx in range(num_layers):
            layer_tensor = hiddens[layer_idx]
            for b in range(len(feature_lens)):
                valid_feat = layer_tensor[b, :feature_lens[b], :].cpu().numpy()
                all_layer_frames[layer_idx].append(valid_feat)
        
        del inputs, input_values, attention_mask, outputs, hiddens
        torch.cuda.empty_cache()
    
    print("Concatenating frames & Computing CKA Matrix...")
    layer_features = [np.concatenate(all_layer_frames[idx], axis=0) for idx in range(num_layers)]
    
    cka_matrix = np.zeros((num_layers, num_layers))
    for i in tqdm(range(num_layers), desc="Calculating CKA"):
        for j in range(num_layers):
            if i <= j:
                cka_val = feature_space_linear_CKA(layer_features[i], layer_features[j])
                cka_matrix[i, j] = cka_val
                cka_matrix[j, i] = cka_val 
                
    # ================= 绘图 =================
    model_short_name = model_id.split('/')[-1]
    df = pd.DataFrame(cka_matrix, 
                      index=[f"Layer{i}" for i in range(num_layers)], 
                      columns=[f"Layer{i}" for i in range(num_layers)])
    df = df.iloc[::-1] # 翻转 Y 轴，让 L0 在最下方
    
    plt.figure(figsize=(10, 8), dpi=300)
    sns.set_theme(style="white")
    sns.heatmap(df, cmap="viridis", square=True, linewidths=0.5, cbar_kws={'label': 'CKA Similarity'})
    
    # plt.title(f'Frame-level CKA Across Layers: {model_short_name}', fontsize=16, fontweight='bold', pad=20)
    # plt.xlabel('Transformer Layer Depth', fontsize=14, fontweight='bold')
    # plt.ylabel('Transformer Layer Depth', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10, rotation=0)
    plt.tight_layout()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    pdf_path = os.path.join(output_dir, f"{model_short_name}_cka_matrix.pdf")
    png_path = os.path.join(output_dir, f"{model_short_name}_cka_matrix.png")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight')
    print(f"✅ Saved plots for {model_short_name} to {output_dir}")
    plt.close()

if __name__ == "__main__":
    DATASET_PATH = "./data/audio_partial4_train_dataset.pickle"
    OUTPUT_FOLDER = "./plots/cka_comparisons"
    
    # 填入您想对比的所有 HuggingFace 模型 ID
    ssl_models_to_test = [
        "microsoft/wavlm-large",
        "facebook/wav2vec2-large-robust", # 或者 facebook/wav2vec2-large-960h
        "facebook/hubert-large-ll60k"
    ]
    
    for model_id in ssl_models_to_test:
        process_single_model_cka(model_id, DATASET_PATH, OUTPUT_FOLDER, num_samples=150, batch_size=4)