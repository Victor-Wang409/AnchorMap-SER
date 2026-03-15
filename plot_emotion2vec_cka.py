import os
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
import tempfile
from tqdm import tqdm
from funasr import AutoModel

# ==========================================
# 🌟 全局学术字体设置：Times New Roman
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

# ==========================================
# 1. 核心数学模块：高效 Feature-space Linear CKA
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
# 2. PyTorch Hook: 底层隐层特征劫持
# ==========================================
current_audio_features = {}

def get_forward_hook(layer_name):
    def hook(module, input, output):
        # 兼容 Fairseq 的 tuple 输出
        hs = output[0] if isinstance(output, tuple) else output
        hs_np = hs.detach().cpu().numpy()
        
        # 强制 batch=1，安全地压缩 batch 维度
        hs_np = np.squeeze(hs_np) 
        
        # 处理可能的异常一维标量
        if hs_np.ndim == 1:
            hs_np = hs_np.reshape(1, -1)
            
        current_audio_features[layer_name] = hs_np
    return hook

# ==========================================
# 3. 主干执行流程
# ==========================================
def extract_emotion2vec_cka(dataset_path, output_dir, num_samples=150):
    print("\n" + "="*50)
    print("🚀 Processing Model: emotion2vec_plus_large (via FunASR)")
    print("="*50)
    
    # 1. 加载官方 FunASR 模型 (自动从 ModelScope 下载，极其稳定)
    model_id = "iic/emotion2vec_plus_large"
    print("Loading model via FunASR / ModelScope...")
    funasr_model = AutoModel(model=model_id, disable_update=True, log_level="ERROR")
    pytorch_model = funasr_model.model
    pytorch_model.eval()
    
    # 2. 动态寻找所有的 Transformer Block 并挂载 Hook
    layer_names = []
    for name, module in pytorch_model.named_modules():
        cname = module.__class__.__name__.lower()
        if 'encoderlayer' in cname or 'transformerblock' in cname or 'altblock' in cname:
            if not isinstance(module, torch.nn.ModuleList):
                layer_names.append(name)
                
    import re
    def extract_layer_idx(name):
        nums = re.findall(r'\d+', name)
        return int(nums[-1]) if nums else 0
    
    layer_names = sorted(list(set(layer_names)), key=extract_layer_idx)
    print(f"✅ Successfully hijacked {len(layer_names)} Transformer layers! Attaching hooks...")
    
    hooks = []
    for name in layer_names:
        module = dict(pytorch_model.named_modules())[name]
        hooks.append(module.register_forward_hook(get_forward_hook(name)))
        
    # 3. 加载数据集
    try:
        with open(dataset_path, "rb") as f:
            dataset_dict = pickle.load(f)
            dataset = dataset_dict.get('test', dataset_dict.get('train'))
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
        
    np.random.seed(42)
    sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    audios = [dataset[int(idx)]['audio']['array'] for idx in sample_indices]
    
    # 4. 逐句前向推理 (安全使用临时音频文件作为输入接口)
    all_layer_frames = {name: [] for name in layer_names}
    
    print(f"Extracting hidden states sentence by sentence (Total: {len(audios)})...")
    for audio in tqdm(audios, desc="Forward Pass"):
        current_audio_features.clear()
        
        # 采用最稳妥的方式：保存为临时 wav 文件再传入 FunASR，100% 避开张量形状异常
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, 16000)
            tmp_path = tmp.name
            
        with torch.no_grad():
            _ = funasr_model.generate(input=tmp_path, granularity="frame", extract_embedding=True)
            
        os.remove(tmp_path)
            
        for name in layer_names:
            if name in current_audio_features:
                all_layer_frames[name].append(current_audio_features[name])
                
    for h in hooks:
        h.remove()
        
    # 5. 计算 CKA 矩阵
    print("Concatenating frames & Computing 24x24 CKA Matrix...")
    layer_features = [np.concatenate(all_layer_frames[name], axis=0) for name in layer_names]
    
    num_layers = len(layer_features)
    cka_matrix = np.zeros((num_layers, num_layers))
    for i in tqdm(range(num_layers), desc="Calculating CKA"):
        for j in range(num_layers):
            if i <= j:
                cka_val = feature_space_linear_CKA(layer_features[i], layer_features[j])
                cka_matrix[i, j] = cka_val
                cka_matrix[j, i] = cka_val 
                
    # 6. 生成高规格学术图表
    df = pd.DataFrame(cka_matrix, 
                      index=[f"Layer{i+1}" for i in range(num_layers)], 
                      columns=[f"Layer{i+1}" for i in range(num_layers)])
    df = df.iloc[::-1] # 反转Y轴让 L1 处于底部
    
    plt.figure(figsize=(10, 8), dpi=300)
    sns.set_theme(style="white")
    sns.heatmap(df, cmap="viridis", square=True, linewidths=0.5)
    
    # plt.title('Frame-level CKA Across Layers: emotion2vec_plus_large', fontsize=16, fontweight='bold', pad=20)
    # plt.xlabel('Transformer Layer Depth', fontsize=14, fontweight='bold')
    # plt.ylabel('Transformer Layer Depth', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10, rotation=0)
    plt.tight_layout()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    pdf_path = os.path.join(output_dir, "emotion2vec_plus_large_cka_matrix.pdf")
    png_path = os.path.join(output_dir, "emotion2vec_plus_large_cka_matrix.png")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight')
    print(f"\n🎉 大功告成！已完美保存 emotion2vec_plus_large 的 CKA 热力图。")

if __name__ == "__main__":
    DATASET_PATH = "./data/audio_partial4_train_dataset.pickle"
    if not os.path.exists(DATASET_PATH):
        DATASET_PATH = "./meta_data/audio_partial4_full_dataset.pickle"
    OUTPUT_FOLDER = "./plots/cka_comparisons"
    
    extract_emotion2vec_cka(DATASET_PATH, OUTPUT_FOLDER, num_samples=150)