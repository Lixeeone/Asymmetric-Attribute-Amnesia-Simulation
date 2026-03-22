import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

class AttentionalGlueSimulation:
    """
    Core implementation of the 'Attentional Glue' lesion paradigm.
    Simulates the collapse of representational manifolds under attentional load.
    """
    def __init__(self, dim=512):
        self.dim = dim
        # 模拟注意力模板 (W_Q, W_K)
        self.W_Q = torch.randn(dim, dim) * 0.02
        self.W_K = torch.randn(dim, dim) * 0.02

    def compute_binding_energy(self, Hv, Hs, alpha=0.0):
        """
        计算绑定能量矩阵并施加数字损伤 alpha。
        Hv: 视觉特征向量, Hs: 语义特征向量
        alpha: 损伤强度 [0, 1]
        """
        # 计算 Attention Energy (Biased Competition 逻辑)
        Q = torch.matmul(Hv, self.W_Q)
        K = torch.matmul(Hs, self.W_K)
        energy = torch.matmul(Q, K.t()) / np.sqrt(self.dim)
        
        # 基础 Softmax 归一化 (资源有限分配)
        A = F.softmax(energy, dim=-1)

        # 核心：施加数字损伤 (Alpha Lesion)
        # 公式: A_lesion = (1 - alpha) * A + alpha * Noise
        if alpha > 0:
            noise = torch.randn_like(A) * 0.1
            A_lesion = (1 - alpha) * A + alpha * noise
            return A_lesion
        return A

    @staticmethod
    def calculate_deff(features):
        """
        计算流形有效维数 (Participation Ratio).
        Deff = (sum lambda_i)^2 / sum (lambda_i^2)
        """
        cov = np.cov(features.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-10) # 避免数值不稳定
        
        deff = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2)
        return float(deff)

    @staticmethod
    def compute_rsa(model_features, target_rdm):
        """
        执行表征相似性分析 (RSA).
        计算模型 RDM 与目标 RDM 之间的 Spearman 相关系数。
        """
        # 计算模型 RDM (Correlation 距离)
        model_rdm = pdist(model_features, metric='correlation')
        
        # 计算 Spearman 相关性 (仅取矩阵的上三角向量)
        target_v = squareform(target_rdm, checks=False)
        corr, _ = spearmanr(model_rdm, target_v)
        return corr

# --- 模拟运行脚本 ---
if __name__ == "__main__":
    print("Initializing In Silico Lesion Simulation...")
    sim = AttentionalGlueSimulation(dim=512)
    
    # 模拟生成 50 个样本的特征 (Race, Gender, Conjunction)
    # 这里生成随机数据仅作演示，实际应使用 CLIP/DINO 提取的特征
    sample_features = torch.randn(50, 512)
    
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = []

    for a in alphas:
        # 1. 模拟特征绑定过程
        lesioned_features = sim.compute_binding_energy(sample_features, sample_features, alpha=a)
        
        # 2. 计算几何指标 (Manifold Dimensionality)
        current_deff = sim.calculate_deff(lesioned_features.detach().numpy())
        
        print(f"Lesion Alpha: {a:.1f} | Effective Dimension (Deff): {current_deff:.2f}")
        results.append(current_deff)

    print("\nSimulation Complete. Ready for Manifold Analysis.")
