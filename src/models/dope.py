import torch
import torch.nn as nn
import torch.nn.functional as F

class TruncatedMatrixEntropy(nn.Module):
    def __init__(self, truncation_r=32, epsilon=1e-8):
        """
        Oblicza Truncated Matrix Entropy zgodnie z Eq. 30 artykułu DoPE.
        
        Args:
            truncation_r (int): Liczba największych wartości osobliwych branych pod uwagę (r).
                                Artykuł sugeruje r=1, 8, 16, 32.
            epsilon (float): Mała wartość dla stabilności logarytmu.
        """
        super().__init__()
        self.r = truncation_r
        self.epsilon = epsilon

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensor wejściowy (Keys lub Queries) 
                              Kształt: [Batch, Heads, Seq_Len, Head_Dim]
        
        Returns:
            entropy (torch.Tensor): Efektywna ranga (Effective Rank) dla każdej głowicy.
                                    Kształt: [Batch, Heads]
        """
        # 1. Obliczenie wartości osobliwych (Singular Values)
        # Używamy svdvals na ostatnich dwóch wymiarach (Seq_Len, Head_Dim)
        # Singular values (S) macierzy X są pierwiastkami wartości własnych macierzy kowariancji X^T*X.
        # W artykule używają wartości własnych (lambda) macierzy Grama (Sigma).
        # Lambda = S^2.
        
        # float32 jest wymagane dla stabilności SVD
        x_float = x.float()
        
        # SVD obliczamy dla każdej głowicy w batchu
        # singular_values shape: [Batch, Heads, min(Seq_Len, Head_Dim)]
        try:
            S = torch.linalg.svdvals(x_float)
        except RuntimeError:
            # Fallback dla niestabilności numerycznej
            return torch.ones(x.shape[0], x.shape[1], device=x.device)

        # 2. Obcięcie do top-r wartości (Truncation)
        # Jeśli wymiar głowicy jest mniejszy niż r, bierzemy wszystkie
        current_r = min(self.r, S.shape[-1])
        S_top = S[..., :current_r]

        # 3. Konwersja na wartości własne (Eigenvalues) macierzy kowariancji
        # Lambda_i = S_i^2
        eigenvalues = S_top.pow(2)

        # 4. Normalizacja (Trace normalization) - Eq. 24
        # p_i = lambda_i / sum(lambda)
        sum_eigenvalues = eigenvalues.sum(dim=-1, keepdim=True)
        probs = eigenvalues / (sum_eigenvalues + self.epsilon)

        # 5. Obliczenie Entropii Shannona - Eq. 25
        # H = - sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + self.epsilon), dim=-1)

        # 6. Obliczenie Efektywnej Rangi (Effective Rank) - Eq. 30
        # rho = exp(H)
        effective_rank = torch.exp(entropy)

        return effective_rank

class DoPERegularizationLoss(nn.Module):
    def __init__(self, target_rank=16.0, r=32):
        super().__init__()
        self.entropy_calc = TruncatedMatrixEntropy(truncation_r=r)
        self.target_rank = target_rank

    def forward(self, keys):
        """
        Args:
            keys: Zrotowane przez RoPE klucze [Batch, Heads, Seq, Dim]
        """
        # Obliczamy efektywną rangę dla każdej głowicy
        # [Batch, Heads]
        eff_rank = self.entropy_calc(keys)
        
        # Chcemy, aby ranga była wysoka. 
        # Jeśli ranga < target_rank, nakładamy karę.
        # Loss = ReLU(Target - Current)^2
        loss = F.relu(self.target_rank - eff_rank).pow(2).mean()
        
        return loss

def apply_dope_inference(q, k, truncation_r=32, top_k_heads_to_remove=3):
    """
    Implementacja DoPE-by-Gaussian (Inference time).
    Zastępuje 'zepsute' głowice (o niskiej entropii) szumem Gaussa.
    
    Args:
        q, k: Zrotowane przez RoPE Query i Key [Batch, Heads, Seq, Dim]
    """
    batch_size, n_heads, seq_len, head_dim = k.shape
    
    # 1. Oblicz metrykę entropii dla każdej głowicy (na podstawie Keys)
    # Zgodnie z Table 1, najlepsze wyniki daje kryterium oparte na Keys (post_ntk_key)
    entropy_module = TruncatedMatrixEntropy(truncation_r=truncation_r)
    
    with torch.no_grad():
        # effective_rank: [Batch, Heads]
        ranks = entropy_module(k)
        
        # Uśredniamy rangę po batchu, żeby wybrać głowice globalnie (opcjonalne)
        avg_ranks = ranks.mean(dim=0) # [Heads]

        # 2. Wybierz głowice o NAJNIŻSZEJ entropii (ASC sort order w Table 1)
        # To są głowice, które powodują "attention sinks" i "bright bands".
        _, indices = torch.topk(avg_ranks, k=top_k_heads_to_remove, largest=False)
        
        # Tworzymy maskę głowic do zachowania (mh = 1)
        # Domyślnie wszystko 1
        mask = torch.ones(n_heads, device=q.device)
        mask[indices] = 0.0 # Te głowice zerujemy (zastąpimy szumem)
        
        # Rozszerzamy maskę do kształtu [1, Heads, 1, 1] dla broadcastingu
        mask = mask.view(1, n_heads, 1, 1)

    # 3. Generujemy szum Gaussa (epsilon) - Eq. 36, 37
    # Wariancja szumu powinna pasować do wariancji oryginalnych tensorów
    sigma_k = k.std()
    sigma_q = q.std()
    
    noise_k = torch.randn_like(k) * sigma_k
    noise_q = torch.randn_like(q) * sigma_q

    # 4. Aplikujemy DoPE
    # K_D = m * K + (1-m) * noise
    k_denoised = mask * k + (1 - mask) * noise_k
    q_denoised = mask * q + (1 - mask) * noise_q
    
    return q_denoised, k_denoised
