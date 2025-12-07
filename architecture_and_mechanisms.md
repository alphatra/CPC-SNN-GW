# Deep Analysis: CPC-SNN Architecture & Mechanics

## 1. Adaptive Input Normalization: DAIN Layer
**File:** `src/models/layers.py`

Most SNNs fail because they rely on fixed thresholds ($V_{th}$). If the input signal amplitude varies significantly (common in GW data due to non-stationary noise), the neurons either remain silent (sub-threshold) or fire uncontrollably (saturation). **DAIN (Deep Adaptive Input Normalization)** solves this by dynamically recalibrating the input statistics *before* they reach the SNN.

### Mechanism Details
DAIN performs a 3-step transformation on the input tensor $X \in \mathbb{R}^{B \times C \times T}$:

1.  **Adaptive Centering:**
    *   Calculates the mean $\mu$ along the time dimension.
    *   Learns a shift parameter $\alpha$ via a linear layer.
    *   $X' = X - \alpha(\mu)$
    *   *Effect:* Removes DC offsets and drifts.

2.  **Adaptive Scaling:**
    *   Calculates the root-mean-square (RMS) $\sigma$.
    *   Learns a scaling parameter $\beta$.
    *   $X'' = \frac{X'}{\beta(\sigma) + \epsilon}$
    *   *Effect:* Normalizes the signal amplitude so that noise fits within a standard range, ensuring the SNN's $V_{th}$ works robustly across different noise floors.

3.  **Adaptive Gating (Attention):**
    *   Computes a gating scalar $g \in [0, 1]$ using a sigmoid activation on the mean features.
    *   $X_{out} = X'' \cdot g$
    *   *Effect:* Can completely suppress channels that are deemed "useless" or too noisy by the network.

---

## 2. Encoding Strategy: Learnable vs. Delta Modulation
**File:** `src/models/cpc_snn.py` & `src/models/encoders.py`

The encoder is the bridge between the continuous world (waveforms) and the discrete world (spikes).

### A. Learnable Encoding (Dense) - *Current Default*
*   **Implementation:** `nn.Identity()`
*   **Logic:** The continuous signal $X_{out}$ (from DAIN) is passed directly to the first `nn.Conv1d` layer of the `SpikingCNN`.
*   **Physics:** The first convolutional layer acts as a **learnable current injector**. It learns filters $W$ such that the current $I = W * X$ drives the LIF neurons to spike at optimal times for feature extraction. 
*   **Advantage:** Preserves information. Unlike threshold-based encoders which discard sub-threshold details, a learnable layer can integrate weak signals over time until they trigger a spike.

### B. Fast Delta Modulation (Sparse) - *Alternative*
*   **Implementation:** `FastDeltaEncoder`
*   **Logic:** Generates a spike if the change in signal amplitude exceeds a threshold ($\Delta V > \delta$).
*   **Formula:** Vectorized `diff` calculation. $S_t = 1 \iff |x_t - x_{t-1}| > \delta$.
*   **Advantage:** Extremely sparse (energy efficient), but information-lossy for low-frequency, low-amplitude signals like GWs.

---

## 3. Feature Extractor: Spiking CNN
**File:** `src/models/architectures.py`

This module extracts temporal features from the input. It is fully spiking, meaning information is transmitted as binary events (0/1).

### Architecture Structure
It consists of 3 "Converge" blocks. A single block performs:
1.  **Conv1d:** Spatial/Temporal filtering. Stride is used heavily for downsampling (effective compression of $32\times$).
2.  **BatchNorm1d:** Stabilizes membrane potentials.
3.  **LIF Neuron:** Leaky Integrate-and-Fire.
    *   **Surrogate Gradient:** `atan` (Arctangent). This is critical. Since spikes are non-differentiable step functions, we use a smooth `atan` function during backpropagation to allow gradient flow.
    *   **Beta ($\beta$):** Decay rate (default 0.85). Controls memory. High $\beta$ = long memory.
4.  **MaxPool1d:** Reduces dimensionality.

**Output:** A sequence of latent vectors $z_t$, representing the "state" of the gravitational wave at each time step.

---

## 4. Context Network: Recurrent SNN (RSNN)
**File:** `src/models/architectures.py` : `RSNN`

Standard CNNs only "see" a local window. To detect a chirp (which evolves over seconds), we need global context.
The RSNN uses **Recurrent LIF** neurons. The membrane potential of the recurrent layer serves as the **Context Vector** $c_t$. This vector accumulates history, acting effectively as a "memory state" of the entire event seen so far.

---

## 5. Training Objective: Hybrid SSL (Barlow + CPC)
**File:** `src/train/cpc_trainer.py`

The model learns without labels by solving two simultaneous pretext tasks using a **Siamese Network** setup (Shared Weights).

### Mechanism A: "GW Twins" (Barlow Twins) - The Invariance Task
*   **Logic:** If H1 sees a wave and L1 sees a wave, their internal representations ($c_{H1}, c_{L1}$) should be identical, even if the noise is totally different.
*   **Implementation:** Calculates Cross-Correlation matrix $\mathcal{C}$ between batch-normalized embeddings of H1 and L1.
    *   **On-Diagonal Loss:** $\sum (1 - \mathcal{C}_{ii})^2$. Forces H1 and L1 to agree.
    *   **Off-Diagonal Loss:** $\sum (0 - \mathcal{C}_{ij})^2$. Forces different feature dimensions to be independent (decorrelation), preventing "feature collapse" where all neurons learn the same thing.

### Mechanism B: CPC (Contrastive Predictive Coding) - The Prediction Task
*   **Logic:** To understand a wave, you must be able to predict its future.
*   **Implementation:** InfoNCE Loss.
    *   Given context $c_t$, predict $z_{t+k}$ using linear projections $W_k$.
    *   **Positive Pair:** The actual $z_{t+k}$ from the same signal.
    *   **Negative Pairs:** $z_{t'}$ from other times or other signals in the batch.
    *   **Formula:** $\mathcal{L} = -\log \frac{\exp(z_{t+k}^T W_k c_t)}{\sum_{neg} \exp(z_{neg}^T W_k c_t)}$

**Total Loss:** `Loss_Barlow + 0.5 * (Loss_CPC_H1 + Loss_CPC_L1)`
*Note: We average CPC loss from both detectors to ensure the model predicts well regardless of which detector it's looking at.*

---

## 6. Monitoring: DoPE (Denoising Rotary Position Embedding) Metric
**File:** `src/models/dope.py`

We use the **Effective Rank** metric from the DoPE paper to monitor the "health" of the latent space.

*   **Logic:** If the model collapses, all embedding vectors become identical or lie on a low-dimensional manifold.
*   **Calculation (`TruncatedMatrixEntropy`):**
    1.  Compute Singular Values (SVD) of the embedding matrix.
    2.  Normalize squared singular values to get a probability distribution.
    3.  Compute Shannon Entropy $H$ of this distribution.
    4.  $Rank_{eff} = e^H$.
*   **Interpretation:** A drop in Effective Rank signals **Dimensional Collapse** – the model is losing capacity and learning simple/trivial features.

---

## 7. Data Physics: Preferential Accretion
**File:** `src/data_handling/gw_utils.py`

This is a subtle but physically crucial detail in data generation.
*   **Hypothesis:** Binary Black Holes in AGN disks tend to accrete gas, which drives their mass ratio $q = m_2/m_1$ towards 1.0 (Equal Mass).
*   **Implementation:**
    *   `preferential_prob=0.5`: 50% of synthetic signals are generated with "hard" random masses.
    *   The other 50% undergo a transformation where $q$ is heavily biased towards 1.0.
*   **Why:** Training on equal-mass binaries is generally "easier" and provides a stable baseline, while the random set provides robust generalization.
