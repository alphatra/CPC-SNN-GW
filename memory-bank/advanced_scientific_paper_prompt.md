# Advanced Scientific Paper Prompt: CPC-SNN-GW Framework

## Objective
Generate a comprehensive, publication-ready scientific article detailing the novel CPC-SNN-GW pipeline for gravitational wave detection. The article must be technically rigorous, self-contained, and suitable for submission to a high-impact physics or machine learning journal (e.g., Physical Review D, Nature Machine Intelligence).

## Target Audience
Researchers in gravitational wave astronomy, machine learning for science, and neuromorphic computing.

## Core Requirements

1.  **Title & Abstract:**
    *   Propose a concise, informative title.
    *   Write a structured abstract (Background, Methods, Results, Conclusions) of ~250 words.

2.  **Introduction (1-2 pages):
    *   Contextualize the challenge of real-time, low-latency GW detection amidst non-stationary noise.
    *   Critically review limitations of current methods (matched filtering, standard deep learning).
    *   Introduce the core innovation: the integration of Contrastive Predictive Coding (CPC) for unsupervised representation learning with Spiking Neural Networks (SNNs) for energy-efficient, event-driven classification.
    *   Clearly state the hypothesis and the specific scientific/technical contributions of this work.

3.  **Methods (3-5 pages):
    *   **Pipeline Architecture:** Provide a detailed, multi-panel figure (describe conceptually) of the end-to-end CPC-SNN-GW pipeline. Break down the stages: data ingestion, CPC pre-training, spike encoding, SNN classification, and output.
    *   **CPC Encoder (Mathematical Formulation):
        *   Define the autoregressive context network `g_\theta` and the encoder network `f_\theta`.
        *   Derive the InfoNCE loss function `L_{CPC}` for predicting future time steps `k` ahead. Explicitly show the equation:
          `L_{CPC} = -\log \frac{\exp(s_{t+k})}{\sum_{i=0}^{N} \exp(s_{t+i})}`
          where `s_{t+k} = z_{t+k}^\top W g_\theta(c_t)` is the similarity score, `z_{t+k}` is the encoded future sample, `c_t` is the context vector, and `W` is a learnable weight matrix. Explain the role of negative samples.
        *   Detail the architecture of `f_\theta` (e.g., 1D ResNet) and `g_\theta` (e.g., GRU).
    *   **Spike Encoding & SNN Classifier (Mathematical Formulation):
        *   Describe the conversion of the CPC-learned latent representation `z_t` into spike trains. Specify the encoding method (e.g., rate coding, latency coding).
        *   Define the SNN dynamics using the Leaky Integrate-and-Fire (LIF) neuron model. Provide the differential equation for membrane potential `V_m`:
          `\tau_m \frac{dV_m}{dt} = -(V_m - V_{rest}) + R_m I(t)`
          where `\tau_m` is the membrane time constant, `V_{rest}` is the resting potential, `R_m` is the membrane resistance, and `I(t)` is the synaptic current.
        *   Specify the reset mechanism upon spike emission (`V_m \leftarrow V_{reset}`).
        *   Detail the SNN architecture (e.g., number of layers, neurons per layer, connectivity).
    *   **Spike Bridge (Mathematical Formulation):
        *   Explain the purpose of the Spike Bridge module (e.g., gradient approximation, temporal alignment).
        *   If using a surrogate gradient, define the surrogate function `\sigma'(V_m - V_{th})` (e.g., sigmoid, arctan) used during backpropagation through time (BPTT).
    *   **Training Procedure:
        *   Describe the two-stage training: (1) Unsupervised CPC pre-training on unlabeled GW data, (2) Supervised fine-tuning of the SNN classifier (and potentially the encoder) on labeled data.
        *   Specify optimization algorithms, learning rates, batch sizes, and hardware used (e.g., GPU for CPC, potential neuromorphic hardware for SNN inference).

4.  **Results (2-3 pages):
    *   **Dataset:** Describe the training and testing datasets (e.g., simulated signals from `gw_synthetic_generator.py`, real O3 data from `real_ligo_integration.py`). Include statistics.
    *   **Baselines:** Compare against relevant baselines (e.g., standard CNN, pure SNN, matched filter) implemented in `pycbc_baseline.py` and `baseline_comparisons.py`.
    *   **Metrics:** Report key metrics: detection accuracy, false alarm rate (FAR), latency, computational efficiency (FLOPs, energy consumption estimation), and robustness to glitches (using `glitch_injector.py`).
    *   **Figures & Tables:** Include conceptual figures for the pipeline and SNN dynamics, and tables comparing performance.

5.  **Discussion (1-2 pages):
    *   Interpret the results. Why does CPC-SNN-GW outperform baselines?
    *   Discuss the significance of unsupervised pre-training for GW data.
    *   Analyze the trade-offs between accuracy and energy efficiency.
    *   Address limitations and potential failure modes.

6.  **Conclusion (0.5 page):
    *   Summarize the main findings and contributions.
    *   Outline future work (e.g., deployment on neuromorphic hardware, multi-detector analysis).

7.  **Code Integration:
    *   **CRITICAL:** Integrate **exact, verbatim code snippets** from the following key files to illustrate the methodology. Ensure the code is correct and matches the described equations.
        *   CPC Encoder: `models/cpc_encoder.py` (show `forward` method and context network)
        *   CPC Loss: `models/cpc_losses.py` (show `InfoNCELoss` class)
        *   SNN Classifier: `models/snn_classifier.py` (show LIF neuron implementation and network structure)
        *   Spike Bridge: `models/spike_bridge.py` (show surrogate gradient function)
        *   Data Preprocessing: `data/gw_preprocessor.py` (show key normalization steps)
        *   Training Loop: `training/complete_enhanced_training.py` (show the two-stage training logic)
    *   Place each code snippet in a dedicated subsection of the Methods or an Appendix, with a caption explaining its role.

## Verification Protocol (Chain-of-Drafts)

1.  **Draft 1 (Structure & Content):** Generate the complete article following the outline above. Focus on logical flow, completeness, and clarity.
2.  **Draft 2 (Mathematical & Code Verification):** 
    *   **Triple-Check:** Meticulously verify every mathematical equation for correctness and consistency with the cited literature and the project's implementation.
    *   **Code-Text Alignment:** For every code snippet, cross-reference it with the corresponding file in the repository (`read_file` each one). Ensure the snippet is accurate, complete for its context, and correctly described in the text. Verify that variable names in the code match those used in the equations (e.g., `z_t`, `c_t`).
    *   **Equation-Code Alignment:** Ensure the mathematical formulation of `L_{CPC}`, the LIF model, and the surrogate gradient exactly matches the logic implemented in the code.
3.  **Draft 3 (Final Polish):** Refine the language for conciseness and impact. Ensure all figures are properly referenced. Perform a final proofread for grammar and typos.

## Output
Return the final, verified scientific article in Markdown format, ready for conversion to LaTeX.