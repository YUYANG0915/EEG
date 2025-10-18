# VMoGE: Variational Mixture of Graph Neural Experts for Multi-Band Brain Network Analysis

## Overview

Alzheimer's disease (AD) and frontotemporal dementia (FTD) exhibit overlapping electrophysiological features in electroencephalogram (EEG) signals. Traditional methods typically use full-band analysis, which can easily lead to cross-band interference.

This paper proposes **VMoGE (Variational Mixture of Graph Neural Experts)**, combining graph neural networks (GNNs), variational inference (VI), and mixtures of experts (MoE) for interpretable diagnosis of multi-band brain networks.

## Datasets

### 1. Open AD Dataset
- **Subjects**: 88 (AD, FTD, and healthy controls)
- **EEG Configuration**: 19-channel

### 2. Session-based AD Dataset
- **Subjects**: 123
- **Stratification**: CDR = 0/1/2 (disease stage classification)

## Baseline Models

### Transformer-based
- EEGNet
- EEGViT
- Deformer
- ADformer
- MGFormer

### Graph/MoE-based
- GraphMoRE
- GraphDIVE
- MoGE
- Mowst

## Key Findings

### 1. Interpretation of Frequency Band Weights

| Band | Key Finding |
|------|-------------|
| **δ/θ waves** | Highest weight in HC vs AD, reflecting enhanced pathological slow waves |
| **α waves** | Dominant role in HC vs FTD, revealing dysfunction of the frontal-temporal lobe network |
| **β waves** | Used to distinguish FTD from AD, more prominent in young patients |

### 2. Correlation between Cognition and Age

- **δ waves**: Significantly negatively correlated with cognitive scores (MMSE) (**r = −0.336**), reflecting cognitive decline
- **β waves**: Negatively correlated with age, suggesting potential biomarkers for early-onset dementia

### 3. Spatial Distribution Pattern

#### AD Pathology
- **α/θ abnormalities** concentrated in occipital and parietal lobes
- Indicates functional degeneration of the posterior brain region

#### FTD Pathology
- **β abnormalities** concentrated in frontal and temporal lobes
- Indicates degeneration of the anterior brain region

#### Disease Progression
- With disease progression (CDR = 0→2), abnormal activity gradually **expands from posterior to anterior regions**

## Clinical Implications

The multi-band approach reveals distinct electrophysiological signatures:
- **AD**: Posterior brain degeneration with slow-wave dominance
- **FTD**: Anterior brain degeneration with beta-wave alterations
- **Age-related**: Beta waves serve as early-onset dementia biomarkers
- **Cognitive decline**: Delta waves correlate with MMSE deterioration

---

# BSG-Transformer: Balanced Signed Graph Algorithm Unrolling Transformer

## Problem Statement

EEG signals often exhibit both **positively and negatively correlated brain region activities** (e.g., epilepsy patients vs. healthy controls). However, mainstream graph neural networks and Transformer models only consider **positive edges** (positive correlations).

This paper proposes an **interpretable Transformer framework** — **Balanced Signed Graph Algorithm Unrolling Transformer (BSG-Transformer)**, which "unrolls" a balanced signed graph spectral denoising algorithm into a neural network.

## Background: Balanced Signed Graph

### Definition
- A graph containing both **positive and negative edges**
- Considered **"balanced"** if there are no cycles with an odd number of negative edges
- Graphs satisfying this condition can be interpreted through the **Cartwright-Harary Theorem (CHT)**
- Can be mapped to corresponding positive graphs via similarity transformations, enabling frequency and filtering operations in the **spectral domain**

## Model Architecture

### 1️⃣ Balanced Graph Learning (BGL)

- **Nodes**: EEG sensors
- **Edge weights**: Calculated using feature distance (Mahalanobis Distance)
- **Edge signs**: Adjusted based on node polarity β_i (±1) to ensure graph balance
- **Guarantee**: Laplacian matrix is positive semi-definite (PSD) for spectral filtering

**Formulation:**
```
L_B = balanced Laplacian matrix
T = diag(β) (polarity matrix)
L_+ = T·L_B·T^(-1) (corresponding positive graph Laplacian)
```

### 2️⃣ Graph Signal Denoising

- Design ideal low-pass filter: **g_ω(L_+)**
- Preserve low-frequency components (smooth brain region activity)
- **Lanczos approximation** for efficient spectral filtering (linear time, no eigendecomposition)
- Denoising network serves as a **pretext task** to learn signal distribution

### 3️⃣ Algorithm Unrolling

- Iteratively unroll **"graph learning module + low-pass filtering module"** into neural network layers
- Each layer learns its own **cutoff frequency ω**
- Equivalent to interpretable Transformer layers:
  - **Graph attention ≈ Self-attention mechanism**
  - **Normalized edge weights w̄_ij ≈ Attention scores**
- Extract node features using shallow CNN with minimal parameters

### 4️⃣ Denoiser-Based Classification

Train two denoisers:
- **Ψ_0(·)**: Healthy EEG denoiser
- **Ψ_1(·)**: Epilepsy EEG denoiser

**Classification rule:**
```
c* = argmin_{c∈{0,1}} ||y - Ψ_c(y)||²
```
The class with **lower reconstruction error** is predicted.

## Architecture Diagram

```
Input EEG Signal (y)
    ↓
[Shallow CNN Feature Extraction]
    ↓
[Layer 1: BGL + Spectral Filter (ω₁)]
    ↓
[Layer 2: BGL + Spectral Filter (ω₂)]
    ↓
    ...
    ↓
[Layer K: BGL + Spectral Filter (ω_K)]
    ↓
[Reconstruction: ŷ = Ψ_c(y)]
    ↓
[Classification: argmin ||y - ŷ||²]
```

## Datasets

- **Turkish Epilepsy EEG Dataset**
- **TUH Abnormal EEG Corpus**

## Baseline Models

- DGCNN
- GIN
- EEGNet

## Key Innovations

### 1. Theoretical Contributions

| Innovation | Description |
|------------|-------------|
| **Balanced Signed Graph + Spectral Filtering** | First integration with Transformer structure |
| **Algorithm Unrolling** | Interpretable modeling approach |

### 2. Interpretable Transformer Mechanism

| Component | Mapping |
|-----------|---------|
| **Graph Attention** | ↔ Self-Attention |
| **Cutoff Frequency ω** | ↔ Attention weight control |

### 3. Efficiency

| Metric | Performance |
|--------|-------------|
| **Parameters** | Only 15,000 |
| **Training Time** | 40% of EEGNet |
| **Inference Time** | 55 seconds (same dataset) |

### 4. Generalization

- ✅ Validated on both **LOSO** and **TUH** datasets
- ✅ Statistical significance: **p < 0.001**

## Advantages Over Existing Methods

| Feature | Traditional GNN/Transformer | BSG-Transformer |
|---------|----------------------------|-----------------|
| **Edge Types** | Positive only | Positive + Negative (balanced) |
| **Interpretability** | Black-box attention | Spectral filtering with physical meaning |
| **Parameters** | Millions | 15K |
| **Training Efficiency** | Baseline | 2.5× faster |
| **Domain Knowledge** | Implicit | Explicit (spectral graph theory) |

## Mathematical Foundation

**Balanced Graph to Positive Graph Mapping:**
```
L_+ = T·L_B·T^(-1)
where T = diag(β), β_i ∈ {-1, +1}
```

**Spectral Low-Pass Filter:**
```
g_ω(L_+) = ideal low-pass filter with cutoff ω
X_filtered = g_ω(L_+) · X
```

**Lanczos Approximation:**
```
Linear time complexity O(|E|·K)
No eigendecomposition required
```

## Clinical Implications

- **Interpretable EEG analysis** through spectral graph theory
- **Efficient deployment** with minimal computational requirements
- **Robust classification** via denoiser-based approach
- **Captures both excitatory and inhibitory** brain network interactions

## Conclusion

BSG-Transformer achieves **state-of-the-art performance** with:
- 📊 **Superior accuracy** on epilepsy detection
- 🧠 **Interpretable mechanism** via spectral filtering
- ⚡ **High efficiency** (15K parameters, fast training)
- 🔬 **Strong theoretical foundation** (balanced signed graph theory)
- ✨ **Novel architecture** bridging algorithm unrolling and Transformers

---

# Spatial-Functional Awareness Transformer-Based Graph Archetype Contrastive Learning for Decoding Visual Neural Representations from EEG
## Problem Statement

EEG signals are characterized by:
- **High dimensionality**
- **High noise levels**
- **Complex non-Euclidean structure**

Current research in **EEG visual decoding** suffers from **insufficient utilization of spatial-functional coupling information**.

## Proposed Framework: SFTG

This paper proposes **SFTG (Spatial-Functional awareness Transformer-based Graph Archetype Contrastive Learning)**, which models EEG signals as graph structures where each electrode is a node, integrating:
- **Spatial connectivity** (anatomical relationships)
- **Functional connectivity** (neural correlations)

These form a **spatiotemporal graph structure** for comprehensive brain activity modeling.

---

## Core Innovations

### 1️⃣ Graph Archetype Contrastive Learning (GAC)

A novel contrastive learning approach designed specifically for EEG graph representations.

#### Mechanism
- **Cluster** EEG graph representations to form **archetype features** for each subject
- Perform **dual-level contrastive learning**:
  - **Sequence-level**: Temporal dynamics
  - **Channel-level**: Spatial patterns
- Enables the model to identify **individual differences** and **neural representation patterns**

#### Theoretical Foundation
Essentially implements an **Expectation-Maximization (EM)** optimization strategy to combat:
- High EEG variability
- Signal noise

```
┌─────────────────────────────────────┐
│   Graph Archetype Contrastive      │
│         Learning (GAC)             │
├─────────────────────────────────────┤
│  1. Cluster EEG graphs → Archetypes│
│  2. Sequence-level contrastive     │
│  3. Channel-level contrastive      │
│  4. EM-style optimization          │
└─────────────────────────────────────┘
         ↓
    Combat variability & noise
```

### 2️⃣ EEG Graph Transformer (EGT)

An extension of traditional multi-head attention to graph structures.

#### Key Features

| Component | Description |
|-----------|-------------|
| **Full-Relation Heads (FR)** | Multi-head attention extended to graph topology |
| **Laplacian Position Encoding** | Graph Laplacian features as positional encoding for brain region awareness |
| **Local + Global Dependencies** | Captures dynamic interactions across different brain regions |

#### Architecture

```
Input EEG Graph
    ↓
[Node Features + Laplacian Position Encoding]
    ↓
┌─────────────────────────────────┐
│  Full-Relation Multi-Head       │
│  Attention (FR-Attention)       │
│  • Spatial connectivity aware   │
│  • Functional connectivity aware│
└─────────────────────────────────┘
    ↓
[Local Dependencies] + [Global Dependencies]
    ↓
[Graph Representation]
```

---

## Complete SFTG Pipeline

```
Raw EEG Signal
    ↓
┌──────────────────────────────────────┐
│  Graph Construction                  │
│  • Spatial connectivity (anatomy)    │
│  • Functional connectivity (correlation)│
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  EEG Graph Transformer (EGT)        │
│  • Laplacian position encoding       │
│  • Full-Relation multi-head attention│
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  Graph Archetype Contrastive (GAC)  │
│  • Cluster → Archetypes              │
│  • Sequence-level contrastive        │
│  • Channel-level contrastive         │
└──────────────────────────────────────┘
    ↓
Visual Decoding Output
```

---

## Dataset

**THINGS-EEG Dataset**

### Experimental Settings

| Scenario | Description |
|----------|-------------|
| **Subject-dependent** | Single subject train/test (within-subject) |
| **Subject-independent** | Cross-subject generalization (transfer learning) |

---

## Baseline Models

- **BraVL**
- **NICE**
- **ATM-S**
- **VE-SDN**
- **UBP**

---

## Key Results

### 1. Semantic Similarity Analysis (RSA)

Model outputs show **clear clustering** by semantic categories:
- 🐾 Animals
- 🍎 Food
- 🔧 Tools
- And more...

**EEG representations exhibit semantic structure aligned with visual categories.**

### 2. t-SNE Visualization

| Finding | Interpretation |
|---------|---------------|
| **High alignment** between EEG representations and image semantics | Model learns **cross-modal consistent neural features** |
| **Semantic clusters** in embedding space | Neural patterns reflect categorical organization |
| **Clear separation** between categories | Robust discriminative representations |

#### Visualization Example

```
t-SNE Embedding Space
    
    🐾 Animals cluster
        • Dog, Cat, Lion...
    
    🍎 Food cluster
        • Apple, Bread, Cake...
    
    🔧 Tools cluster
        • Hammer, Scissors, Wrench...
    
→ EEG neural patterns mirror semantic organization
```

---

## Technical Highlights

### Spatial-Functional Integration

| Connectivity Type | Information Captured |
|-------------------|---------------------|
| **Spatial** | Anatomical proximity, physical electrode layout |
| **Functional** | Neural correlation, information flow |
| **Combined** | Comprehensive brain network dynamics |

### Graph Laplacian Position Encoding

```python
# Conceptual formulation
L = D - A  # Laplacian matrix
λ, V = eigen_decomposition(L)
position_encoding = V[:, :k]  # First k eigenvectors

# Encodes brain region topology
```

### EM-Style Optimization in GAC

```
E-step: Assign EEG graphs to archetypes (clustering)
M-step: Update archetypes via contrastive learning
Iterate until convergence
```

---

## Advantages Over Existing Methods

| Feature | Traditional Methods | SFTG |
|---------|-------------------|------|
| **Connectivity** | Spatial only or functional only | **Both integrated** |
| **Architecture** | CNN/RNN/basic Transformer | **Graph-aware Transformer** |
| **Contrastive Learning** | Image-level only | **Sequence + Channel dual-level** |
| **Subject Variability** | Poor handling | **Archetype-based robustness** |
| **Interpretability** | Limited | **High (RSA, t-SNE, graph analysis)** |

---

## Clinical & Research Implications

| Application | Potential Impact |
|-------------|-----------------|
| **Brain-Computer Interfaces (BCI)** | Improved visual decoding for assistive devices |
| **Cognitive Neuroscience** | Understanding neural coding of visual perception |
| **Cross-subject Transfer** | Reduced calibration time for BCIs |
| **Semantic Brain Mapping** | Identifying neural correlates of semantic categories |

---

## Conclusion

**SFTG** achieves state-of-the-art EEG visual decoding through:

- 🧠 **Comprehensive modeling** of spatial-functional brain networks
- 🎯 **Novel contrastive learning** tailored for EEG graphs
- 🔍 **High interpretability** with semantic alignment
- 🚀 **Strong generalization** across subjects
- 📊 **Clear visualization** of learned neural representations

The framework bridges the gap between **brain activity patterns** and **semantic visual understanding**, advancing both theoretical neuroscience and practical BCI applications.
