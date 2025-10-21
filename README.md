> [*Variational Mixture of Graph Neural Experts for Alzheimer’s Disease Biomarker Recognition in EEG Brain Networks*], [Aug 8, 2021]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 🧠 Multi-band EEG brain network analysis for Alzheimer's disease (AD) and frontotemporal dementia (FTD) diagnosis
- _Author_: Jun-En Ding, Anna Zilverstand, Shihao Yang, Albert Chih-Chieh Yang, and Feng Liu
- _Group_: Stevens Institute of Technology, National Yang-Ming Chiao Tung University, University of Minnesota
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 AD and FTD exhibit **overlapping electrophysiological features** in EEG signals. Traditional methods use **full-band analysis**, which easily leads to **cross-band interference**.

- _Focus problem_: 🔍 How to leverage **multi-band EEG analysis** (δ, θ, α, β waves) to distinguish between AD, FTD, and healthy controls while capturing **spatial-functional coupling** information?

- _Why important_: 💡 Different frequency bands carry distinct pathological information:
  - **δ/θ waves**: Pathological slow waves in AD
  - **α waves**: Frontal-temporal dysfunction in FTD  
  - **β waves**: Early-onset dementia biomarkers
  
  Understanding **which bands matter for which diagnosis** enables more accurate and interpretable disease classification.
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 Different frequency bands have **distinct diagnostic importance**:

| Band | Key Finding | Clinical Significance |
|------|-------------|----------------------|
| **δ/θ waves** | Highest weight in HC vs AD | Enhanced pathological slow waves |
| **α waves** | Dominant in HC vs FTD | Frontal-temporal lobe dysfunction |
| **β waves** | Distinguish FTD from AD | More prominent in young patients |

**Spatial patterns also differ**:
- **AD**: α/θ abnormalities in **occipital & parietal** (posterior brain)
- **FTD**: β abnormalities in **frontal & temporal** (anterior brain)
- **Disease progression** (CDR 0→2): Abnormalities expand **posterior → anterior**

**Correlations**:
- δ waves ↔ MMSE: **r = -0.336** (cognitive decline)
- β waves ↔ age: Negative correlation (early-onset marker)

- _Why necessary_: 🏥 Full-band analysis **loses frequency-specific information** critical for differential diagnosis. A **multi-band, graph-based approach** can capture both spectral and spatial patterns unique to each dementia type.
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 Propose **VMoGE (Variational Mixture of Graph Neural Experts)**, combining:
  - **Graph Neural Networks (GNNs)**: Model spatial brain connectivity
  - **Variational Inference (VI)**: Handle uncertainty in band selection
  - **Mixture of Experts (MoE)**: Assign different frequency bands to specialized experts

- _Method_: 🔧
  - **Datasets**:
    - Open AD Dataset: 88 subjects (AD, FTD, HC), 19-channel EEG
    - Session-based AD: 123 subjects, stratified by CDR = 0/1/2
  - **Architecture**: Multi-band graph experts process δ, θ, α, β bands separately, then combine via variational gating
  - **Baselines**: 
    - Transformer: EEGNet, EEGViT, Deformer, ADformer, MGFormer
    - Graph/MoE: GraphMoRE, GraphDIVE, MoGE, Mowst

- _Result_: 📈
  - **Band importance successfully identified** (δ/θ for AD, α for FTD, β for FTD vs AD)
  - **Spatial patterns revealed**: Posterior (AD) vs. Anterior (FTD) degeneration
  - **Strong correlations**: δ with cognition (r=-0.336), β with age
  - **Disease progression tracked**: CDR 0→2 shows posterior→anterior spread

- _Conclusion_: 🎓 **VMoGE enables interpretable multi-band analysis** that reveals:
  - ✅ **Frequency-specific biomarkers** for AD/FTD differential diagnosis
  - ✅ **Spatial-spectral coupling** patterns unique to each disease
  - ✅ **Disease progression mapping** via band weight evolution
  - ✅ **Clinical insights**: Posterior (AD) vs. Anterior (FTD), slow waves (AD) vs. fast waves (FTD in young patients)
  
  Multi-band graph modeling > full-band analysis for interpretable dementia diagnosis. 🧠
</details>
</details>

---

> [*Lightweight Transformer For EEG Classification Via Balanced Signed Graphed Algorithmic Unrolling*], [Oct 17, 2025]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: ⚡ Interpretable EEG classification using balanced signed graphs with both positive and negative correlations (epilepsy detection)
- _Core Author_: Junyi Yao, Parham Eftekhar, Gene Cheung, Xujin Chris Liu, Yao Wang, Wei Hu
- _Core Group_: Peking University, York University, New York University
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 EEG signals exhibit both **positively and negatively correlated brain region activities** (e.g., epilepsy vs. healthy). However, mainstream GNNs and Transformers only consider **positive edges** (positive correlations), losing critical inhibitory/excitatory dynamics.

- _Focus problem_: 🔍 How to build an **interpretable graph-based model** that captures **both positive and negative brain correlations** while maintaining computational efficiency?

- _Why important_: 💡 Brain networks involve both:
  - **Excitatory connections** (positive correlations)
  - **Inhibitory connections** (negative correlations)
  
  Ignoring negative edges means **missing half the story** of neural dynamics. Traditional models can't distinguish between positively and negatively correlated regions.
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **Balanced signed graphs** (containing both +/− edges) can be:
  - Theoretically grounded via **Cartwright-Harary Theorem**
  - Mapped to positive graphs via similarity transformation
  - Analyzed in **spectral domain** using graph Laplacian
  
  **Algorithm unrolling** transforms a **spectral denoising algorithm** into an **interpretable Transformer**:
  - Graph attention ↔ Self-attention
  - Cutoff frequency ω ↔ Attention weight control

- _Why necessary_: 🏥 **Interpretability + Efficiency**:
  - Traditional black-box models lack neurophysiological interpretation
  - Complex Transformers require millions of parameters
  - Need **lightweight, interpretable, theoretically grounded** approach
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 Propose **BSG-Transformer**, which "unrolls" a **balanced signed graph spectral denoising algorithm** into neural network layers:

| Component | Description |
|-----------|-------------|
| **Balanced Graph Learning** | Construct graphs with +/− edges using node polarity β_i (±1) |
| **Spectral Filtering** | Low-pass filter g_ω(L_+) preserves smooth brain activity |
| **Algorithm Unrolling** | Each layer = graph learning + filtering with learnable ω |
| **Denoiser-Based Classification** | Train separate denoisers Ψ₀ (healthy), Ψ₁ (epilepsy) |

- _Method_: 🔧
  - **Graph construction**:
    ```
    L_+ = T·L_B·T^(-1)
    T = diag(β), β_i ∈ {-1, +1}
    Edge weights: Mahalanobis Distance
    ```
  - **Spectral filtering**: Lanczos approximation (linear time, no eigendecomposition)
  - **Classification**: `c* = argmin ||y - Ψ_c(y)||²` (reconstruction error)
  - **Datasets**: Turkish Epilepsy EEG, TUH Abnormal EEG Corpus
  - **Baselines**: DGCNN, GIN, EEGNet

- _Result_: 📈

| Metric | Performance |
|--------|-------------|
| **Parameters** | Only **15,000** (vs. millions in Transformers) |
| **Training Time** | **40% of EEGNet** (2.5× faster) |
| **Inference Time** | **55 seconds** |
| **Accuracy** | State-of-the-art on both datasets |
| **Statistical Significance** | **p < 0.001** |

**Advantages**:
- ✅ **Positive + negative edges** (balanced signed graph)
- ✅ **Interpretable** (spectral filtering = neurophysiological meaning)
- ✅ **Ultra-efficient** (15K params, 2.5× faster training)
- ✅ **Theoretically grounded** (spectral graph theory)

- _Conclusion_: 🎓 **BSG-Transformer achieves state-of-the-art epilepsy detection** with:
  - 🧠 **Interpretable mechanism** via spectral filtering (not black-box attention)
  - ⚡ **High efficiency** (15K parameters, fast training)
  - 🔬 **Strong theoretical foundation** (balanced signed graph theory)
  - ✨ **Novel architecture** bridging algorithm unrolling and Transformers
  - 📊 **Captures excitatory AND inhibitory** brain dynamics
  
  Algorithm unrolling + signed graphs = interpretable, efficient EEG analysis. 🏆
</details>
</details>

---

> [*Spatial-Functional Awareness Transformer-based Graph Archetype Contrastive Learning for Decoding Visual Neural Representation from EEG*], [Oct 9, 2025]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 👁️ EEG visual decoding - reconstructing what people see from brain signals
- _Author_: Yueming Sun, Long Yang
- _Group_: Durham University
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 EEG signals have **high dimensionality, high noise, complex non-Euclidean structure**. Current visual decoding research suffers from **insufficient utilization of spatial-functional coupling information**.

- _Focus problem_: 🔍 How to integrate **spatial connectivity** (anatomical layout) and **functional connectivity** (neural correlations) to decode visual information from noisy EEG signals?

- _Why important_: 💡 Brain activity during visual perception involves:
  - **Spatial relationships**: Physical electrode positions matter
  - **Functional relationships**: Which brain regions communicate
  
  Traditional methods use **one or the other**, missing their **synergy**. This limits accuracy in decoding what people are seeing from brain signals.
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **Spatial-functional integration + archetype learning** enables robust visual decoding:

**Key insights**:
- **Semantic clustering emerges**: Animals, food, tools cluster separately in EEG embedding space
- **Cross-modal alignment**: EEG representations align with image semantics (t-SNE visualization)
- **Subject variability handled**: Archetype-based approach robust to individual differences

| Connectivity | Information |
|--------------|-------------|
| **Spatial** | Anatomical proximity, electrode layout |
| **Functional** | Neural correlation, information flow |
| **Combined** | Comprehensive brain network dynamics 🌟 |

- _Why necessary_: 🏥 **Brain-Computer Interfaces (BCIs)** need:
  - Accurate visual decoding for assistive devices
  - Robust cross-subject generalization (no per-person calibration)
  - Interpretable representations showing what brain patterns mean
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 Propose **SFTG** combining:
  - **Spatial + Functional** graph construction
  - **EEG Graph Transformer (EGT)**: Graph-aware attention with Laplacian position encoding
  - **Graph Archetype Contrastive (GAC)**: Dual-level (sequence + channel) contrastive learning with EM-style clustering

- _Method_: 🔧

- **Dataset**: THINGS-EEG (visual stimuli)
- **Scenarios**: Subject-dependent + **Subject-independent** (cross-subject)
- **Baselines**: BraVL, NICE, ATM-S, VE-SDN, UBP

- _Result_: 📈

**Semantic Analysis (RSA)**:
- ✅ Clear clustering by categories: 🐾 Animals, 🍎 Food, 🔧 Tools
- ✅ EEG representations mirror semantic organization

**t-SNE Visualization**:
- ✅ High alignment between EEG and image semantics
- ✅ Robust discriminative representations

**Performance**:
- ✅ State-of-the-art on subject-dependent
- ✅ **Strong cross-subject generalization** (subject-independent)

| Feature | Traditional | SFTG |
|---------|-------------|------|
| Connectivity | Spatial OR Functional | **Both integrated** ✅ |
| Architecture | CNN/RNN/basic Transformer | **Graph-aware Transformer** ✅ |
| Contrastive | Image-level only | **Sequence + Channel dual-level** ✅ |
| Variability | Poor handling | **Archetype-based robustness** ✅ |

- _Conclusion_: 🎓 **SFTG achieves state-of-the-art EEG visual decoding** through:
  - 🧠 **Comprehensive modeling** of spatial-functional brain networks
  - 🎯 **Novel contrastive learning** tailored for EEG graphs (dual-level, archetype-based)
  - 🔍 **High interpretability**: Semantic clustering (animals, food, tools), cross-modal alignment
  - 🚀 **Strong generalization** across subjects (EM-style optimization handles variability)
  - 📊 **Clear visualization** of learned neural representations
  
  Spatial-functional integration + archetype contrastive learning = robust, interpretable visual decoding from brain signals. 👁️🧠
</details>
</details>

---

> [*DRDCAE-STGNN: Discriminative Autoencoder with Spatio-Temporal Graph Learning for Motor Imagery*], [Sep 7, 2025]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 🤖 Motor Imagery (MI) classification for Brain-Computer Interfaces - decoding imagined movements from EEG
- _Core Author_: Yi Wang, Haodong Zhang and Hongqi Li
- _Core Group_: Northwestern Polytechnical University
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 Traditional CNN/RNN models for Motor Imagery (MI) only capture **local features** and ignore **complex spatio-temporal dependencies** between brain regions during imagined movements.

- _Focus problem_: 🔍 How to model **both spatial connectivity** (which brain regions interact) and **temporal dynamics** (how activity evolves) for robust MI classification?

- _Why important_: 💡 Motor imagery involves:
  - **Spatial patterns**: Motor cortex, sensorimotor regions
  - **Temporal evolution**: Preparation → execution phases
  - **Brain connectivity**: Coordinated network activity
  
  Traditional models treating channels independently **miss the network-level dynamics** critical for accurate MI decoding.
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **Dual-objective autoencoder + dynamic graph learning** captures MI patterns:

**Key insights**:
- **Mutual Information (MI)-based graphs** reveal functional brain connectivity
- **Discriminative latent space** improves class separability
- **Hierarchical spatio-temporal modeling** captures multi-scale dependencies

| Component | Innovation |
|-----------|-----------|
| **MI-based adjacency** | Data-driven, adaptive, captures non-linear dependencies |
| **Dual loss** | Reconstruction + discrimination = better features |
| **Dynamic graphs** | Subject-specific connectivity patterns |

- _Why necessary_: 🏥 **BCI applications** (assistive devices for paralyzed patients) need:
  - High accuracy across different MI tasks (left/right hand, feet, tongue)
  - Robust cross-subject performance (no extensive calibration)
  - Interpretable brain connectivity patterns
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 Propose **DRDCAE-STGNN** combining:
  - **DRDCAE**: Discriminative autoencoder with dual loss (reconstruction + discrimination)
  - **STGNN**: Spatio-Temporal GNN with MI-based dynamic graphs

- _Method_: 🔧

**Architecture**:
```
EEG → DRDCAE (latent z) → Graph (MI-based) → STGNN (Spatial GCN + Temporal GRU) → Classification
```

**Datasets**:
- BCI Competition IV-2a: 4 classes (left hand, right hand, feet, tongue)
- High Gamma Dataset (BCI-HGD): Binary classification

Baselines: EEGNet, ShallowConvNet, DeepConvNet, RGNN, ST-GCN, DAFNet

- _Result_: 📈**Performance**:

✅ **State-of-the-art** on BCI IV-2a (4-class)
✅ **State-of-the-art** on BCI-HGD (binary)
✅ **Cross-subject robustness**
✅ **Interpretable MI connectivity maps**

| Aspect | Traditional CNN/RNN | DRDCAE-STGNN |
|--------|-------------------|--------------|
| **Spatial** | Local convolution | **Global graph connectivity** ✅ |
| **Temporal** | Sequential RNN | **Hierarchical GRU + spatial context** ✅ |
| **Features** | Reconstruction OR classification | **Both simultaneously** ✅ |
| **Connectivity** | Fixed/ignored | **Dynamic MI-based** ✅ |
| **Interpretability** | Low | **High (MI maps, latent space)** ✅ |

- _Conclusion_:🎓 **DRDCAE-STGNN advances Motor Imagery BCI** through:
  - 🧠 **Dynamic brain connectivity modeling** via mutual information (adaptive, subject-specific)
  - 🎯 **Discriminative feature learning** with dual-objective autoencoder (reconstruction + separation)
  - 🔗 **Spatio-temporal integration** via hierarchical GNN (spatial GCN + temporal GRU)
  - 📈 **State-of-the-art performance** on multiple MI datasets
  - 🔍 **High interpretability** with functional connectivity insights
  - 🚀 **Robust generalization** across subjects and tasks
  
  Discriminative autoencoding + MI-based dynamic graphs = superior MI classification with mechanistic understanding. 🤖🧠

</details>
</details>

---

> [*Towards Generalizable Learning Models for EEG-Based Identification of Pain Perception*], [Aug 12, 2025]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 🧠 EEG-based pain perception recognition with cross-subject generalization
- _Author_: Mathis Rezzouk, Fabrice Gagnon, Alyson Champagne, Mathieu Roy, Philippe Albouy, Michel-Pierre Coll, Cem Subakan
- _Core Group_: McGill University, Concordia University
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 Brain-signal models struggle to recognize pain across different individuals because each person's EEG signal exhibits significant variability. Most existing systems only perform well for the specific person they were trained on.

- _Focus problem_: 🔍 How well can different machine learning and deep learning models identify pain when tested on previously unseen individuals (cross-subject generalization)?

- _Why important_:

| Benefit | Impact |
|---------|--------|
| **🏥 Objective pain assessment** | No need for subjective self-reporting |
| **⚡ No per-person retraining** | Deploy once, use for any patient |
| **👶 Critical patient populations** | Helps patients unable to communicate pain (dementia, coma, infants) |

> ❌ Without cross-person generalization, these models cannot be deployed in real clinical settings.  
> ✅ A robust model must recognize brain patterns of pain that are consistent across everyone.
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_:

📊 **Simple models** work great when training and testing on the same person but **fail badly on new people**.
🚀 **Deep learning models**, especially **graph-based ones**, handle new people **much better**.

| Model Type | Within-Person | Cross-Person | Generalization |
|------------|---------------|--------------|----------------|
| **SVM** | ~90-95% | ~45-50% ⚠️ | ❌ Poor (-45% drop) |
| **Logistic Regression** | ~85-90% | ~40-50% ⚠️ | ❌ Poor (-40% drop) |
| **Deep4Net** | ~92% | ~70-75% | ✅ Good (-20% drop) |
| **GGN (Graph)** | ~93% | **~75-80%** | ✅✅ **Best (-15% drop)** |

- _Why necessary_: 🏥 A robust model must recognize brain patterns of pain that are consistent across everyone. Current person-dependent models require calibration for each new patient, making clinical deployment impractical.
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 Compare many models—from simple classifiers (SVM, Logistic Regression) to advanced neural networks (Deep4Net, EEGNet, **Graph Neural Networks**)—to see which ones can best find shared patterns of pain perception across people.

- _Method_: 🔧 Collected EEG data from **108 subjects** exposed to **heat pain** and **unpleasant sounds**. Carefully preprocessed the data, trained several models, and tested performance in two scenarios:
  - **Subject-dependent**: Train & test on same person
  - **Subject-independent**: Train on some people, test on completely new people ⭐

- _Result_: 📈
  - **Classical ML** (SVM, LogReg): ~90-95% within-person → ~45-50% cross-person ❌ (massive failure)
  - **Deep4Net**: ~92% within-person → ~72% cross-person ✅ (-20% drop)
  - **GGN (Graph)**: ~93% within-person → **~78% cross-person** 🏆 (-15% drop, **best generalization**)

- _Conclusion_: 🎓 Deep learning—especially **graph-based models that model brain connections**—can capture pain-related brain activity that stays similar across different people. **Brain connectivity > individual channel patterns**. This makes them promising for future real-world, **person-independent pain monitoring systems** with ~78% zero-calibration accuracy. 🏥
</details>
</details>

---

> [*Graph Convolutional Neural Networks to Model the Brain for Insomnia*], [Jul 2, 2025]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 😴 EEG-based brain network modeling for insomnia detection and understanding altered brain connectivity during sleep
- _Author_: Kevin Monteiro, Sam Nallaperuma-Herzberg, Martina Mason, Steve Niederer
- _Group_: University of Cambridge
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 It's hard to understand **how insomnia changes the brain's activity during sleep** because brain signals are **complex, noisy, and vary a lot between people**.

- _Focus problem_: 🔍 How to use **EEG signals** to build a model of brain behavior in insomnia patients, identifying **which brain regions and signal patterns differ** from normal sleepers?

- _Why important_: 💡 Many people suffer from insomnia, and current treatments often have **side effects**. Better understanding of insomnia brain function may help:
  - 🏥 **Safer diagnosis** (non-invasive, objective)
  - 💊 **More effective treatments** (personalized, potentially non-drug)
  - 🧠 **Mechanistic insights** into sleep disorder neurobiology
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **Brain connections near motor, sensory, and auditory areas** play an important role in identifying insomnia:

**Key discoveries**:
- 🔑 **Critical connections**: C4-P4, F4-C4, C4-A1 (motor-sensory-auditory regions)
- ⏱️ **Optimal time window**: **50-second segments** capture meaningful brain activity patterns
- 📍 **Spatial distance matters**: Including electrode proximity improves accuracy

**Performance drops** when removing:
| Removed Connection | Brain Region | Impact |
|-------------------|--------------|--------|
| **C4-P4** | Motor-Sensory | Largest accuracy drop |
| **F4-C4** | Frontal-Motor | Significant drop |
| **C4-A1** | Motor-Auditory | Notable drop |

→ These regions are **frequently disturbed in insomnia**

- _Why necessary_: 🏥 Without understanding **neural patterns of insomnia**, it's difficult to:
  - Develop **personalized treatments**
  - Create **non-drug interventions**
  - Provide **objective diagnosis**
  
  Modeling the brain's network provides a clearer picture of **how insomnia affects brain communication**.
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 Build a **"brain network" from EEG data**:
  - **Nodes**: EEG channels (brain regions)
  - **Edges**: Connection strength (how brain areas interact)
  - **Task**: Train a **Graph Neural Network (GNN)** to classify insomnia vs. healthy brain patterns

- _Method_: 🔧

**Data collection**:
- **~13-hour continuous EEG recordings** per subject
- Both **insomnia patients** and **healthy controls**

**Pipeline**:
```
Continuous EEG → Time windows → Filtering → Brain graphs → GCN → Classification
```

**Graph construction**:
1. **Nodes**: EEG channels
2. **Node features**: Power in different **frequency bands** (δ, θ, α, β)
3. **Edge weights**: Combination of:
   - **Signal similarity** (functional connectivity)
   - **Spatial distance** (electrode proximity) ⭐

**Model**: Graph Convolutional Neural Network (GCN)

**Analysis windows**: Tested different durations (found **50 seconds optimal**)

- _Result_: 📈

**Critical brain connections** (removal causes largest accuracy drops):

| Connection | Region | Clinical Relevance |
|-----------|--------|-------------------|
| **C4-P4** 🏆 | Motor-Parietal-Sensory | Often disturbed in insomnia |
| **F4-C4** | Frontal-Motor | Arousal & motor control |
| **C4-A1** | Motor-Auditory | Sensory processing |

**Insights**:
- 🧠 Motor, sensory, and auditory areas show **altered connectivity** in insomnia
- ⏱️ 50-second window captures **optimal temporal dynamics**
- 📊 Graph-based modeling reveals **network-level disruptions** (not just individual channels)

- _Conclusion_: 🎓 **Graph-based deep learning reveals how insomnia changes brain communication patterns**:
  - 🔍 **Identifies key brain regions** (motor-sensory-auditory network)
  - ⏱️ **Establishes effective analysis strategy** (50-second windows, spatial+functional connectivity)
  - 🏥 **Lays groundwork for brain-based, non-invasive sleep diagnostics**
  - 🧠 **Network perspective** > individual channel analysis
  - 💡 **Potential for personalized, non-drug treatments** based on connectivity patterns
  
  Brain network modeling captures insomnia's disrupted neural communication, opening paths to objective diagnosis and targeted interventions. 😴🧠
</details>
</details>

---

> [*Transformer-based EEG Decoding- A Survey*], [Jul 3, 2025]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 🤖 Comprehensive review of Transformer models for EEG signal decoding across multiple brain-related tasks.
- _Author_: Haodong Zhang, Hongqi Li
- _Group_: Northwestern Poly-technical University
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 It's difficult for computers to **read and understand brain signals (EEG)** because the data are:
  - 📊 **Noisy**
  - 👥 **Vary across people**
  - ⚡ **Change quickly over time**
  
  Traditional models often **miss long-term relationships** between brain activities.

- _Focus problem_: 🔍 How have **Transformer models** (good at finding long-range patterns) been used to better **decode and interpret EEG signals** in different brain-related tasks?

- _Why important_: 💡 EEG decoding is used in many **real-world areas**:
  - 😴 **Sleep monitoring**
  - 😊 **Emotion recognition**
  - 🏥 **Disease detection** (epilepsy, dementia, etc.)
  - 🧠 **Brain-Computer Interfaces (BCI)**
  
  Improving EEG decoding could lead to:
  - ✅ Better health care
  - ✅ More natural brain-computer interfaces
  - ✅ Help people with disabilities communicate
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **Transformers understand both short and long patterns** in EEG data much better than older models:

**Key insights**:

| Model Type | Capability |
|------------|-----------|
| **Pure Transformers** | Capture long-range temporal dependencies |
| **Hybrid (Transformer + CNN)** | Local details + global context |
| **Hybrid (Transformer + GNN)** | Spatial connectivity + temporal patterns |
| **EEG-specific Transformers** | Tailored to brain signal characteristics |

**Performance highlights**:
- 😊 **Emotion recognition**: Up to **99% accuracy**
- 😴 **Sleep staging**: **84-85% accuracy**
- ⚡ **Epilepsy detection**: Strong results
- 🤖 **Motor imagery**: Robust classification

- _Why necessary_: 🏥 Traditional models have **critical limitations**:

| Model | Limitation |
|-------|-----------|
| **CNNs** | Only capture **local patterns** (miss global context) |
| **RNNs** | Limited to **short-term dependencies** (vanishing gradients) |
| **Both** | Can't handle brain's **wide and long-range connections** |

The brain works through **global, long-range connections** → Need models that handle this **global context** ✅
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 **Gather and organize existing research** on how Transformers are used for EEG decoding:
  - Compare different designs
  - Identify which ideas work best for different brain tasks
  - Provide systematic review and taxonomy

- _Method_: 🔧

**Review scope**:
- 📚 **Over 160 studies** applying Transformers to EEG analysis
- 🗂️ **Grouped into 3 model types**:

| Type | Description | Example |
|------|-------------|---------|
| **1. Basic Transformers** | Standard self-attention architecture | Vanilla Transformer, BERT-style |
| **2. Hybrid Models** | Transformer + CNN/RNN/GNN | Local features + global context |
| **3. EEG-specific Transformers** | Redesigned for brain signals | Custom attention, position encoding |

**Tasks reviewed**:
- 😊 Emotion recognition
- 😴 Sleep staging
- ⚡ Epilepsy detection
- 🤖 Motor imagery (movement imagination)

- _Result_: 📈

**Performance comparison**:

| Task | Best Accuracy | Observation |
|------|--------------|-------------|
| **Emotion recognition** 😊 | **Up to 99%** 🏆 | Transformers >> traditional models |
| **Sleep staging** 😴 | **84-85%** | Consistent improvement |
| **Epilepsy detection** ⚡ | Strong results | Reliable seizure prediction |
| **Motor imagery** 🤖 | Robust | Better BCI control |

**Advantages**:
- ✅ **Long-range dependencies**: Capture global brain dynamics
- ✅ **Self-attention**: Identify important brain regions automatically
- ✅ **Parallelization**: Faster training than RNNs
- ✅ **Hybrid designs**: Combine strengths of multiple architectures

**Challenges** ⚠️:
- ❌ **Data hungry**: Require lots of labeled EEG data
- ❌ **Computation cost**: High memory and processing requirements
- ❌ **Hard to interpret**: Black-box attention mechanisms
- ❌ **Overfitting risk**: Without sufficient data

**Hybrid model benefits**:

| Combination | Advantage |
|------------|-----------|
| **Transformer + CNN** | Local signal details + global context |
| **Transformer + GNN** | Brain connectivity + temporal patterns |
| **Transformer + RNN** | Sequential processing + long-range attention |

- _Conclusion_: 🎓 **Transformers are becoming the leading approach for understanding brain signals**:
  - 🏆 **Consistently outperform older models** (CNNs, RNNs) across all tasks
  - 🧠 **Capture global brain dynamics** that traditional models miss
  - 🔧 **Hybrid designs** (Transformer + CNN/GNN) work best for EEG
  - ⚠️ **Challenges remain**: Data shortage, high computation cost, low interpretability
  - 🚀 **Great promise** for building more **general, reliable, human-like** brain decoding systems
  
  Future directions:
  - 📊 **Self-supervised learning** to reduce data requirements
  - 🔍 **Interpretable attention** for clinical trust
  - ⚡ **Efficient architectures** for real-time BCI
  - 🌐 **Cross-subject transfer** for generalization
  
  Transformers + EEG = next generation of brain signal understanding. 🧠🤖
</details>
</details>

---

> [*EEG2GAIT- A Hierarchical Graph Convolutional Network for EEG-based Gait Decoding*], [Apr 2, 2025]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 🚶 EEG-based gait (walking) motion prediction using brain network modeling for brain-controlled prosthetics and rehabilitation
- _Author_: Xi Fu, Rui Liu, Aung Aung Phyo Wai, Hannah Pulferer, Neethu Robinson, Gernot R Mu ̈ller-Putz, Cuntai Guan
- _Group_: Nanyang Technological University, Graz University of Technology
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 It is very hard to **predict how the brain controls walking** by reading EEG signals because these signals are:
  - 📊 **Noisy**
  - ⚡ **Change quickly**
  - 🧠 **Involve complex interactions between many brain regions**

- _Focus problem_: 🔍 How to teach a computer model to understand how **patterns in EEG data relate to leg movements during walking**, using both:
  - ⏱️ **Timing** (temporal patterns)
  - 🔗 **Spatial connections** between different brain areas (brain network)

- _Why important_: 💡 Understanding how the brain coordinates walking could help:
  - 🏥 **Better rehabilitation tools** for people with movement disorders (stroke, Parkinson's)
  - 🦾 **Brain-controlled prosthetic legs** that move naturally
  - 🧠 **Neuroscience insights** into motor control mechanisms
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **EEG2GAIT** learns both **short-term and long-term brain activity patterns** and connects them to specific leg movements:

**Key discoveries**:
- 🏆 **Correlation ≈ 0.93** between predicted and real joint motion
- 📈 **10-15% better** than other models
- 🧠 **Central motor area signals** most important for predicting gait
- 🔗 **Brain network connections** critical for accurate prediction

**Most informative brain areas**:
| Location | Brain Region | Clinical Relevance |
|----------|--------------|-------------------|
| **Central scalp (C3, Cz, C4)** | Primary motor cortex | Controls voluntary leg movements |
| **Midline electrodes** | Supplementary motor area | Gait coordination & planning |

- _Why necessary_: 🏥 Older models have **critical limitations**:

| Approach | Problem |
|----------|---------|
| **Flat time sequences** | Ignore how brain regions **communicate** |
| **Single-channel analysis** | Miss **network-level coordination** |
| **Local patterns only** | Can't capture **global motor control** |

Walking is **not controlled by one spot** — it's a **network process** 🔗  
→ Understanding **spatial connections is essential** ✅
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 Represent the **brain as a network**:
  - **Nodes**: EEG electrodes (brain regions)
  - **Edges**: Connections showing how brain areas interact
  - **Architecture**: Hierarchical model learning patterns from **local → global levels**
  - **Output**: Link brain activity directly to **joint movements** (hip, knee, ankle)

- _Method_: 🔧

**Data collection**:
- 👥 **50 people** walking naturally
- 📊 **EEG signals** + **leg joint angles** (synchronized)
- 🦵 **6 joint angles**: Hip, knee, ankle (both legs)

**EEG2GAIT Architecture**:

```
EEG Input
    ↓
[1] Short-term Rhythm Module
    ↓
[2] Graph Network (spatial connections)
    ↓
[3] Temporal Module (long-term patterns)
    ↓
Joint Angle Prediction (hip, knee, ankle)
```

**Innovation**: **Time-frequency loss** ensures predictions match both:
- ⏱️ Temporal dynamics (phase of gait cycle)
- 🎵 Frequency content (rhythm of walking)

- _Result_: 📈

**Performance**:

| Metric | EEG2GAIT | Improvement |
|--------|----------|-------------|
| **Correlation** | **≈0.93** 🏆 | **10-15% better** than baselines |
| **Consistency** | ✅ Robust | Across different subjects |
| **Interpretability** | ✅ High | Identifies key brain regions |

**Key brain regions identified**:
- 🧠 **Central motor cortex** (C3, Cz, C4): Most predictive
- 🔗 **Midline areas**: Gait coordination
- 📊 Aligns with **known motor control neuroscience** ✅

**Comparison with baselines**:

| Model Type | Approach | Performance |
|------------|----------|-------------|
| **Traditional RNN** | Time sequences only | Baseline |
| **CNN-based** | Local patterns | Better than RNN |
| **EEG2GAIT** | **Network + Time-frequency** | **Best (10-15% ↑)** 🏆 |

**Advantages**:
- ✅ **Hierarchical learning**: Local → global patterns
- ✅ **Brain network modeling**: Captures spatial interactions
- ✅ **Time-frequency loss**: Better temporal accuracy
- ✅ **Interpretable**: Identifies critical brain regions
- ✅ **Generalizable**: Consistent across subjects

- _Conclusion_: 🎓 **EEG2GAIT "decodes" walking movements from EEG** more accurately and clearly than before:
  - 🏆 **≈0.93 correlation**, 10-15% better than other models
  - 🧠 **Brain network modeling** > flat time sequences (captures spatial interactions)
  - ⏱️ **Time-frequency learning** captures both rhythm and phase
  - 🔍 **Identifies motor cortex** as key region (validates neuroscience)
  - 🚀 **Step toward practical brain-controlled walking systems**
  - 🏥 **Deeper understanding** of how brain organizes movement
  
  Applications:
  - 🦾 Brain-controlled prosthetic legs
  - 🏥 Rehabilitation for stroke/Parkinson's patients
  - 🧠 Neuroscience research on motor control
  - 📊 Gait analysis for clinical diagnosis
  
  Brain network + time-frequency modeling = accurate gait decoding from noisy EEG signals. 🚶🧠
</details>
</details>

---

> [*Flexible and Explainable Graph Analysis for EEG-based Alzheimer’s Disease Classification*], [Apr 2, 2025]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 🧠 Early Alzheimer's disease detection from EEG using learnable brain networks with explainability
- _Author_: Jing Wang, Jun-En Ding, Feng Liu, Elisa Kallioniemi, Shuqiang Wang, Wen-Xiang Tsai, Albert C. Yang
- _Group_: Stevens Institute of Technology, New Jersey Institute of Technology, National Yang-Ming Chiao Tung University, Chinese Academy of Sciences
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 It's difficult to **detect Alzheimer's disease early** using EEG signals because:
  - 🧠 Brain activity is **complex**
  - ⚡ **Changes over time**
  - 👥 **Varies from person to person**
  
  Most existing computer models can **classify patients** but **can't explain why or how** they make decisions.

- _Focus problem_: 🔍 How to build a model that **both**:
  - ✅ Identifies Alzheimer's disease from EEG data
  - ✅ **Explains which brain regions and connections** are most affected

- _Why important_: 💡 Early and reliable detection of Alzheimer's is critical for treatment:
  - 🏥 Current medical scans (MRI, PET) are **expensive and slow**
  - 📊 EEG offers a **cheaper and faster** alternative
  - 🔍 If a model can read it **accurately and transparently** → more **accessible early screening**
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **Letting the computer learn how brain regions connect** (instead of fixing connections in advance) leads to:
  - 📈 **Higher accuracy**
  - 🔍 **Better understanding** of which brain parts are disrupted in Alzheimer's

**Key brain regions identified**:
| Brain Region | Role | Alzheimer's Impact |
|--------------|------|-------------------|
| **Frontal** | Executive function, memory | Most affected |
| **Temporal** | Memory processing | Severely disrupted |
| **Parietal** | Spatial processing, attention | Significantly impacted |

→ Results **match neuroscience knowledge** ✅

- _Why necessary_: 🏥 Most earlier models have **critical limitations**:

| Approach | Problem |
|----------|---------|
| **Fixed connections** | Assume static relationships between EEG channels |
| **Ignore disease changes** | Can't capture how brain connections **change in disease** |
| **Not person-specific** | Miss **dynamic and individual patterns** |

A **flexible, learnable structure** is necessary to capture these patterns ✅
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 Represent the **brain as a network**:
  - **Nodes**: EEG channels (brain regions)
  - **Edges**: Show how different areas interact
  - **Train a model** that learns both local and global brain patterns
  - **Produce attention scores** showing which nodes and links matter most for Alzheimer's detection

- _Method_: 🔧

**Model: FEGL (Flexible Explainable Graph Learning)**

**Architecture components**:

| Component | Function |
|-----------|----------|
| **Learnable brain network** | Learns connections based on **EEG signal similarity** (not fixed) |
| **Multiple layers** | Capture both **small-scale and large-scale** brain activity |
| **Explanation module** | Highlights brain regions & connections responsible for decision |

**Pipeline**:
```
EEG Data → Learn Brain Network → Multi-layer Graph Processing → Classification + Explanation
```

**Data**: EEG from Alzheimer's patients and healthy people

- _Result_: 📈

**Performance**:

| Model | Accuracy | Explainability |
|-------|----------|----------------|
| **SVM** | Lower | ❌ No |
| **CNN** | Lower | ❌ No |
| **Traditional GNN** | Lower | ❌ Limited |
| **FEGL** | **~89%** 🏆 | ✅ **High** |

**Advantages**:

| Feature | FEGL | Traditional Models |
|---------|------|-------------------|
| **Brain connections** | **Learnable** ✅ | Fixed ❌ |
| **Accuracy** | **~89%** 🏆 | Lower |
| **Explainability** | **High** ✅ | Low/None ❌ |
| **Disease-specific patterns** | **Captures** ✅ | Misses ❌ |

- _Conclusion_: 🎓 By combining **flexibility and explainability**, FEGL provides:
  - 📈 **High accuracy** (~89%)
  - 🔍 **Clear insight** into how Alzheimer's alters brain connectivity
  - 🧠 **Identifies affected regions** (frontal, temporal, parietal)
  - 💰 **Practical, low-cost tool** for early Alzheimer's detection (EEG vs. MRI/PET)
  - 🏥 **Understanding** of how disease changes brain communication patterns
  
  Flexible learnable networks + explainability = accurate and interpretable Alzheimer's detection from EEG. 🧠💡
</details>
</details>

---

> [*Geometric Machine Learning on EEG Signals*], [Feb 27, 2025]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 📐 Discovering hidden geometric structure in EEG signals for accurate brain-computer interface decoding
- _Author_: Benjamin J. Choi
- _Group_: Kempner Institute at Harvard University
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 It's very hard to **understand brain activity from EEG signals** because they are:
  - 📊 **Noisy**
  - 📈 **High-dimensional**
  - ⚡ **Change quickly**
  
  This makes it difficult for computers to **accurately detect what a person is thinking or doing** from raw EEG data.

- _Focus problem_: 🔍 How to find the **hidden structure inside EEG signals** — their **"shape" or geometry** — so machines can learn how **different thoughts or mental states are organized** in the brain?

- _Why important_: 💡 If we can discover these **geometric patterns** in brain signals:
  - 🤖 **More accurate** brain-computer interfaces
  - ⚡ **Faster** BCI systems
  - 📊 **Easier to train** (less data needed)
  
  Applications:
  - 🗣️ **Communication** for people with paralysis
  - 🏥 **Early diagnosis** of neurological diseases
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **EEG signals live on a lower-dimensional geometric space**:
  - Although EEG data **looks messy**, it follows a **consistent underlying pattern**
  - By **cleaning signals** and **learning this geometry**, computers can **separate mental states almost perfectly**

**Key insight**:
| Traditional View | Geometric View |
|-----------------|----------------|
| EEG = messy high-D data | EEG = clean low-D manifold 📐 |
| Treat as flat numbers | Discover curved structure ✅ |
| Hard to classify | Nearly perfect separation 🏆 |

- _Why necessary_: 🏥 Traditional models have **critical limitations**:

| Approach | Problem |
|----------|---------|
| **Flat numbers** | Ignore how brain regions **interact** |
| **Simple time series** | Miss **network relationships** |
| **Channel-by-channel** | Individual signals less meaningful |

Brain activity is better understood as a **network with curved geometric structure** 📐  
→ **Relationships between regions** carry more meaning than individual signals ✅
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 Combine **denoising + frequency analysis + geometric learning** into one pipeline:

| Step | Purpose |
|------|---------|
| **1. Clean EEG** | Lightweight Transformer model (AT-AT) removes noise |
| **2. Build network** | Capture how brain regions connect |
| **3. Reveal geometry** | Smooth and reshape to show true geometric form |
| **4. Classify** | Small neural network predicts different thoughts |

- _Method_: 🔧

**Pipeline**:
```
Raw EEG → AT-AT Denoising → Graph Construction → Geometric Processing → Classification
```

**Detailed steps**:

| Step | Description |
|------|-------------|
| **Denoising (AT-AT)** | Remove noise like muscle artifacts |
| **Graph construction** | Nodes = EEG channels, Edges = relationships |
| **Geometric smoothing** | Mathematical steps: frequency transform, neighborhood mapping, smoothing |
| **Compact representation** | Build low-dimensional manifold of brain activity |
| **Graph network** | Learn patterns in geometric space |

**Task**: Predict what person was imagining (e.g., thinking of a number or not)

- _Result_: 📈

**Performance**:

| Metric | Result |
|--------|--------|
| **Accuracy** | **~97%** 🏆 |
| **Data requirement** | Works with **small datasets** ✅ |
| **Key information source** | **Brain region connections + curvature** (not individual channels) |
| **Denoising impact** | **Greatly improved** pattern clarity |

**Key findings**:
- 🏆 **~97% accuracy** in distinguishing mental states
- 📐 Most useful info from **how brain regions connect and curve together**
- 🔍 Individual EEG channels alone → **less meaningful**
- ✨ Denoising step → **critical for clarity**

**Advantages**:

| Feature | Geometric Approach | Traditional Approach |
|---------|-------------------|---------------------|
| **Data view** | **Curved manifold** 📐 | Flat vectors |
| **Accuracy** | **~97%** 🏆 | Lower |
| **Data efficiency** | **Small datasets** ✅ | Requires large data |
| **Interpretability** | **Geometric structure** ✅ | Black box ❌ |
| **Brain relationships** | **Captured** ✅ | Ignored ❌ |

- _Conclusion_: 🎓 **Brain signals have hidden geometric structure that machines can learn**:
  - 📐 **EEG lives on lower-dimensional manifold** (not messy high-D space)
  - 🧠 Viewing EEG as **curved network** (not raw data) → decode thoughts more clearly
  - 🏆 **~97% accuracy** even with small datasets
  - 🔍 **Relationships + geometry** > individual channel values
  - ✨ **Denoising critical** for revealing geometric patterns
  - 🚀 Foundation for future BCIs that are both **accurate and interpretable**
  
  Applications:
  - 🤖 High-accuracy brain-computer interfaces
  - 🗣️ Communication systems for paralysis
  - 🏥 Neurological disease diagnosis
  - 📊 Understanding brain organization
  
  Geometric manifold learning = unlocking hidden structure in noisy EEG signals. 📐🧠
</details>
</details>

---

> [*Subject Representation Learning from EEG using Graph Convolutional Variational Autoencoders (GC-VASE)*], [Jan 13, 2025]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 🧬 Disentangling personal brain signatures from task information in EEG for personalized systems and biometric identification
- _Core Author_: Aditya Mishra, Ahnaf Mozib Samin, Ali Etemad, Javad Hashemi
- _Core Group_: Queen’s University
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 EEG signals:
  - 👥 **Vary widely between people**
  - 📊 **Contain a lot of random noise**
  
  Most models **cannot clearly capture** what makes one person's brain activity unique. They **mix up personal traits** with other unrelated information.

- _Focus problem_: 🔍 How to **separate** the **"personal features"** of brain signals from other **task-related or noisy information**, so computers can understand what **truly defines each individual's EEG pattern**?

- _Why important_: 💡 Learning each person's brain signature enables:
  - 🤖 **Personalized brain-computer systems**
  - 🏥 **Medical tools that adapt to patients**
  - 🔐 **Secure biometric identification**
  - 🧠 Technology that **recognizes individual differences** in human brain
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 By **splitting the model's internal feature space** into two parts:
  - 🧬 **Personal information** (who)
  - 🎯 **Everything else** (task, noise)
  
  → The system can **recognize individuals much more accurately**. This separation makes the model both **cleaner and more general**.

**Key insight**:

| Space | Content | Purpose |
|-------|---------|---------|
| **Subject space** | Personal identity features 🧬 | "Who the person is" |
| **Residual space** | Task + noise + variations | "What's happening" |

- _Why necessary_: 🏥 Traditional models have **critical problems**:

| Traditional Approach | Problem |
|---------------------|---------|
| **Learn everything in one space** | Confusion between who vs. what |
| **No separation** | Unstable with new people/recordings |
| **Mixed representations** | Can't isolate personal traits |

**Separating** these two types of information → model more **stable** when seeing **new people or new EEG recordings** ✅
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 Represent **EEG data as a network** of connected brain regions and train a model that:
  - Encodes signals into **hidden "latent space"**
  - **Divides latent space** into two parts:
    - 🧬 **Personal identity features**
    - 🎯 **Other unrelated variations**
  - Uses **contrastive training** to:
    - ✅ Keep signals from **same person close together**
    - ❌ Push signals from **different people far apart**

- _Method_: 🔧

**Model: GC-VASE (Graph Contrastive Variational AutoEncoder for Subject-specific learning)**

**Architecture**:

| Component | Function |
|-----------|----------|
| **Graph network** | EEG channels as nodes connected by relationships |
| **Variational autoencoder** | Compress EEG into smaller internal space |
| **Split latent space** | "Subject" part + "Residual" part |
| **Contrastive learning** | Strengthen grouping of same-person data |
| **Adapter layer** | Quick adjustment to new users (no full retraining) |

**Pipeline**:
```
EEG Network → Compress to Latent Space → [Subject Part | Residual Part] → Contrastive Training
```

**Testing**: Two large EEG datasets covering multiple brain tasks

- _Result_: 📈

**Performance**:

| Metric | GC-VASE | Previous Methods |
|--------|---------|------------------|
| **Same dataset accuracy** | **~90%** 🏆 | ~80% |
| **Improvement** | **+10%** | Baseline |
| **Different dataset accuracy** | **~70%** 🏆 | Lower |
| **Generalization** | ✅ Strong | ❌ Weak |

**Advantages**:

| Feature | GC-VASE | Traditional Models |
|---------|---------|-------------------|
| **Feature separation** | **Personal + Non-personal split** ✅ | Mixed together ❌ |
| **Accuracy** | **~90%** 🏆 | ~80% |
| **Cross-dataset** | **~70%** ✅ | Poor ❌ |
| **New user adaptation** | **Fast (adapter)** ✅ | Slow retraining ❌ |
| **Interpretability** | **Clear separation** ✅ | Unclear ❌ |

- _Conclusion_: 🎓 By teaching the model to **separate who from what**, the study creates a **clearer and more flexible way** to learn individual brain signatures:
  - 🏆 **~90% accuracy** (+10% improvement)
  - 🧬 **Successful disentanglement** of personal vs. task features
  - 🌐 **Strong generalization** (~70% on different dataset)
  - 📊 **Visual confirmation** of feature separation
  - ⚡ **Quick adaptation** to new users
  - 🚀 **Basis for future**:
    - 🤖 Personalized brain-computer systems
    - 🏥 Mental health tools (individual baselines)
    - 🔐 EEG-based identification technologies
  
  Disentangled graph contrastive learning = clearer, more flexible individual brain signature extraction. 🧬🧠
</details>
</details>

---

> [*Quantum Cognition-Inspired EEG-based Recommendation via Graph Neural Networks*], [Jan 5, 2025]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 🧠 Brain-signal-driven recommendation system using quantum cognition theory and graph neural networks
- _Author_: Jinkun Han, Wei Li, Yingshu Li, Zhipeng Cai
- _Group_: Georgia State University
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 Traditional recommendation systems can only **guess what users might like** based on:
  - 📊 **Past clicks or ratings**
  - ❌ **No way of knowing** what people actually **think or feel in real time**
  
  This gap makes recommendation systems **slow to adapt** to users' **changing moods and interests**.

- _Focus problem_: 🔍 How to build a recommendation system that can **directly understand a person's current thoughts** from their **brain signals (EEG)**, rather than relying on **past behavior or preferences**?

- _Why important_: 💡 People's preferences **change constantly**:
  - 🛍️ What you want to see/buy **now ≠ what you liked yesterday**
  - ⚡ Real-time mental state capture → **instant, more personal recommendations**
  
  Applications:
  - 🛒 Shopping
  - 📺 Media
  - 🎮 Gaming
  - 🏥 Healthcare
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **QUARK** links **how humans think** with **how computers recommend**:
  - 🧠 **EEG signals reveal meaningful patterns** about current interests
  - ⚛️ **Quantum cognition theory**: Model mixed thoughts
  - 🔗 **Graph neural networks**: Capture relationships between thoughts
  
  → System recommends items **more accurately** than existing models

**Key insight**:

| Component | Function |
|-----------|----------|
| **Quantum modeling** | Breaks down "thought mixtures" into clearer parts ⚛️ |
| **Graph learning** | Connects related ideas in the brain 🔗 |
| **EEG patterns** | Reveal real-time interests 🧠 |

- _Why necessary_: 🏥 EEG signals are **messy and contain overlapping traces** of different thoughts:

| Challenge | Problem | Solution Needed |
|-----------|---------|-----------------|
| **Thought mixtures** | Standard ML can't separate | **Quantum approach** ⚛️ |
| **Temporal dependencies** | How past thoughts influence new ones | **Graph learning** 🔗 |
| **Real-time adaptation** | Can't capture current mental state | **EEG-based modeling** 🧠 |

Quantum-based approach + graph learning = break down thought mixtures & connect related ideas ✅
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 Simulate **how people's thoughts evolve**:
  - 📜 **Past ideas** affect **current decisions**
  - ⏩ **Current ideas** influence **future ones**
  
  **Pipeline**:
  1. Divide EEG into **small time segments** (thought events)
  2. Apply **quantum theory** to extract different mental components
  3. Use **graph neural networks** to connect and interpret thoughts
  4. **Predict what the user wants**

- _Method_: 🔧

**Model: QUARK (Quantum Cognition-Inspired EEG-based Recommendation)**

**Four main components**:

| Component | Function |
|-----------|----------|
| **1. Sliding Window Segmentation** | Split EEG into short "thought snapshots" |
| **2. Quantum Space Modeling** | Understand how mixed thoughts form and interfere ⚛️ |
| **3. Graph Neural Networks (GCN)** | Two networks: <br>• **Continuity graph**: Thought flow over time<br>• **Interference graph**: Past → Future influence |
| **4. Recommendation Generation** | Combine graphs → EEG user representation → Match to items |


**Dataset**: MindBigData EEG dataset (participants viewing images with brain signals recorded)

- _Result_: 📈

**Performance vs. Baselines**:

| Model | Type | Performance |
|-------|------|-------------|
| **DeepFM** | Traditional | Baseline |
| **NCF** | Traditional | Baseline |
| **BPR** | Traditional | Baseline |
| **QUARK** | EEG + Quantum + Graph | **Up to 95% improvement** 🏆 |

**Detailed findings**:

| Metric | Improvement |
|--------|-------------|
| **Precision** | Up to **95% higher** 🏆 |
| **Recall** | Up to **95% higher** 🏆 |
| **Thought clustering** | ✅ Clear categories |
| **Emotion detection** | ✅ Successful |
| **Style matching** | ✅ Accurate |


- _Conclusion_: 🎓 **"Recommend what you think"** using EEG is possible:
  - ⚛️ **Quantum-inspired modeling**: Separates mixed thoughts
  - 🔗 **Graph learning**: Captures thought continuity and interference
  - 🧠 **Raw brain signals → meaningful recommendations**
  - 🏆 **Up to 95% improvement** over traditional systems
  - ⚡ **Real-time mental state** understanding
  - 🚀 **Opens door to**:
    - 🤖 Adaptive recommendation systems
    - 😊 Emotion-aware personalization
    - 🧠 Brain-driven interfaces
    - 🛍️ Instant preference capture
  
  Applications:
  - 🛒 Shopping (instant preference detection)
  - 📺 Media streaming (mood-based content)
  - 🎮 Gaming (adaptive difficulty/content)
  - 🏥 Healthcare (mental state monitoring)
  
  Quantum cognition + graph neural networks = new generation of brain-driven personalization. 🧠⚛️🔗
</details>
</details>

---

> [*GEFM: Graph-Enhanced EEG Foundation Model*], [Feb 22, 2025]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 🧠 EEG foundation model combining spatial (graph) and temporal learning for general-purpose brain signal analysis
- _Core Author_: Limin Wang, Toyotaro Suzumura, Hiroki Kanezashi
- _Core Group_: The University of Tokyo
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 Most existing EEG foundation models only focus on **time sequence** of brain signals (how they change over time) but **ignore how different brain regions interact** with each other → Limited understanding of brain network activity and poorer performance across tasks.

- _Focus problem_: 🔍 How to build a foundation model that learns **both timing and relationships** between EEG channels (how different parts of the brain communicate)?

- _Why important_: 💡 EEG data contain valuable information in:
  - ⏱️ **When** signals occur (temporal)
  - 🔗 **How** brain regions connect and interact (spatial)
  
  These interactions are critical for diagnosing brain disorders and understanding cognition. **Ignoring them wastes** a large part of useful EEG information.
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 By adding **graph learning** (models connections between EEG channels) to **masked autoencoder framework**, **GEFM (Graph-Enhanced EEG Foundation Model)** captures both temporal flow and spatial relationships → Performs **better on all tested tasks** than previous models like BENDR.

- _Why necessary_: 🏥 **Labeling EEG data is expensive and time-consuming** → Models that learn from **large amounts of unlabeled data** are essential. Foundation models solve this, but previous ones **ignored inter-channel connections**. GEFM fills this gap using **graph neural networks (GNNs)** to represent and learn these relationships.
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 Represent the **brain as a network**:
  - **Nodes**: EEG channels
  - **Edges**: Weighted by electrode distances on scalp
  
  Feed network into **GNN** (learn spatial relationships) + **Transformer encoder** (learn temporal changes)

- _Method_: 🔧 **GEFM** builds on BENDR:

| Component | Description |
|-----------|-------------|
| **Two-layer GNN** | Learn how EEG channels interact (spatial) |
| **Node features** | Each channel's signal |
| **Edge weights** | Distances between electrodes |
| **Sequence adjustment** | Padding or linear scaling for different EEG lengths |
| **Transformer encoder** | Learn temporal dynamics (BENDR-based) |

**GNN variants tested**: GCN, GAT, GraphSAGE

**Pre-training**: Temple University Hospital EEG Corpus (10,000 subjects)

**Downstream tasks**:
- **MMI**: Imagined hand movement
- **P300**: Visual attention
- **ERN**: Error recognition

- _Result_: 📈 **Best version** (GEFM with GCN + edge weights) outperformed all baselines:

| Task | Improvement |
|------|-------------|
| **MMI** | **+31%** accuracy 🏆 |
| **P300** | **+8%** AUROC 🏆 |
| **ERN** | **+3%** accuracy 🏆 |

**Technical findings**:
- ✅ **Linear transformation** > simple padding for length adjustment
- ✅ **BENDR configuration** > simple linear models (handles extra spatial info from graph)

- _Conclusion_: 🎓 **GEFM is the first EEG foundation model** learning both spatial and temporal information effectively. By combining **graph neural networks + masked autoencoding**, it captures structure and timing of brain activity better than earlier models. Sets **new direction** for general-purpose EEG analysis → enabling accurate, flexible, low-cost applications in:
  - 🏥 Clinical diagnosis
  - 🤖 Brain-computer interfaces
  - 🧠 Cognitive research
</details>
</details>

---

> [*Pre-Training Graph Contrastive Masked Autoencoders are Strong Distillers for EEG*], [Jul 8, 2025]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 🧠 Self-supervised learning and knowledge distillation for transferring high-density (HD) to low-density (LD) EEG analysis
- _Author_: XinxuWei, KanhaoZhao, YongJiao, HuaXie, LifangHe, YuZhang
- _Group_: Lehigh University
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 EEG data are often **difficult to label** and come from devices with **very different numbers of electrodes**:

| EEG Type | Advantages | Disadvantages |
|----------|-----------|---------------|
| **High-density (HD)** | Detailed brain signals | 💰 Expensive, hard to use |
| **Low-density (LD)** | 💰 Cheaper, practical | ❌ Loses useful information |

Challenge: How to train models using **both unlabeled and labeled data** and **transfer knowledge** from HD to LD effectively?

- _Focus problem_: 🔍 How to bridge the gap between HD and LD EEG by:
  - Pre-training large graph models on **massive unlabeled EEG data**
  - **Distilling knowledge** into smaller models for simpler EEG setups

- _Why important_: 💡 In real applications (diagnosing depression, autism):
  - 🏥 **Fewer electrodes** = more practical and affordable
  - ❌ But they **perform poorly**
  
  If small, cheap models can **learn from large, high-quality ones** → EEG-based healthcare tools become **more accessible and reliable worldwide** ✅
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **EEG-DisGCMAE** combines:
  - **Two kinds of pre-training**: Contrastive + Generative
  - **On EEG graphs**: Learn brain-network features from unlabeled data
  - **Knowledge distillation**: Teacher (HD EEG) → Student (LD EEG)
  
  Result: Smaller models become **nearly as good as large ones** 🏆

**Framework components**:

| Component | Function |
|-----------|----------|
| **Contrastive learning** | Compare augmented versions of same signal |
| **Generative learning** | Reconstruct missing nodes (masked autoencoder) |
| **Knowledge distillation** | HD teacher → LD student transfer |

- _Why necessary_: 🏥 Existing EEG methods have **critical limitations**:

| Limitation | Problem |
|------------|---------|
| **Rely on labeled data** | Scarce and expensive |
| **Single density only** | Work with HD or LD, not both |
| **No transfer learning** | Can't transfer HD → LD knowledge |
| **Separate learning** | Can't combine contrastive + generative |

New method necessary to make EEG analysis **both powerful and practical** ✅
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 Represent **EEG signals as a graph**:
  - **Nodes**: Electrodes
  - **Edges**: Correlations between channels
  
  **Two models trained together**:
  - 👨‍🏫 **Teacher** (large, HD EEG)
  - 👨‍🎓 **Student** (small, LD EEG)
  
  Student learns **structural and semantic information** from teacher via **Graph Topology Distillation (GTD)** loss

- _Method_: 🔧

**Model: EEG-DisGCMAE**

**Four main steps**:

| Step | Description |
|------|-------------|
| **1. Graph construction** | EEG channels → nodes; correlations → edges |
| **2. Unified pre-training (GCMAE-PT)** | Contrastive + Generative: reconstruct masked signals while contrasting representations |
| **3. Knowledge distillation** | Student (LD) mimics teacher (HD) using:<br>• Logit alignment<br>• **GTD loss** (learns spatial connections even with missing electrodes) |
| **4. Fine-tuning** | Adapt to classification tasks (depression, autism) with limited labeled data |

**Pipeline**:
```
Unlabeled EEG → Graph Construction → GCMAE-PT (Contrastive+Generative) → Distillation (HD→LD) → Fine-tuning
```

**Datasets**:
- **EMBARC**: Clinical EEG for depression
- **HBN**: EEG for autism
- **Four classification tasks** total

**Baselines**:
- EEGNet
- GraphCL
- GraphMAE
- GPT-GNN

- _Result_: 📈

**Performance**:

| Finding | Result |
|---------|--------|
| **vs. Baselines** | **Outperformed all** (EEGNet, GraphCL, GraphMAE, GPT-GNN) 🏆 |
| **LD Student performance** | **Close to or better than HD models** 🏆 |
| **Robustness** | Under noise/missing electrodes: **much less accuracy loss** ✅ |
| **Datasets** | EMBARC + HBN (4 tasks) |

**Key advantages**:

| Feature | Traditional Methods | EEG-DisGCMAE |
|---------|-------------------|--------------|
| **HD → LD transfer** | ❌ Not possible | ✅ **Effective** |
| **Pre-training** | Single type | ✅ **Contrastive + Generative** |
| **Data requirement** | Large labeled datasets | ✅ **Unlabeled data** |
| **LD model performance** | Poor | ✅ **Near HD-level** |
| **Robustness** | Low | ✅ **High** (noise/missing) |
| **Cost** | Expensive HD required | ✅ **Affordable LD works** |

**Technical innovations**:

| Innovation | Benefit |
|------------|---------|
| **GCMAE-PT** | Unified contrastive + generative learning |
| **GTD loss** | Learns spatial connections despite missing electrodes |
| **Teacher-Student** | HD knowledge → LD model |

- _Conclusion_: 🎓 **Combining contrastive and generative graph pre-training** with **topology-aware distillation** provides powerful EEG training:
  - 🏆 **Outperforms all baselines** on clinical datasets (EMBARC, HBN)
  - 💰 **Small, affordable LD systems** reach **accuracy of expensive HD ones**
  - 🔍 **Learns from unlabeled data** (addresses labeling scarcity)
  - 💪 **Robust** to noise and missing electrodes
  - 🌍 Makes **advanced brain-signal analysis practical** for everyday clinical and research use
  
  Applications:
  - 🏥 Clinical diagnosis (depression, autism) with affordable devices
  - 🌍 Accessible healthcare in low-resource settings
  - 🧠 Research with practical EEG setups
  - 💰 Cost-effective brain monitoring
  
  Graph self-supervised learning + knowledge distillation = bridging HD-LD gap in EEG analysis. 🧠💡
</details>
</details>

---

> [*Graph Adapter for Parameter-Efficient Fine-Tuning of EEG Foundation Models*], [Feb 18, 2025]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: ⚡ Parameter-efficient fine-tuning of EEG foundation models using lightweight graph adapters for spatial learning
- _Author_: Toyotaro Suzumura, Hiroki Kanezashi, Shotaro Akahori
- _Group_: The University of Tokyo
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 Fine-tuning large EEG foundation models (Transformer-based) for specific medical tasks is **very expensive** in:
  - 💻 **Computing**
  - 📊 **Data requirements**
  
  These models learn **temporal features well** but **ignore spatial information** (how different EEG sensors relate). Fully retraining for every task is **unrealistic** with limited, costly-to-label medical EEG data.

- _Focus problem_: 🔍 How to **efficiently adapt** a pre-trained EEG model (BENDR) to new healthcare tasks **without retraining the whole model**? Proposes adding **lightweight graph adapter** to learn spatial relationships while **keeping original model frozen**.

- _Why important_: 💡 In brain disorder prediction (depression, epilepsy):
  - 🧠 **How brain regions interact** = as important as signal timing
  - 📊 Collecting labeled EEG data is **difficult**
  - Need to **reuse existing large models** without wasting computation or overfitting small datasets
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **EEG-GraphAdapter (EGA)** successfully adds **spatial understanding** to pre-trained time-series models:
  - 🔗 Uses simple **Graph Neural Network (GNN)** as adapter
  - 🧠 Captures how brain sensors connect
  - 📈 Improves prediction accuracy
  - ⚡ Reduces trainable parameters by **~80%**

**Performance gains**:

| Task | Improvement |
|------|-------------|
| **MDD classification** | **+12.8%** F1-score 🏆 |
| **TUAB abnormality detection** | **+16.1%** improvement 🏆 |
| **Parameter reduction** | 6.46M → ~1M (**~80% reduction**) ⚡ |

- _Why necessary_: 🏥 Traditional EEG foundation models (BENDR, MAEEG) have **critical limitations**:

| Limitation | Problem |
|------------|---------|
| **Only model time** | "See" each EEG channel separately |
| **Ignore spatial relationships** | Miss how brain areas connect |
| **Neurological disorders** | Often linked to **abnormal connections** (e.g., frontal-parietal) |
| **Full retraining** | Expensive, impractical for each task |

Model must learn **graph structure of brain** → adapter provides this ✅
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 Keep **large pre-trained BENDR frozen**, plug in **small GNN-based module** (GraphAdapter):
  - **Nodes**: EEG channels
  - **Edges**: Weighted by electrode distances on scalp
  - **Adapter**: Processes signals to include info from nearby sensors
  
  System learns:
  - ⏱️ **Temporal patterns** (from BENDR)
  - 🔗 **Spatial relationships** (from adapter)
  
  **Without retraining** entire network ✅

- _Method_: 🔧

**Model: EEG-GraphAdapter (EGA)**

**Pipeline**:

| Step | Description |
|------|-------------|
| **1. Pre-training** | BENDR trained on massive EEG dataset (69,000+ samples, Temple University Hospital) |
| **2. Adapter module** | Insert **two-layer GNN** (GCN/GraphSAGE/GAT) before BENDR |
| **3. Freeze backbone** | BENDR parameters remain **fixed** ❄️ |
| **4. Train adapter** | Only adapter + final classifier trained |
| **5. Downstream tasks** | Evaluate on clinical datasets |

**GNN variants tested**:
- **GCN** (Graph Convolutional Network)
- **GraphSAGE**
- **GAT** (Graph Attention Network)

**Downstream tasks**:

| Dataset | Task |
|---------|------|
| **MDD** | Major Depressive Disorder detection |
| **TUAB** | EEG abnormality detection |

**Metrics**: F1-score, AUROC

- _Result_: 📈

**Performance improvements**:

| Task | Best Adapter | Improvement | Baseline (BENDR) |
|------|-------------|-------------|------------------|
| **MDD** | EGA-GAT | **+12.8%** F1-score 🏆 | Baseline |
| **TUAB** | EGA-GraphSAGE | **+16.1%** 🏆 | Baseline |

**Efficiency gains**:

| Metric | Full Model | EGA |
|--------|-----------|-----|
| **Trainable parameters** | 6.46M | **~1M** ⚡ |
| **Reduction** | 100% | **~80% reduction** |
| **Computation** | High | **Up to 75% cut** |
| **Training speed** | Slower | **Faster** ✅ |
| **Accuracy** | Lower | **Higher** 🏆 |

**Key advantages**:

| Feature | Traditional Fine-tuning | EGA |
|---------|------------------------|-----|
| **Parameters trained** | All (millions) | **Adapter only (~1M)** ⚡ |
| **Spatial learning** | ❌ None | ✅ **GNN captures** |
| **Temporal learning** | ✅ Yes | ✅ **Preserved (BENDR)** |
| **Data requirement** | Large | **Small datasets work** ✅ |
| **Computation** | Expensive | **75% reduction** ⚡ |
| **Overfitting risk** | High | **Low** ✅ |

- _Conclusion_: 🎓 **EEG-GraphAdapter (EGA) is a parameter-efficient fine-tuning method** allowing existing EEG foundation models to learn **spatial brain relationships** without retraining:
  - 🏆 **+12.8% (MDD)**, **+16.1% (TUAB)** improvements
  - ⚡ **~80% parameter reduction** (6.46M → ~1M)
  - 💻 **Up to 75% computation cut**
  - 📊 **High accuracy with far less data and compute**
  - 🏥 **Ideal for healthcare** where labeled EEG data are scarce
  - 🔗 **Bridges gap** between time-based and space-aware EEG modeling
  - 🚀 **Paves way for scalable, clinically useful** EEG analysis tools
  
  Applications:
  - 🧠 Depression detection
  - ⚡ Epilepsy diagnosis
  - 🎯 ADHD assessment
  - 🏥 General brain disorder screening
  
  Lightweight graph adapter = adding spatial awareness to temporal models without expensive retraining. 🧠🔗⚡
</details>
</details>

---

> [*BrainGPT- Unleashing the Potential of EEG Generalist Foundation Model by Autoregressive Pre-training*], [Aug 29, 2025]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 🤖 First GPT-like generalist foundation model for EEG across multiple datasets, devices, and tasks
- _Author_: Tongtian Yue, Xuange Gao, Shuning Xue, Yepeng Tang, Longteng Guo, Jie Jiang, Jing Liu
- _Group_: Chi- nese Academy of Sciences, University of Chinese Academy of Sciences
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 Most existing EEG models are **task-specific** — each trained for **only one dataset or one task** (emotion recognition, sleep staging). They **can't generalize** across:
  - 📟 Devices
  - 🔌 Electrode layouts
  - 📊 Data formats
  
  Every new EEG application requires **retraining from scratch** → wastes time, data, and compute.

- _Focus problem_: 🔍 How to build a **generalist EEG foundation model** that handles **multiple datasets, devices, and tasks** within a single framework — **like GPT does for language**?

- _Why important_: 💡 EEG signals widely used in healthcare, neuroscience, BCIs, but:
  - Each dataset differs in: sampling rate, electrode number, preprocessing
  - Hard to **combine data or transfer knowledge**
  
  **Universal model** understanding EEG from different sources → much easier to apply AI to brain science and medical diagnostics ✅
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **BrainGPT** = first GPT-like foundation model for EEG with **three key innovations**:

| Innovation | Function |
|------------|----------|
| **1. Electrode-wise modeling** | Treats each electrode as individual data stream → training across datasets with different layouts (up to **138 channels unified**) |
| **2. Autoregressive pre-training** | Predicts **next time point** (not masked reconstruction) → better matches brain's temporal dynamics |
| **3. Task-shared graph network** | Learns how electrodes (brain regions) interact → enables multitask learning |

**Performance** (Average improvements vs. state-of-the-art):

| Task | Improvement |
|------|-------------|
| **Emotion recognition** | **+5.07%** |
| **Motor imagery** | **+6.05%** |
| **Cognitive workload** | **+8.50%** |
| **Sleep staging** | **+11.20%** 🏆 |
| **Cross-modal BCI** | **+5.10%** |

- _Why necessary_: 🏥 Traditional self-supervised EEG models have **critical limitations**:

| Method | Limitation |
|--------|-----------|
| **Masked reconstruction** | Only captures partial features |
| **Contrastive learning** | Fails to learn long-term dependencies |
| **Both** | Don't align with how brain activity unfolds |

BrainGPT's **autoregressive method** aligns with temporal dynamics: **"the past influences the future"** ✅
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 **BrainGPT treats EEG like language**:

| EEG Component | Language Analogy |
|---------------|------------------|
| **Electrode** | "Sentence" |
| **Time step** | "Word" |
| **Task** | Predict next "word" (signal value) |

**Architecture**:
- **Transformer with causal attention** (like GPT)
- **Graph neural layer** → electrodes communicate, learn spatial connections

- _Method_: 🔧

**Two-stage pipeline**:

### **(1) Pre-training**

| Component | Description |
|-----------|-------------|
| **Data** | **37.5M electrode samples** (≈1B time points) |
| **Task** | Autoregressive prediction (next time point) |
| **Architecture** | **Electrode Temporal Encoder (ETE)** — Transformer with causal attention |
| **Key finding** | Establishes **scaling laws for EEG**: Bigger model + more data → better performance 📈 |

### **(2) Multi-task fine-tuning**

| Component | Description |
|-----------|-------------|
| **Datasets** | **12 public EEG datasets** |
| **Tasks** | Emotion recognition, motor imagery, workload, sleep staging, cross-modal BCI |
| **Graph module** | **Task-Shared Electrode Graph (TEG)** models electrode relationships |
| **Efficiency** | **All tasks share same backbone** — no separate fine-tuning needed ✅ |


- _Result_: 📈

**Performance** (12 benchmarks, 5 task categories):

| Task | Improvement | Note |
|------|-------------|------|
| **Emotion recognition** | **+5.07%** | vs. SOTA |
| **Motor imagery** | **+6.05%** | vs. SOTA |
| **Cognitive workload** | **+8.50%** | vs. SOTA |
| **Sleep staging** | **+11.20%** 🏆 | vs. SOTA (highest) |
| **Cross-modal BCI** | **+5.10%** | vs. SOTA |


**Key advantages**:

| Feature | Task-Specific Models | BrainGPT |
|---------|---------------------|----------|
| **Generalization** | ❌ One task only | ✅ **Multiple tasks** |
| **Cross-device** | ❌ Fixed layout | ✅ **Up to 138 channels** |
| **Pre-training** | Limited/None | ✅ **37.5M samples** |
| **Temporal modeling** | Masked/Contrastive | ✅ **Autoregressive** |
| **Spatial modeling** | Fixed/None | ✅ **Task-shared graph** |
| **Transfer learning** | ❌ Requires retraining | ✅ **Strong zero-shot** |

- _Conclusion_: 🎓 **BrainGPT is the first true generalist EEG foundation model** by combining:
  - 🔌 **Electrode-wise representation**: Cross-device compatibility (up to 138 channels)
  - ⏱️ **Autoregressive pre-training**: Temporal prediction aligned with brain dynamics
  - 🔗 **Graph-based task sharing**: Spatial reasoning across electrodes
  - 🏆 **Unifies multiple EEG tasks** within one scalable model
  
  **Key achievements**:
  - 📈 **Establishes EEG scaling laws**: Bigger model + more data = better performance
  - 🎯 **Outperforms SOTA** on 12 benchmarks (+5-11% improvements)
  - 🌐 **Strong zero-shot transfer** to unseen datasets
  - ⚡ **Multi-task efficiency**: Shared backbone, no separate fine-tuning
  
  Applications:
  - 🏥 Clinical diagnosis (cross-device compatibility)
  - 🧠 Neuroscience research (unified analysis)
  - 🤖 Brain-computer interfaces (multi-task)
  - 😴 Sleep monitoring
  - 😊 Emotion recognition
  
  BrainGPT = GPT for brain signals, enabling universal EEG understanding across tasks and devices. 🧠🤖
</details>
</details>

---

> [*Towards Explainable Graph Neural Networks for Neurological Evaluation on EEG Signals*], [Sep 24, 2025]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 🏥 Explainable Graph Neural Networks for predicting stroke severity from EEG and visualizing brain network reorganization
- _Author_: Andrea Protani, Lorenzo Giusti, Chiara Iacovelli, Albert Sund Aillet, Diogo Reis Santos, Giuseppe Reale, Aurelia Zauli, Marco Moci, Marta Garbuglia, Pierpaolo Brutti, Pietro Caliandro, Luigi Serio
- _Group_: CERN, Sapienza University of Rome, Fondazione Policlinico Universitario Agostino Gemelli IRCCS
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 After a stroke, doctors need to **quickly and accurately assess severity** to choose best treatment. But traditional methods:
  - 📊 Rely on **hand-crafted EEG features**
  - 📝 Manual clinical scoring (NIH Stroke Scale)
  - ❌ May **miss complex ways** the brain reorganizes itself after stroke

- _Focus problem_: 🔍 How to create a model that **automatically predicts stroke severity** from EEG recordings and **shows which brain regions and connections matter most** for that prediction? Solution: **Graph Neural Networks (GNNs)** treating brain as network of connected regions.

- _Why important_: 💡 After stroke, brain **changes its functional connections**. Understanding these changes helps doctors:
  - 📈 **Monitor brain recovery**
  - 🔮 **Predict long-term outcomes**
  - 🏥 **Personalize rehabilitation treatments**
  
  Model that **both predicts and explains** could become valuable **decision-support tool** in hospitals ✅
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **GNNs can predict stroke severity (NIHSS score)** directly from EEG with **high accuracy**, while **identifying which connections** between brain regions are most affected:
  - 🔍 Analyzing **attention weights** in GNN reveals **interpretable patterns**
  - 🧠 Highlights which brain areas (Brodmann regions) show **strongest reorganization**
  - ✅ **Not only predicts how bad** the stroke is but also **explains where** damage affects brain network

**Key results**:

| Metric | Performance |
|--------|-------------|
| **MAE** | **3.57 ± 0.6** between predicted and real NIHSS scores |
| **Best performance** | Moderate stroke severity (NIHSS 9–15) |
| **Model size** | Lightweight (~60k parameters) |

- _Why necessary_: 🏥 Traditional approaches have **critical limitations**:

| Challenge | Problem |
|-----------|---------|
| **EEG signals** | Messy and complex |
| **Hand-designed features** | Lose brain's **network information** |
| **Classic EEG models** | Can't capture **relationships between regions** |

By modeling EEG as graph: each region = node, functional connections = edges → learn how stroke **disrupts brain's structure** ✅
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 Model treats **brain as multi-layer graph**:

| Component | Description |
|-----------|-------------|
| **Layers** | Each = frequency band (α1, α2, β1) |
| **Nodes** | Brodmann areas (defined brain regions) |
| **Edges** | Strong signal correlations (Lagged Linear Coherence, LLC) |

Use **Graph Attention Networks (GATs)** to learn which connections are most important for predicting stroke severity.

**Graph optimization**: "Rewire" graphs — keep only **top 5% strongest/nearest connections**, remove noisy/irrelevant links.

- _Method_: 🔧

**Data**: EEGs from **71 stroke patients**, each scored with **NIHSS (2–22)**

**Processing pipeline**:

| Step | Description |
|------|-------------|
| **1. EEG cleaning** | ICA artifact removal |
| **2. Source localization** | eLORETA |
| **3. Frequency bands** | Extract 5 bands (δ, θ, α1, α2, β1) |

**Graph construction**:

| Component | Details |
|-----------|---------|
| **Layers** | α1, α2, β1 (3 frequency bands used for final model) |
| **Nodes** | 84 Brodmann areas × 3 layers = **252 total nodes** per patient |
| **Edge weights** | Functional (LLC) + Structural (distance) links |

**Model**:
- **Architecture**: Lightweight **2-layer GATv2** (~60k parameters)
- **Training**: 5-fold cross-validation
- **Task**: Predict NIHSS score


- _Result_: 📈

**Prediction accuracy**:

| Metric | Result |
|--------|--------|
| **MAE** | **3.57 ± 0.6** 🏆 |
| **Best performance** | Moderate severity (NIHSS 9–15) |
| **Comparable to** | Human-level clinical scoring variability |

**Interpretability findings**:

| Finding | Details |
|---------|---------|
| **Attention maps** | Show brain regions on **affected side receive higher weights** |
| **Left-hemisphere strokes** | Key activity on **left** ✅ |
| **Right-hemisphere strokes** | Key activity on **right** ✅ |
| **Brain reorganization** | Reduced α and β activity in damaged areas (consistent with neuroscience) |


- _Conclusion_: 🎓 **Explainable Graph Neural Networks** serve as powerful, interpretable tool for neurological evaluation:
  - 🏆 **Accurately predicts** stroke severity from resting-state EEG (MAE 3.57)
  - 🔍 **Visualizes brain connectivity changes** after stroke
  - 🧠 **Identifies affected regions** via attention weights (left vs. right hemisphere)
  - 📊 **Reflects brain reorganization** (reduced α/β activity in damaged areas)
  - ⚡ **Lightweight and fast** (~60k parameters)
  
  **Opens new possibilities**:
  - ⏱️ **Real-time monitoring** of stroke patients through EEG
  - 🏥 **Personalized rehabilitation** based on brain network changes
  - 🤖 **Integration with clinical systems** for automated assessment
  - 💰 **Low-cost alternative** to expensive imaging (MRI, PET)
  
  Explainable GNNs = predicting severity AND understanding brain network disruption in stroke patients. 🧠🏥
</details>
</details>

---

> [*Parkinson’s Disease Detection from Resting State EEG using Multi-Head Graph Structure Learning with Gradient Weighted Graph Attention Explanations*], [Aug 1, 2024]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 🧠 Explainable graph-based deep learning for Parkinson's disease detection from resting-state EEG
- _Author_: Christopher Neves, Yong Zeng, and Yiming Xiao
- _Group_: Concordia University
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 Parkinson's disease (PD) is a major neurodegenerative disorder severely affecting motor and cognitive function:
  - 🏥 **MRI** reveals brain changes but is **expensive and not easily accessible**
  - 📊 **EEG** offers cheaper, portable alternative but most deep learning approaches have **three big issues**:

| Issue | Problem |
|-------|---------|
| **1. Ignore spatial relationships** | Between electrodes ❌ |
| **2. Overfitting** | EEG datasets are small |
| **3. Lack interpretability** | Clinicians can't trust/understand reasoning |

- _Focus problem_: 🔍 How to build an **explainable graph-based model** that can:
  - ✅ **Automatically detect PD** from resting-state EEG
  - ✅ **Show which brain regions and connections** contribute most to diagnosis

- _Why important_: 💡 PD is **second most common neurodegenerative disorder** worldwide. If EEG provides **reliable and interpretable biomarkers**:
  - 🌍 **Early screening** in low-resource areas (no MRI access)
  - 📈 **Continuous monitoring** and rehabilitation
  - 🏥 **Build physician trust** through transparency and interpretability
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **Multi-Head Graph Structure Learning (MH-GSL)** framework:
  - 🔗 **Dynamically learns** brain connectivity structure (not fixed correlations)
  - 💪 **Combines contrastive learning** for stronger generalization
  - 🔍 **Uses gradient-weighted attention maps** to visualize relevant brain connections

**Key results**:

| Metric | Performance | Note |
|--------|-------------|------|
| **Accuracy** | **69.4%** | Leave-one-subject-out validation |
| **Dataset** | UCSD resting-state EEG | 15 PD, 16 healthy |
| **Interpretability** | ✅ Brain network visualizations | Stronger **occipital connectivity** in PD patients |

- _Why necessary_: 🏥 Traditional EEG graph methods have **critical limitations**:

| Traditional Approach | Problem |
|---------------------|---------|
| **Fixed correlations** | Use static Pearson correlation |
| **Can't capture** | Nonlinear or dynamic interactions |
| **Overestimate** | Nearby electrodes (signal mixing) |
| **Fail to represent** | True brain network changes |

**MH-GSL solves this** by **learning graph structure during training** → each attention head learns unique connectivity pattern = more robust and biologically meaningful ✅
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 Framework integrates **four main components**:

| Component | Function |
|-----------|----------|
| **1. Structured Global Convolution (SGConv)** | Encodes long-range temporal patterns efficiently |
| **2. Contrastive Learning (SimCLR)** | Pre-trains encoder for general EEG features, improves robustness with small datasets |
| **3. Multi-Head Graph Structure Learning (MH-GSL)** | Each head learns different adjacency matrix (brain connectivity), fused through Chebyshev GNN |
| **4. Gradient-Weighted Graph Attention** | Uses gradients to measure connection contributions → interpretable connectivity maps |

- _Method_: 🔧

**Dataset**: UC San Diego (UCSD) resting-state EEG

| Property | Details |
|----------|---------|
| **Participants** | 15 PD patients, 16 healthy controls |
| **Electrodes** | 32 channels |
| **Sampling rate** | 512 Hz |
| **Recording** | 3-minute resting-state |


**Training setup**:

| Step | Description |
|------|-------------|
| **Validation** | Leave-One-Subject-Out (ensures generalization) |
| **Pre-training** | Contrastive learning on EEG segments |
| **Fine-tuning** | Frozen encoder + 2-head graph structure learner |
| **Spatial aggregation** | Chebyshev GNN |

**Pipeline**:
```
EEG → SGConv (temporal) → Contrastive Pre-training → MH-GSL (spatial) → Chebyshev GNN → Classification
                                                                              ↓
                                                                    Gradient-Weighted Explanation
```

- _Result_: 📈

**Performance comparison**:

| Model | Accuracy | F1-score | AUC |
|-------|----------|----------|-----|
| **CNN baseline** | 62.99% | 0.63 | 0.64 |
| **LongConv Encoder** | 64.68% | 0.64 | 0.64 |
| **+ GNN** | 66.97% | 0.66 | 0.67 |
| **+ Multi-Head GSL** | 67.73% | 0.67 | 0.72 |
| **+ Contrastive Learning** | **69.40%** 🏆 | **0.68** | 0.66 |

**Key observations**:

| Finding | Details |
|---------|---------|
| **Component contributions** | Both **contrastive pre-training** and **multi-head structure learning** significantly improved accuracy |
| **Interpretability** | Gradient-based visualization revealed **enhanced occipital connectivity** in PD patients |
| **Clinical relevance** | Consistent with known **motor and visual processing abnormalities** in PD ✅ |


- _Conclusion_: 🎓 **Dynamic and explainable graph neural network** for EEG-based Parkinson's disease detection effectively combines:
  - ⏱️ **Global convolution** (temporal encoding)
  - 💪 **Contrastive learning** (robustness)
  - 🔗 **Multi-head graph learning** (spatial representation)
  - 🔍 **Gradient-weighted explanations** (interpretability)
  
  **Key achievements**:
  - 🏆 **69.4% accuracy** (leave-one-subject-out validation)
  - 🧠 **Visualizes neural connections** driving decisions
  - 📊 **Reveals enhanced occipital connectivity** in PD patients
  - 🏥 **Interpretable and clinically useful** diagnostic tool
  
  Applications:
  - 🌍 Early PD screening (low-resource areas)
  - 📈 Continuous patient monitoring
  - 🏥 Rehabilitation planning
  - 🔬 Understanding PD brain network changes
  
  Multi-head graph structure learning = accurate AND explainable PD detection from affordable EEG. 🧠💡
</details>
</details>

---

> [*REST- Efficient and Accelerated EEG Seizure Analysis through Residual State Updates*], [Jun 3, 2024] （ICML 2024）:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: ⚡ Ultra-lightweight, real-time EEG seizure detection using graph neural networks with residual state updates
- _Author_: Arshia Afzal, Grigorios Chrysos, Volkan Cevher, Mahsa Shoaran
- _Group_: EPFL, University of Wisconsin-Madison
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 Traditional deep-learning methods for EEG seizure analysis (CNNs, RNNs, LSTMs, Transformers) are **accurate but too heavy**:
  - 💾 **Large memory** usage
  - ⏱️ **Take too long** to run
  
  → **Unsuitable for real-time clinical devices** (Responsive Neurostimulation - RNS, Deep Brain Stimulation - DBS) that must **react instantly** to prevent seizures.

- _Focus problem_: 🔍 How to design a **lightweight, real-time EEG model** that:
  - ✅ Captures **complex spatial and temporal** brain patterns for seizure detection
  - ❌ **Without** slow "gating" or "attention" mechanisms (RNNs, Transformers)

- _Why important_: 💡 **Millions of epilepsy patients** rely on seizure detection systems:
  - ⚡ Systems must trigger **brain stimulation in milliseconds**
  - ❌ Current models **too large or slow** for medical hardware deployment
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **REST** combines:

| Component | Function |
|-----------|----------|
| **Graph Neural Networks (GNNs)** | Model how EEG sensors relate **spatially** on scalp |
| **Residual State Updates** | New mechanism (ResNet-inspired) updates states **efficiently without LSTM gates** |

→ Captures brain rhythms over time while staying **compact and fast** ✅

**Key results**:

| Metric | Performance |
|--------|-------------|
| **AUROC** | Up to **96.7%** on seizure detection 🏆 |
| **Speed** | **9× faster** than SOTA models ⚡ |
| **Memory** | Only **37 KB** (14× smaller than smallest baseline) 💾 |
| **Inference time** | **~1.3 ms** per sample (real-time capable) |

- _Why necessary_: 🏥 **Residual updates replace complex recurrent gates** while keeping temporal learning ability → enables real-time, low-power detection ✅
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 **REST architecture**:
  - 🔗 **Graph structure**: Efficiently encode spatial EEG relationships
  - 🔄 **Residual updates**: Replace LSTM gates with lightweight state updates
  - 🎲 **Binary random masking**: Skip updates for speed + regularization

- _Method_: 🔧

**Core mechanism**:

**1. Graph construction**:
- Each **electrode = node**
- Connections based on **distance** (10-20 placement system)

**2. Residual update** (each time step):
```
S_t = H_t + δS_t

where:
- H_t = hidden state projection
- δS_t = computed using graph convolution
```

**3. Binary random masking**:
- Similar to dropout
- Randomly **skip updates** → speed up inference

**4. Multiple updates per time step**:
- Mimics deep network
- **No extra layers or memory** needed


**Datasets**:
- **TUSZ** (Temple University Seizure Database)
- **CHB-MIT** (Children's Hospital Boston - MIT)

- _Result_: 📈

**Performance comparison**:

| Metric | REST | SOTA Models |
|--------|------|-------------|
| **AUROC** | **96.7%** 🏆 | Lower |
| **Inference speed** | **9× faster** ⚡ | Baseline |
| **Memory** | **37 KB** 💾 | 518 KB (14× larger) |
| **Inference time** | **~1.3 ms** | Much slower |

**Key advantages**:

| Feature | Traditional Models | REST |
|---------|-------------------|------|
| **Temporal modeling** | LSTM gates (slow) | **Residual updates** ⚡ |
| **Spatial modeling** | Limited/None | **Graph structure** 🔗 |
| **Speed** | Slow | **9× faster** |
| **Memory** | Large (500+ KB) | **37 KB** (14× smaller) |
| **Real-time capable** | ❌ No | ✅ **Yes (~1.3 ms)** |
| **Hardware deployment** | ❌ Difficult | ✅ **Medical devices ready** |


- _Conclusion_: 🎓 **REST is a compact, ultra-fast EEG model** balancing accuracy, speed, and efficiency:
  - 🏆 **96.7% AUROC** on seizure detection
  - ⚡ **9× faster** than state-of-the-art
  - 💾 **37 KB memory** (14× smaller)
  - ⏱️ **~1.3 ms inference** (real-time capable)
  - 🔄 **Residual updates** replace LSTM gates effectively
  - 🔗 **Graph structure** captures spatial relationships
  
  **Key insight**: Graph-based residual state updates can enable **real-time, low-power seizure detection** without sacrificing accuracy.
  
  Applications:
  - 🏥 **Responsive Neurostimulation (RNS)** devices
  - 🧠 **Deep Brain Stimulation (DBS)** systems
  - 📱 **On-chip epilepsy monitoring**
  - ⚡ **Next-generation neural prosthetics**
  
  REST = making real-time seizure prevention practical with ultra-efficient hardware deployment. ⚡🧠
</details>
</details>

---

> [*Dynamic GNNs for Precise Seizure Detection and Classification from EEG Data*], [May 8, 2024]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 🧠 Dynamic multi-view graph neural network combining temporal, spatial, and semantic information for seizure detection and classification
- _Author_: Arash Hajisafi, Haowen Lin, Yao-Yi Chiang, and Cyrus Shahabi
- _Group_: University of Southern California, Los Angeles, University of Minnesota, Minneapolis
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 Detecting and classifying epileptic seizures from EEG is difficult because:
  - 📊 EEG signals are **complex, noisy**
  - 👥 **Vary a lot between patients**
  
  Traditional deep learning models find **time patterns** but **fail to understand how brain regions** (EEG electrodes) **interact** with each other.

- _Focus problem_: 🔍 How to build a model that captures:
  - ⏱️ **Changing relationships** between brain regions over time
  - 🧠 **Meaning** of different brain areas
  
  → Recognize various seizure types more accurately

- _Why important_: 💡 Each seizure type:
  - 🧠 **Affects different brain areas**
  - 💊 **Requires different treatments**
  
  Current models miss deeper relationships — only see EEG as **flat signal**. If model tracks **both activity changes AND region importance** → doctors can:
  - ⚡ **Detect seizures faster**
  - 🎯 **Understand seizure types better** (even with limited data)
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **NeuroGNN** learns from **multiple perspectives** simultaneously:

| View | What It Captures |
|------|------------------|
| **1. Time** | How signal changes over time ⏱️ |
| **2. Space** | How nearby electrodes interact 🔗 |
| **3. Meaning** | What brain region each electrode represents 🧠 |
| **4. Hierarchy** | How smaller areas connect to larger functional regions 🏗️ |

→ Multi-view approach understands **both sequence and structure** → more precise seizure detection and classification ✅

**Performance improvements**:

| Task | Improvement |
|------|-------------|
| **Detection accuracy** | **+5%** 🏆 |
| **Classification accuracy** | **+12-13%** 🏆 |
| **With 20% data** | Strong performance ✅ |

- _Why necessary_: 🏥 In real-world EEG data:

| Challenge | Problem |
|-----------|---------|
| **Same seizure type** | Looks different across patients |
| **Overlapping info** | Electrodes capture redundant signals |
| **Fixed graphs** | Can't reflect constantly shifting brain networks |

**Dynamic graph model** needed to handle this fluidity ✅
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 Treat **brain as dynamic graph**:

| Component | Description |
|-----------|-------------|
| **Nodes** | EEG electrodes (brain regions) |
| **Edges** | Change over time (how areas interact) |
| **Meta-nodes** | Represent large brain regions (motor, visual areas) |
| **Semantic knowledge** | Language model (MPNet) gives brain region function understanding |

System sees **both signal and meaning** behind it ✅

- _Method_: 🔧

**Five-step pipeline**:

| Step | Description |
|------|-------------|
| **1. Feature extraction** | Node features include:<br>• Time info (GRUs)<br>• Brain meaning (text embeddings) |
| **2. Dynamic adjacency matrix** | New graph every few seconds using:<br>• Spatial similarity (electrode distance)<br>• Temporal correlation (attention-based)<br>• Semantic similarity (brain region meaning) |
| **3. Graph learning** | Modified GNN passes info across nodes<br>• Hierarchical pooling for brain-region patterns |
| **4. Pre-training** | Self-supervised: predict future EEG signals |
| **5. Fine-tuning** | Two tasks:<br>• Seizure detection<br>• Seizure type classification |


**Dataset**: Temple University Seizure Corpus (TUSZ, **8 seizure types**)

**Baselines**: CNN, LSTM, DCRNN

- _Result_: 📈

**Data efficiency**:

| Training Data | Performance |
|---------------|-------------|
| **100%** | Best |
| **20%** | **Still strong** ✅ |

→ Learns efficiently from **limited samples**

**Key advantages**:

| Feature | Traditional Models | NeuroGNN |
|---------|-------------------|----------|
| **Temporal modeling** | ✅ Yes | ✅ **Enhanced** |
| **Spatial modeling** | Limited | ✅ **Dynamic graph** 🔗 |
| **Semantic understanding** | ❌ None | ✅ **Brain region meaning** 🧠 |
| **Hierarchy** | ❌ Flat | ✅ **Multi-level** 🏗️ |
| **Adaptability** | ❌ Fixed | ✅ **Dynamic** (updates in real-time) |
| **Data efficiency** | Poor | ✅ **Strong with 20% data** |

- _Conclusion_: 🎓 **NeuroGNN successfully combines time, space, and meaning** to model brain behavior during seizures:
  - ⏱️ **Time**: Signal evolution (GRU)
  - 🔗 **Space**: Dynamic electrode interactions (graph)
  - 🧠 **Meaning**: Brain region functions (MPNet embeddings)
  - 🏗️ **Hierarchy**: Small → large brain areas
  
  Applications:
  - 🏥 Clinical seizure detection systems
  - 🎯 Seizure type classification (8 types)
  - 📈 Patient-specific monitoring
  - 🧠 Understanding seizure mechanisms
  
  Multi-view dynamic graph learning = capturing temporal dynamics + spatial interactions + semantic meaning for superior seizure analysis. 🧠⏱️🔗
</details>
</details>

---

> [*EEG Decoding for Datasets with Heterogenous Electrode Configurations using Transfer Learning Graph Neural Networks*], [Jun 20, 2023]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 🔗 Graph neural networks with transfer learning for combining heterogeneous EEG datasets with different electrode configurations
- _Author_: Jinpei Han, Xiaoxi Wei, A. Aldo Faisal
- _Group_: Imperial College London
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 EEG data from different labs use **different electrode numbers and layouts** → very hard to combine datasets. Deep learning needs **lots of data**, but EEG data is:
  - 📊 **Small**
  - 🔀 **Inconsistent**
  - 👥 **Varies across subjects and devices**

- _Focus problem_: 🔍 How to **combine EEG datasets** with different electrode configurations and still **accurately classify motor imagery** (imagining body movements)?

- _Why important_: 💡 If we can **merge EEG data** from many different setups:
  - 💪 **Train stronger, more general models** for BCIs
  - 💰 **Reduce data collection costs**
  - 🏥 **More reliable systems** for medical and assistive technologies
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **GNNs + transfer learning** can align data from different electrode layouts and subjects:
  - 🔗 Transfer knowledge across datasets
  - 📈 **Higher accuracy and stability** than CNN-based approaches

**Performance**:

| Dataset | Channels | Accuracy |
|---------|----------|----------|
| **BCIC IV 2a** | 22 | **72.5%** 🏆 |
| **PhysioNet MI** | 64 | **74.4%** 🏆 |
| **OpenBMI** | 62 | **72.6%** 🏆 |

- _Why necessary_: 🏥 Existing methods **fail** because:

| Problem | Issue |
|---------|-------|
| **CNN/RNN** | Treat EEG as flat grid, ignore electrode positions |
| **Common electrodes only** | Wastes information |
| **Device differences** | Signals differ across devices |

Need model capturing **both spatial relationships AND cross-dataset differences** ✅
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 Represent **EEG sensors as graph nodes**:
  - **Edges**: Show how brain regions connect
  - **Each dataset**: Own GNN encoder (learns unique electrode structure)
  - **Shared alignment**: Distance-based loss (**Maximum Density Divergence, MDD**) makes feature spaces overlap
  
  → Model learns **general patterns**, not dataset-specific ones ✅

- _Method_: 🔧

**Five-step pipeline**:

| Step | Description |
|------|-------------|
| **1. Temporal feature extraction** | CNN blocks process EEG time-series signals |
| **2. Spatial graph learning** | Convert electrodes to graph:<br>• **Neighbourhood method** (electrode geometry)<br>• **Correlation method** (signal similarity) |
| **3. Graph pooling** | **SAGPooling**: Keep most important brain nodes |
| **4. Latent alignment** | MLP block + **MDD loss** align features across datasets |
| **5. Classification** | Predict left vs. right-hand motor imagery |


**Datasets tested**:

| Dataset | Channels | Configuration |
|---------|----------|---------------|
| **BCIC IV 2a** | 22 | Different layout |
| **PhysioNet MI** | 64 | Different layout |
| **OpenBMI** | 62 | Different layout |

- _Result_: 📈

**Performance comparison**:

| Dataset | GNN + Transfer | Traditional CNN |
|---------|---------------|-----------------|
| **BCIC IV 2a** | **72.5%** 🏆 | Lower |
| **PhysioNet MI** | **74.4%** 🏆 | Lower |
| **OpenBMI** | **72.6%** 🏆 | Lower |

**Key advantages**:

| Metric | Result |
|--------|--------|
| **Accuracy** | Best across all datasets 🏆 |
| **Stability** | Lower standard deviation ✅ |
| **Generalization** | Better on unseen subjects ✅ |

**Visualization findings**:
- ✅ After alignment: Features **grouped by task type** (left vs. right hand)
- ❌ Not by dataset or subject
- → Model learned **shared brain patterns** ✅

**Feature alignment**:

| Before Alignment | After Alignment |
|------------------|-----------------|
| Grouped by dataset/subject ❌ | Grouped by task (left/right) ✅ |
| Poor generalization | Strong generalization ✅ |

- _Conclusion_: 🎓 **Combining GNNs with transfer learning** allows learning from **heterogeneous EEG datasets** without losing spatial or functional information:
  - 🔗 **Graph representation**: Captures electrode spatial relationships
  - 🔄 **Transfer learning**: Aligns different datasets (MDD loss)
  - 📈 **72-74% accuracy** across datasets with different configurations
  - 📊 **More stable** (lower variance)
  - 🌐 **Better generalization** to unseen subjects
  - 🧠 Learns **shared brain patterns** (not dataset-specific)
  
  **Key innovation**: MDD loss makes features from different datasets **overlap in latent space** → unified representation ✅
  
  Applications:
  - 🤖 Robust brain-computer interfaces
  - 🏥 Cross-lab medical research
  - 💰 Reduced data collection costs
  - 🔬 Larger-scale brain studies
  - 📊 May extend to other signals (fNIRS, EMG)
  
  GNN + transfer learning = unifying fragmented biomedical EEG data for stronger, more general models. 🧠🔗
</details>
</details>

---

> [*GMSS- Graph-Based Multi-Task Self-Supervised Learning for EEG Emotion Recognition*], [Apr 12, 2022]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 😊 Graph-based multi-task self-supervised learning for robust EEG emotion recognition
- _Author_: Yang Li, Member, Ji Chen, Fu Li, Boxun Fu, Hao Wu, Youshuo Ji, Yijin Zhou, Yi Niu, Guangming Shi, Wenming Zheng
- _Group_: Southeast University
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 Most existing EEG-based emotion recognition models rely on **single-task learning**, which:
  - 📉 Tends to **overfit**
  - ❌ Lacks **generalization across subjects**
  
  Moreover, EEG emotion labels are often **noisy** since emotional responses are **inconsistent**.

- _Focus problem_: 🔍 How to learn **robust and general EEG emotion representations** without depending on **large amounts of high-quality labeled data**?

- _Why important_: 💡 EEG signals capture emotions **more directly** than facial or vocal data but are:
  - 👥 **Highly individual**
  - ⚡ **Nonstationary**
  
  Improving **generalization and noise robustness** is crucial for real-world emotion-aware systems:
  - 🤖 Affective computing
  - 💬 Human-computer interaction
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **GMSS (Graph-based Multi-task Self-Supervised Learning)** learns rich EEG representations through **three complementary self-supervised tasks**:

| Task | Function |
|------|----------|
| **1. Spatial jigsaw puzzle** | Shuffles EEG channels among brain regions → capture **spatial dependencies** 🧠 |
| **2. Frequency jigsaw puzzle** | Shuffles frequency bands (δ, θ, α, β, γ) → identify **emotion-relevant spectral information** 📊 |
| **3. Contrastive learning** | Pulls together augmented versions of same signal, pushes apart different samples → learn **semantic representations** 🔗 |

**Key findings**:

| Finding | Result |
|---------|--------|
| **Unsupervised mode** | **+2-8% accuracy** vs. SOTA SSL methods (SimCLR, MoCo, SeqCLR) 🏆 |
| **Supervised mode** | **Highest accuracy** among all models (BiHDM, RGNN, DGCNN) 🏆 |
| **Most influential component** | **Spatial jigsaw** (but combining all three = best) ✅ |
| **Noise robustness** | Performs well even with **noisy or limited labels** 💪 |

- _Why necessary_: 🏥 Need to overcome:
  - Limited high-quality labeled EEG emotion data
  - High inter-subject variability
  - Noisy emotion labels (inconsistent responses)
  
  Multi-task self-supervised learning provides solution ✅
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 Integrate **three self-supervised tasks** to learn from **spatial, spectral, and semantic** perspectives:

```
Spatial (where) + Frequency (what) + Contrastive (meaning) = Robust emotion representation
```

- _Method_: 🔧

**Architecture**:

| Component | Description |
|-----------|-------------|
| **Shared feature extractor** | Graph Neural Network (GNN) modeling EEG spatial topology |
| **Chebyshev polynomial filters** | Reduce computation while preserving multi-hop neighborhood info ⚡ |
| **Three self-supervised heads** | Spatial jigsaw, Frequency jigsaw, Contrastive learning |
| **Two training modes** | Unsupervised (SSL only) + Supervised (SSL + classification) |

**Three self-supervised tasks**:

**1️⃣ Spatial Jigsaw Puzzle**
```
Shuffle EEG channels among brain regions
→ Model predicts original spatial arrangement
→ Learns spatial dependencies between brain areas
```

**2️⃣ Frequency Jigsaw Puzzle**
```
Shuffle frequency bands (δ, θ, α, β, γ)
→ Model predicts original frequency order
→ Identifies emotion-relevant spectral patterns
```

**3️⃣ Contrastive Learning**
```
Augment same EEG signal → pull representations together
Different EEG signals → push representations apart
→ Learns discriminative semantic features
```

**Training modes**:

| Mode | Description |
|------|-------------|
| **Unsupervised** | Train only on self-supervised tasks → test with linear classifier |
| **Supervised** | Jointly optimize self-supervised + emotion classification using **uncertainty-based loss weighting** |


**Datasets**: 
- **SEED**
- **SEED-IV**
- **MPED**

**Baselines**: SimCLR, MoCo, SeqCLR, BiHDM, RGNN, DGCNN

- _Result_: 📈

**Key advantages**:

| Feature | Traditional Models | GMSS |
|---------|-------------------|------|
| **Learning paradigm** | Single-task | **Multi-task SSL** 💪 |
| **Data requirement** | Large labeled datasets | **Works with limited labels** ✅ |
| **Generalization** | Poor cross-subject | **Strong generalization** 🌐 |
| **Noise robustness** | Sensitive to label noise | **Robust to noisy labels** 💪 |
| **Feature learning** | Spatial only | **Spatial + Spectral + Semantic** 🎯 |

- _Conclusion_: 🎓 **GMSS is the first framework** integrating **multi-task self-supervised learning** into EEG emotion recognition:
  - 🧠 **Spatial jigsaw**: Captures brain region dependencies
  - 📊 **Frequency jigsaw**: Identifies emotion-relevant spectral patterns
  - 🔗 **Contrastive learning**: Learns semantic representations
  - 🏆 **Superior performance**: +2-8% unsupervised, highest in supervised mode
  - 💪 **Noise robust**: Works with noisy or limited labels
  - 🎯 **Interpretable**: Clear emotion clusters in visualizations
  
  **Key insight**: Combining spatial, spectral, and semantic learning = robust emotion recognition without heavy reliance on labeled data ✅
  
  Applications:
  - 😊 Affective computing
  - 💬 Human-computer interaction
  - 🎮 Emotion-aware gaming
  - 🏥 Mental health monitoring
  
  Multi-task self-supervised learning = achieving superior EEG emotion recognition with less labeled data. 😊🧠
</details>
</details>

---

> [*EEG-GNN- Graph Neural Networks for Classification of Electroencephalogram (EEG) Signals*], [Jun 16, 2021]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 🧠 Graph neural networks for EEG classification preserving brain topology and functional connectivity
- _Author_: Andac Demir, Toshiaki Koike-Akino, Ye Wang, Masaki Haruna, Deniz Erdogmus
- _Group_: Northeastern University, Mitsubishi Electric Research Laboratories (MERL), Mitsubishi Electric Corporation (MELCO)
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 Traditional **CNN-based EEG classifiers** assume electrodes are arranged like **pixels on a grid** — equidistant and independent. However:
  - 🧠 EEG channels are **not spatially uniform**
  - 🔗 Brain's **functional connectivity** (how regions influence each other) is much **more complex**
  
  → This leads to **loss of neuroscientific information** when using CNNs.

- _Focus problem_: 🔍 How to design a model that:
  - 🔵 Represents electrodes as **nodes in a graph** (reflecting true spatial and functional relationships)
  - 🧠 Learns features that capture **inter-regional connectivity**
  - 📈 Improves **classification accuracy** while offering **interpretability**

- _Why important_: 💡 Brain regions don't work in isolation — they form **complex networks**. Models should respect this structure to:
  - ✅ Preserve neuroscientific meaning
  - ✅ Enable interpretation of brain activity patterns
  - ✅ Improve BCI performance
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **EEG-GNN** maps electrodes to graph nodes with **flexible adjacency construction**:

| Adjacency Method | Description |
|-----------------|-------------|
| **Fully connected** | All electrodes connected |
| **k-nearest neighbors (k-NNG)** | Connect k closest electrodes (sparse) |
| **Distance threshold** | Connect if distance < threshold |
| **Functional connectivity** | Weight edges by Pearson correlation |

→ Allows neuroscientists to **tailor graph to specific experiments** or brain regions ✅

**Performance**:

| Dataset | EEG-GNN | CNN | Bayesian |
|---------|---------|-----|----------|
| **ErrP** | **76.7%** 🏆 | 74.7% | 75.9% |
| **RSVP** | **93.5%** 🏆 | 93.1% | - |

**Efficiency**:
- GNN parameters: **≈80-100k**
- CNN/Bayesian: **Millions**

- _Why necessary_: 🏥 CNNs treat EEG as **flat grid** → ignore:
  - Brain's true spatial topology
  - Functional connectivity patterns
  - Inter-regional relationships
  
  Need graph-based approach to preserve neuroscientific structure ✅
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 **EEG-GNN framework**:
  - 🔵 Each **electrode = graph node**
  - 🔗 Each **connection = edge** (spatial proximity or neural correlation)
  - 📊 **Adjacency matrix** built flexibly (fully connected, k-NNG, distance threshold, functional)

- _Method_: 🔧

**1️⃣ Graph Representation**

| Component | Description |
|-----------|-------------|
| **Nodes** | EEG electrodes |
| **Node features** | EEG time samples |
| **Edges** | Spatial proximity OR neural correlation |

**2️⃣ Model Architecture**

**GNN variants tested**:

| Model | Type |
|-------|------|
| **GraphSAGE** | Neighborhood aggregation |
| **GIN** (Graph Isomorphism Network) | Expressive graph learning |
| **SortPool** | Pooling-based |
| **EdgePool** | Edge-based pooling |
| **SagPool** | Self-attention pooling |
| **Set2Set** | Set aggregation |

**Each GNN layer**: Aggregates information from neighbors → learns spatial-temporal patterns

**3️⃣ Data Preprocessing**

| Technique | Purpose |
|-----------|---------|
| **Temporal compression** | 1D convolutions reduce feature size, prevent overfitting |
| **Data augmentation** | Add Gaussian noise → improve generalization |
| **Regularization** | L1, L2, ElasticNet → reduce model bias |

**4️⃣ Datasets**

| Dataset | Task |
|---------|------|
| **ErrP** | Error-related potentials during P300 spelling tasks |
| **RSVP** | Rapid visual presentation keyboard tasks |


- _Result_: 📈

**Model efficiency**:

| Model Type | Parameters |
|------------|-----------|
| **EEG-GNN** | **≈80-100k** ⚡ |
| **CNN/Bayesian** | Millions |

**Graph sparsity findings**:
- ✅ **Sparse adjacency matrices** (k-nearest neighbor) perform **as well as** fully connected
- ✅ But with **lower computational cost**

**Interpretability**:
- ✅ Identifies **important nodes** (electrodes)
- ✅ Reveals **critical connections** between regions
- ✅ Enables **neuroscientific insight**

**Advantages**:

| Feature | CNN | EEG-GNN |
|---------|-----|---------|
| **Spatial modeling** | Grid assumption ❌ | **True topology** ✅ |
| **Connectivity** | Ignored ❌ | **Preserved** ✅ |
| **Parameters** | Millions | **80-100k** ⚡ |
| **Interpretability** | Limited | **High** 🔍 |
| **Accuracy** | Lower | **Higher** 🏆 |

- _Conclusion_: 🎓 **EEG-GNN introduces graph-based paradigm** for EEG classification that:
  - 🧠 **Preserves brain's topology** (not flat grid)
  - 📈 **Improves accuracy**: +2% (ErrP), +0.4% (RSVP)
  - ⚡ **Reduces complexity**: 80-100k params vs. millions
  - 🔍 **Enables interpretability**: Identifies key electrodes and connections
  - 🎯 **Flexible graph construction**: Tailored to experiments
  
  **Future directions**:
  - 🤖 **Data-driven edge learning** (learn adjacency automatically)
  - 🔗 **Weighted graphs** (correlation-based edges)
  - 🌐 **More general BCI tasks** (motor imagery, emotion, etc.)
  
  Applications:
  - 🤖 Brain-computer interfaces
  - 🏥 Clinical EEG analysis
  - 🧠 Neuroscience research
  - 📊 Error detection systems
  
  Graph neural networks = respecting brain's network structure for better EEG classification. 🧠🔗
</details>
</details>

---

> [*Self-Supervised Graph Neural Networks for Improved Electroencephalographic Seizure Analysis*], [Mar 13, 2022] (ICLR 2022):
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: ⚡ Graph-based recurrent neural network with self-supervised pre-training for seizure detection, classification, and localization
- _Author_: Siyi Tang, Jared Dunnmon, Khaled Saab, Xuan Zhang, Qianying Huang, Florian Dubost, Daniel Rubin, Christopher Lee-Messer
- _Group_: Stanford University
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 Traditional deep learning models for seizure detection often have **three issues**:

| Issue | Problem |
|-------|---------|
| **1. Euclidean assumption** | Treat EEG as images, **ignore non-Euclidean brain topology** 🧠 |
| **2. Data imbalance** | **Struggle with rare seizure types** 📊 |
| **3. Lack interpretability** | Cannot **explain where seizures occur** ❌ |

- _Focus problem_: 🔍 How to solve **three issues simultaneously**:
  - ✅ Represent EEG's **spatiotemporal and network structure** faithfully
  - ✅ Improve **classification performance** (especially rare seizure classes)
  - ✅ **Quantitatively evaluate** seizure localization ability

- _Why important_: 💡 Seizure localization is **clinically critical** for:
  - 🏥 Diagnosis and treatment planning
  - 🔪 Surgical intervention (identifying seizure onset zones)
  - 💊 Personalized therapy
  
  Current models can't provide this **spatial information** ❌
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **Graph-based RNN + Self-supervised pre-training** achieves:

**Performance**:

| Task | Metric | Result | vs. Baseline |
|------|--------|--------|-------------|
| **Detection** | AUROC | **0.875** 🏆 | Surpasses CNN/LSTM |
| **Classification** | Weighted F1 | **0.749** 🏆 | Higher than previous |
| **Rare seizure (tonic)** | Accuracy | **+47 points** 🏆 | Massive improvement |
| **Focal localization** | Precision | **25.4%** 🏆 | vs. 3.5% (CNN) |

**Key components**:

| Component | Function |
|-----------|----------|
| **Graph structure** | Captures non-Euclidean brain topology 🧠 |
| **Self-supervised pre-training** | Predicts future EEG → learns robust representations 💪 |
| **Interpretability module** | Occlusion-based analysis → localizes seizures 🔍 |

- _Why necessary_: 🏥 Existing models have **critical gaps**:
  - Treat brain as flat grid → **lose spatial relationships**
  - Fail on rare seizures → **data imbalance problem**
  - Black box predictions → **no clinical insight** where seizures occur
  
  Need **graph + self-supervised + interpretable** approach ✅
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 **Propose Graph-based RNN with self-supervised pre-training**:
  - 🔵 **Nodes**: EEG electrodes
  - 🔗 **Edges**: Electrode relationships
  - 🤖 **Pre-training**: Predict future EEG signals (no labels needed)
  - 🔍 **Interpretability**: Occlusion analysis → localize seizures

- _Method_: 🔧

**1️⃣ Graph Construction** (two types)

| Graph Type | Edge Definition |
|------------|----------------|
| **Distance graph** | Physical distance between electrodes (10-20 system) |
| **Correlation graph** | Functional connectivity (cross-correlation of signals) |

**2️⃣ Model Architecture**

| Component | Description |
|-----------|-------------|
| **Base model** | **DCRNN** (Diffusion Convolutional Recurrent Neural Network) |
| **Spatial modeling** | Diffusion graph convolution |
| **Temporal modeling** | Gated Recurrent Units (GRU) |

**Architecture extends DCRNN** to model **both spatial diffusion and temporal dynamics**

**3️⃣ Self-Supervised Pre-Training**

| Aspect | Details |
|--------|---------|
| **Task** | Predict **next 12 seconds** of EEG |
| **Input** | Current 12/60-second window |
| **Loss** | Mean Absolute Error (MAE) |
| **Learning** | Temporal dynamics + global representations (no labels) |

**4️⃣ Interpretability Module**

**Occlusion-based analysis**:
```
Mask channels/time windows → Observe prediction changes
```

**Two metrics**:

| Metric | Definition |
|--------|-----------|
| **Coverage** | How much of **true seizure region** is detected |
| **Localization** | How **precisely** model pinpoints seizure region |


**Dataset**: 
- **TUSZ** (Temple University Hospital EEG Seizure Corpus)
- **5,499 EEGs**
- **8 seizure types**

**Tasks evaluated**:
1. **Seizure detection** (seizure vs. non-seizure)
2. **Seizure classification** (seizure type prediction)
3. **Localization** (find where seizures occur)

**Baselines**: CNN, LSTM

- _Result_: 📈

**Key findings**:

| Finding | Clinical Value |
|---------|---------------|
| **Graph modeling captures non-Euclidean structure** | Better accuracy + interpretability ✅ |
| **Self-supervised pre-training** | Robust initialization, helps class imbalance 💪 |
| **Correlation graph better for localization** | Than distance-based graph 🔍 |
| **Highlights abnormal brain regions** | Clinically valuable feature 🏥 |

**Advantages**:

| Feature | Traditional Models | Graph RNN + SSL |
|---------|-------------------|-----------------|
| **Spatial modeling** | Euclidean (grid) ❌ | **Non-Euclidean (graph)** ✅ |
| **Rare seizures** | Poor performance | **+47 points** 🏆 |
| **Localization** | 3.5% (CNN) | **25.4%** (7× better) ✅ |
| **Pre-training** | None/Supervised | **Self-supervised** 💪 |
| **Interpretability** | Black box | **Occlusion analysis** 🔍 |

- _Conclusion_: 🎓 **First integration of GNNs and self-supervised learning** for EEG seizure analysis:
  - 🏆 **State-of-the-art performance**: AUROC 0.875 (detection), F1 0.749 (classification)
  - ⚡ **Massive improvement on rare seizures**: +47 points (tonic type)
  - 🔍 **7× better localization**: 25.4% vs. 3.5% (CNN)
  - 🧠 **Respects brain topology**: Graph-based non-Euclidean modeling
  - 💪 **Robust to data imbalance**: Self-supervised pre-training
  - 🏥 **Clinically valuable**: Visualizes seizure onset zones
  
  **Clinical applications**:
  - 🔪 Surgical planning (identify resection targets)
  - 💊 Treatment personalization
  - 📈 Continuous monitoring
  - 🏥 Diagnosis support
  
  **Key innovation**: Two graph types tested:
  - **Distance graph**: Physical electrode proximity
  - **Correlation graph**: Better for **focal seizure localization** ✅
  
  Graph RNN + self-supervised learning = accurate detection + interpretable localization on large public dataset (TUSZ). ⚡🧠🔍
</details>
</details>

---

> [*Temporal Graph Convolutional Networks for Automatic Seizure Detection*], [May 3, 2019]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: ⚡ Graph-based recurrent neural network with self-supervised pre-training for seizure detection, classification, and localization
- _Author_: Ian C. Covert, Balu Krishnan, Imad Najm, Jiening Zhan, Matthew Shore, John Hixson, Ming Jack Po
- _Group_: University of Washington, Cleveland Clinic Foundation, Google AI Healthcare
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 Traditional deep learning models for seizure detection often have **three issues**:

| Issue | Problem |
|-------|---------|
| **1. Euclidean assumption** | Treat EEG as images, **ignore non-Euclidean brain topology** 🧠 |
| **2. Data imbalance** | **Struggle with rare seizure types** 📊 |
| **3. Lack interpretability** | Cannot **explain where seizures occur** ❌ |

- _Focus problem_: 🔍 How to solve **three issues simultaneously**:
  - ✅ Represent EEG's **spatiotemporal and network structure** faithfully
  - ✅ Improve **classification performance** (especially rare seizure classes)
  - ✅ **Quantitatively evaluate** seizure localization ability

- _Why important_: 💡 Seizure localization is **clinically critical** for:
  - 🏥 Diagnosis and treatment planning
  - 🔪 Surgical intervention (identifying seizure onset zones)
  - 💊 Personalized therapy
  
  Current models can't provide this **spatial information** ❌
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **Graph-based RNN + Self-supervised pre-training** achieves:

**Performance**:

| Task | Metric | Result | vs. Baseline |
|------|--------|--------|-------------|
| **Detection** | AUROC | **0.875** 🏆 | Surpasses CNN/LSTM |
| **Classification** | Weighted F1 | **0.749** 🏆 | Higher than previous |
| **Rare seizure (tonic)** | Accuracy | **+47 points** 🏆 | Massive improvement |
| **Focal localization** | Precision | **25.4%** 🏆 | vs. 3.5% (CNN) |

**Key components**:

| Component | Function |
|-----------|----------|
| **Graph structure** | Captures non-Euclidean brain topology 🧠 |
| **Self-supervised pre-training** | Predicts future EEG → learns robust representations 💪 |
| **Interpretability module** | Occlusion-based analysis → localizes seizures 🔍 |

- _Why necessary_: 🏥 Existing models have **critical gaps**:
  - Treat brain as flat grid → **lose spatial relationships**
  - Fail on rare seizures → **data imbalance problem**
  - Black box predictions → **no clinical insight** where seizures occur
  
  Need **graph + self-supervised + interpretable** approach ✅
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 **Propose Graph-based RNN with self-supervised pre-training**:
  - 🔵 **Nodes**: EEG electrodes
  - 🔗 **Edges**: Electrode relationships
  - 🤖 **Pre-training**: Predict future EEG signals (no labels needed)
  - 🔍 **Interpretability**: Occlusion analysis → localize seizures

- _Method_: 🔧

**1️⃣ Graph Construction** (two types)

| Graph Type | Edge Definition |
|------------|----------------|
| **Distance graph** | Physical distance between electrodes (10-20 system) |
| **Correlation graph** | Functional connectivity (cross-correlation of signals) |

**2️⃣ Model Architecture**

| Component | Description |
|-----------|-------------|
| **Base model** | **DCRNN** (Diffusion Convolutional Recurrent Neural Network) |
| **Spatial modeling** | Diffusion graph convolution |
| **Temporal modeling** | Gated Recurrent Units (GRU) |

**Architecture extends DCRNN** to model **both spatial diffusion and temporal dynamics**

**3️⃣ Self-Supervised Pre-Training**

| Aspect | Details |
|--------|---------|
| **Task** | Predict **next 12 seconds** of EEG |
| **Input** | Current 12/60-second window |
| **Loss** | Mean Absolute Error (MAE) |
| **Learning** | Temporal dynamics + global representations (no labels) |

**4️⃣ Interpretability Module**

**Occlusion-based analysis**:
```
Mask channels/time windows → Observe prediction changes
```

**Two metrics**:

| Metric | Definition |
|--------|-----------|
| **Coverage** | How much of **true seizure region** is detected |
| **Localization** | How **precisely** model pinpoints seizure region |


**Dataset**: 
- **TUSZ** (Temple University Hospital EEG Seizure Corpus)
- **5,499 EEGs**
- **8 seizure types**

**Tasks evaluated**:
1. **Seizure detection** (seizure vs. non-seizure)
2. **Seizure classification** (seizure type prediction)
3. **Localization** (find where seizures occur)

**Baselines**: CNN, LSTM

- _Result_: 📈


**Key findings**:

| Finding | Clinical Value |
|---------|---------------|
| **Graph modeling captures non-Euclidean structure** | Better accuracy + interpretability ✅ |
| **Self-supervised pre-training** | Robust initialization, helps class imbalance 💪 |
| **Correlation graph better for localization** | Than distance-based graph 🔍 |
| **Highlights abnormal brain regions** | Clinically valuable feature 🏥 |

**Advantages**:

| Feature | Traditional Models | Graph RNN + SSL |
|---------|-------------------|-----------------|
| **Spatial modeling** | Euclidean (grid) ❌ | **Non-Euclidean (graph)** ✅ |
| **Rare seizures** | Poor performance | **+47 points** 🏆 |
| **Localization** | 3.5% (CNN) | **25.4%** (7× better) ✅ |
| **Pre-training** | None/Supervised | **Self-supervised** 💪 |
| **Interpretability** | Black box | **Occlusion analysis** 🔍 |

- _Conclusion_: 🎓 **First integration of GNNs and self-supervised learning** for EEG seizure analysis:
  - 🏆 **State-of-the-art performance**: AUROC 0.875 (detection), F1 0.749 (classification)
  - ⚡ **Massive improvement on rare seizures**: +47 points (tonic type)
  - 🔍 **7× better localization**: 25.4% vs. 3.5% (CNN)
  - 🧠 **Respects brain topology**: Graph-based non-Euclidean modeling
  - 💪 **Robust to data imbalance**: Self-supervised pre-training
  - 🏥 **Clinically valuable**: Visualizes seizure onset zones
  
  **Clinical applications**:
  - 🔪 Surgical planning (identify resection targets)
  - 💊 Treatment personalization
  - 📈 Continuous monitoring
  - 🏥 Diagnosis support
  
  Graph RNN + self-supervised learning = accurate detection + interpretable localization on large public dataset (TUSZ). ⚡🧠🔍
</details>
</details>

---

> [*EEG-Based Emotion Recognition Using Regularized Graph Neural Networks*], [Date]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 😊 Biologically-inspired regularized graph neural network for robust EEG emotion recognition
- _Author_: Peixiang Zhong, Di Wang, Chunyan Miao,
- _Group_: Nanyang Technological University
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 Most EEG-based emotion recognition models **fail to use real brain structure**:
  - 🧠 Treat EEG channels as **independent points**
  - ❌ Ignore **how brain regions interact**
  - 👥 Struggle with **large variations between people**
  - 📊 Struggle with **noisy emotional labels**

- _Focus problem_: 🔍 How to design a model that:
  1. ✅ Captures **both local and distant relationships** among EEG electrodes
  2. ✅ Stays **stable across different people**
  3. ✅ Handles **labeling errors** (subjects don't feel exact intended emotion)

- _Why important_: 💡 EEG signals reflect **inner emotions** that people cannot easily fake. Making emotion recognition reliable can help:
  - 💬 Human-computer interaction
  - 🏥 Mental health monitoring
  - 🤖 Adaptive systems responding to emotional states
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **Biologically inspired GNN + two regularizers** achieves:

**Performance**:

| Setup | Accuracy |
|-------|----------|
| **Subject-dependent** | **~94%** 🏆 |
| **Subject-independent** | **~85%** 🏆 |

**Key brain regions for emotion** (learned by model):
- 🧠 **Frontal** regions
- 🧠 **Parietal** regions
- 🧠 **Occipital** regions

**Two regularizers**:

| Regularizer | Function |
|-------------|----------|
| **NodeDAT** | Makes each channel's feature **domain-invariant** → reduces subject differences 👥 |
| **EmotionDL** | Replaces one-hot labels with **soft distributions** → reduces noisy/uncertain labels 📊 |

- _Why necessary_: 🏥 Critical gaps in existing models:

| Problem | Issue |
|---------|-------|
| **Ignore connections** | EEG channels are physically and functionally connected → wastes **valuable spatial information** |
| **Subject differences** | Can mislead standard models ❌ |
| **Noisy labels** | Subjects may not feel exact intended emotion |

**Regularization needed** to make system robust ✅
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 **Treat each EEG electrode as node** in brain-like graph:
  - 🔗 **Adjacency matrix** links nearby electrodes
  - 🧠 Connects **symmetric areas** across left/right hemispheres
  
  **Add two regularizers**:

| Regularizer | Purpose |
|-------------|---------|
| **1. NodeDAT** | Node-wise adversarial training → domain-invariant features (reduce subject differences) |
| **2. EmotionDL** | Emotion distribution learning → soft labels (reduce label noise) |

- _Method_: 🔧

**Architecture**:

| Component | Description |
|-----------|-------------|
| **Base model** | **SGC** (Simple Graph Convolutional Network) extended |
| **Adjacency matrix** | Sparse, combining:<br>• **Local**: Distance-based connections<br>• **Global**: Hemisphere-based connections |
| **Regularizers** | NodeDAT + EmotionDL |

**Graph construction**:

**Biologically inspired adjacency**:
```
A = A_local + A_global

where:
• A_local: Nearby electrode connections (distance-based)
• A_global: Symmetric hemisphere connections (left-right)
```

**Training components**:

| Component | Function |
|-----------|----------|
| **NodeDAT** | Node-wise adversarial training for domain invariance |
| **EmotionDL** | Soft emotion label distributions (not one-hot) |


**Datasets**:

| Dataset | Channels | Emotion Classes |
|---------|----------|-----------------|
| **SEED** | 62 | 3 classes |
| **SEED-IV** | 62 | 4 classes |

**Evaluation setups**:
- **Subject-dependent**: Train and test on same subject
- **Subject-independent**: Train on some subjects, test on others

**Baselines**: CNN-based, RNN-based, other GNN-based models

- _Result_: 📈

**Key findings**:

| Finding | Result |
|---------|--------|
| **Outperforms all baselines** | CNN, RNN, other GNN models ✅ |
| **Stable across subjects** | Robust cross-subject performance 👥 |
| **Clear brain activation** | Frontal, parietal, occipital regions 🧠 |
| **Both regularizers help** | Ablation study confirms ✅ |
| **Biological adjacency helps** | vs. simple adjacency matrices ✅ |

**Advantages**:

| Feature | Traditional Models | RGNN |
|---------|-------------------|------|
| **Brain structure** | Ignored ❌ | **Biologically inspired** 🧠 |
| **Subject variability** | Poor handling | **NodeDAT regularization** 👥 |
| **Label noise** | Sensitive | **EmotionDL soft labels** 📊 |
| **Spatial relationships** | Local only | **Local + Global** 🔗 |
| **Subject-dependent** | Good | **~94%** 🏆 |
| **Subject-independent** | Poor | **~85%** ✅ |

- _Conclusion_: 🎓 **Biologically grounded and well-regularized GNN** for EEG emotion recognition:
  - 🧠 **Captures how brain regions cooperate** during emotional responses
  - 👥 **Resists subject differences** (NodeDAT)
  - 📊 **Handles label noise** (EmotionDL)
  - 🏆 **~94% subject-dependent**, **~85% subject-independent**
  - 🔍 **Learns meaningful brain patterns**: Frontal, parietal, occipital regions
  - ✅ **Ablation studies confirm** both regularizers + biological adjacency improve results
  
  Applications:
  - 💬 Human-computer interaction
  - 🏥 Mental health monitoring
  - 🤖 Adaptive affective systems
  - 😊 Emotion-aware interfaces
  
  **Sets new standard** for future affective EEG analysis by combining biological priors with robust regularization. 😊🧠
</details>
</details>

---

> [*GCNs-Net: A Graph Convolutional Neural Network Approach for Decoding Time-resolved EEG Motor Imagery Signals*], [Aug 26, 2022]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 🤖 Graph convolutional networks using functional connectivity for motor imagery EEG decoding in BCI systems
- _Core Author_: Yimin Hou, Shuyue Jia, Xiangmin Lun, Ziqian Hao, Yan Shi, Yang Li, Rui Zeng, Jinglei Lv
- _Core Group_: City University of Hong Kong, University of Sydney
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 Traditional EEG decoding models (CNNs) treat EEG electrodes as **independent points in Euclidean space**. However:
  - 🧠 EEG signals generated by **interconnected brain regions**
  - ❌ Ignoring **functional and topological relationships** weakens decoding accuracy

- _Focus problem_: 🔍 How to use **functional connectivity** among EEG electrodes (reflecting real brain network dynamics) to improve classification of **time-resolved motor imagery** signals for:
  - 👤 Individual-level EEG data
  - 👥 Group-level EEG data

- _Why important_: 💡 Accurate motor imagery decoding essential for **BCI systems**:
  - 🦾 Control external devices (prosthetic limbs, wheelchairs)
  - 🏥 Using **only brain signals**
  - 🎯 Improving accuracy and stability → **more practical real-world medical applications**
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **GCNs-Net** learns both **spatial and temporal patterns** by building **graph Laplacian** based on electrode correlations:

**Performance**:

| Level | Dataset | Accuracy |
|-------|---------|----------|
| **Subject-level** | PhysioNet | **98.72%** 🏆 |
| **Subject-level** | High Gamma | **96.24%** 🏆 |
| **Group-level** | Average | **88-89%** 🏆 |

**Key features**:
- ✅ **Superior performance** and robustness
- ✅ **Stable** across 10-fold validation
- ✅ **Outperforms CNN/RNN** (p < 0.05)
- ✅ **Robust to individual differences**
- ✅ **Scales to large datasets** (tested up to 100 subjects)

- _Why necessary_: 🏥 **Brain is not a grid but a complex network**:

| Model Type | Brain Representation | Problem |
|------------|---------------------|---------|
| **Standard CNN** | Euclidean grid ❌ | Cannot capture **long-range relationships** between brain regions |
| **GCNs-Net** | Graph network ✅ | Reflects **brain's real connectivity** → improves generalization |
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 **Represent EEG electrodes as graph nodes**:
  - 🔵 **Nodes**: EEG electrodes
  - 🔗 **Edges**: Functional connections (computed using **absolute Pearson correlation coefficients**)
  
  **Apply spectral graph convolution** with Chebyshev polynomial approximation → efficient feature extraction → graph pooling → softmax classifier

- _Method_: 🔧

**Architecture components**:

| Component | Description |
|-----------|-------------|
| **Graph Laplacian** | Represents EEG **topological relationships** |
| **GCN layers** | Capture **spatial dependencies** |
| **Pooling layers** | Reduce dimensionality |
| **Chebyshev spectral filtering** | Reduce computation while preserving **local structure** ⚡ |

**Graph construction**:
```
Edge weights = Absolute Pearson correlation between electrode signals
Graph Laplacian = Normalized adjacency matrix representing brain connectivity
```


**Datasets**:

| Dataset | Task | Classes |
|---------|------|---------|
| **PhysioNet** | Motor imagery | 4 classes (left hand, right hand, both hands, both feet) |
| **High Gamma** | Motor imagery | 4 classes |

**Evaluation**:

| Setup | Method |
|-------|--------|
| **Subject-level** | Individual performance |
| **Group-level** | Cross-subject generalization |
| **Validation** | **10-fold cross-validation** (stability) |

**Technical optimization**:
- **Chebyshev polynomial approximation**: Efficient spectral filtering
- **Graph pooling**: Dimensionality reduction
- **Functional connectivity**: Absolute Pearson correlation

- _Result_: 📈

**Robustness findings**:

| Test | Result |
|------|--------|
| **Individual differences** | High robustness ✅ |
| **Large datasets** | Tested up to **100 subjects** ✅ |
| **Cross-validation** | Stable performance across folds |

**Key advantages**:

| Feature | CNN | GCNs-Net |
|---------|-----|----------|
| **Brain representation** | Euclidean grid ❌ | **Graph network** ✅ |
| **Long-range relationships** | Cannot capture ❌ | **Captured** ✅ |
| **Subject-level** | Lower | **98.72%** 🏆 |
| **Group-level** | Lower | **88-89%** 🏆 |
| **Robustness** | Limited | **High** (100 subjects) ✅ |
| **Computational efficiency** | Standard | **Chebyshev approximation** ⚡ |


- _Conclusion_: 🎓 **GCNs-Net successfully integrates brain's functional topology** into deep learning:
  - 🏆 **98.72% subject-level** (PhysioNet), **96.24%** (High Gamma)
  - 👥 **88-89% group-level** (stable across subjects)
  - ⚡ **Efficient**: Chebyshev spectral filtering
  - 🔍 **Interpretable**: Graph structure reflects brain connectivity
  - 💪 **Robust**: Handles individual differences, scales to 100 subjects
  - 📊 **Statistically significant**: p < 0.05 vs. CNN/RNN
  
  Applications:
  - 🦾 Prosthetic limb control
  - ♿ Wheelchair navigation
  - 🏥 Rehabilitation systems
  - 🤖 Real-time BCI
  
  **Important step toward**: Real-time, robust BCI systems capable of **generalizing across individuals and tasks** by modeling brain's true network structure. 🤖🧠
</details>
</details>

---

> [*EEG-GCNN- Augmenting Electroencephalogram-based Neurological Disease Diagnosis using a Domain-guided Graph Convolutional Neural Network*], [Nov 17, 2020]:
<details>
<summary><strong>V0:</strong></summary>
<details>
<summary><strong>Bases</strong></summary>

- _Topic_: 🏥 Domain-guided graph convolutional network for detecting hidden neurological abnormalities in visually "normal" EEG
- _Core Author_: Neeraj Wagh, Yogatheesan Varatharajah
- _Core Group_: University of Illinois at Urbana-Champaign
</details>

<details>
<summary><strong>Problems</strong></summary>

- _Main problem_: 🎯 In clinical practice, EEG is **primary tool for diagnosing neurological diseases**. However:
  - 👨‍⚕️ **Expert visual diagnosis**: Only **~50% sensitive**
  - ❌ Many "normal-looking" EEGs from **diseased patients** mistakenly labeled as **healthy**

- _Focus problem_: 🔍 Can a data-driven model **detect hidden abnormalities** in EEG signals that **appear normal to human experts**, effectively distinguishing:
  - 🏥 EEGs from **neurologically diseased patients**
  - ✅ vs. EEGs from **healthy individuals**

- _Why important_: 💡 **Missed diagnoses** lead to:
  - ⏰ **Delayed clinical intervention**
  - ⚠️ Increased patient **risk of injury or comorbidities**
  
  AI model identifying disease-related patterns **earlier and more reliably** → greatly improve medical outcomes + reduce clinician burden ✅
</details>

<details>
<summary><strong>Motivations</strong></summary>

- _Main finding/insight_: 📊 **EEG-GCNN** (domain-guided graph CNN) captures **both spatial and functional connectivity**:

| Connectivity Type | Measurement |
|------------------|-------------|
| **Spatial** | Electrode distance (geodesic) |
| **Functional** | Signal coherence (spectral) |

**Performance**:

| Model | AUC | vs. Human Experts |
|-------|-----|-------------------|
| **Human experts** | ~0.50 sensitivity | Baseline ❌ |
| **Random Forest** | 0.80 | Better |
| **FCNN** | 0.71 | Lower |
| **EEG-GCNN** | **0.90** 🏆 | **Far superior** ✅ |

**Key achievement**: Successfully **differentiates "normal" EEGs** from:
- 🏥 Neurological patients (hidden abnormalities)
- ✅ Healthy individuals

- _Why necessary_: 🏥 Conventional approaches have **critical limitations**:

| Approach | Problem |
|----------|---------|
| **CNN** | Treats channels as **independent** ❌ |
| **Handcrafted features** | Miss brain **network structure** ❌ |
| **Both** | Fail to represent **true brain connectivity** |

**Graph modeling** mirrors brain's **interconnected regions** → improves diagnostic sensitivity ✅
</details>

<details>
<summary><strong>Solutions</strong></summary>

- _Idea_: 💭 Represent each EEG recording as **fully connected weighted graph**:

| Component | Description |
|-----------|-------------|
| **Nodes** | EEG channels |
| **Edges** | Combine **spatial distances** + **functional coherences** |

Use **spectral graph convolutions** (Kipf & Welling, 2016) → learn connectivity-aware features → aggregate to graph-level embeddings → classify

- _Method_: 🔧

**Datasets**:

| Dataset | Description | Size |
|---------|-------------|------|
| **TUH EEG Corpus (TUAB)** | "Normal" EEGs from **neurological patients** | 1,385 EEGs |
| **MPI LEMON** | EEGs from **healthy participants** | 208 EEGs |

**Feature extraction**:
- **Power Spectral Density (PSD)** across **six frequency bands**:
  - δ (delta)
  - θ (theta)
  - α (alpha)
  - β_L (low beta)
  - β_H (high beta)
  - γ (gamma)

**Graph construction**:

| Component | Method |
|-----------|--------|
| **Spatial adjacency** | Geodesic distance between electrodes |
| **Functional adjacency** | Spectral coherence between signals |
| **Combined adjacency** | Weighted combination of both |


**Training setup**:

| Component | Details |
|-----------|---------|
| **Cross-validation** | **10-fold** |
| **Class imbalance** | **Weighted loss** |
| **Model variants** | Shallow + Deep EEG-GCNN |

**Baselines**:
- Random Forest
- Fully Connected Neural Network (FCNN)

- _Result_: 📈

**Key findings**:

| Finding | Clinical Significance |
|---------|----------------------|
| **Detects subtle deviations** | That experts **often miss** 🔍 |
| **Hidden neurological patterns** | Exist in "normal" EEGs ✅ |
| **Graph-based modeling** | Reflects **neurophysiological connectivity** 🧠 |

**Advantages**:

| Feature | Traditional Methods | EEG-GCNN |
|---------|-------------------|----------|
| **Brain representation** | Independent channels ❌ | **Graph (spatial + functional)** ✅ |
| **Hidden patterns** | Cannot detect ❌ | **Detects** 🔍 |
| **AUC** | 0.71-0.80 | **0.90** 🏆 |
| **vs. Human experts** | ~50% sensitivity | **Far superior** ✅ |
| **Generalization** | Limited | **Strong** (10-fold validated) |
| **Interpretability** | Limited | **t-SNE clear separation** 📊 |

- _Conclusion_: 🎓 **EEG-GCNN introduces novel graph-based approach** reflecting neurophysiological connectivity:
  - 🔍 **Identifies subtle abnormalities** invisible to experts
  - 🏆 **AUC = 0.90** vs. 0.80 (RF), 0.71 (FCNN), ~0.50 (human experts)
  - 🧠 **Models brain's true structure**: Spatial + functional connectivity
  - 📊 **Clear separation**: t-SNE validates healthy vs. diseased embeddings
  - ✅ **Proves existence** of hidden neurological patterns in "normal" EEGs
  
  Applications:
  - 🏥 Neurological disease screening
  - 🔍 Early diagnosis support
  - 👨‍⚕️ Clinical decision assistance
  - 📊 Objective EEG assessment
  
  **Path toward**: Early, automated, and reliable diagnosis of neurological diseases by detecting patterns human experts cannot see. 🏥🧠🔍
</details>
</details>

---
