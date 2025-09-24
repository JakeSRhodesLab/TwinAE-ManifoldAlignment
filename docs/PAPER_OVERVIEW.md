# Guided Manifold Alignment with Geometry-Regularized Twin Autoencoders

**Accepted at MMAI 2025 Workshop @ IEEE ICDM 2025**

**Authors:** Jake S. Rhodes¹, Adam G. Rustad², Marshall S. Nielsen¹, Morgan McClellan¹, Dallan Gardner¹, Dawson Hedges³  
¹Department of Statistics, Brigham Young University  
²Department of Computer Science, Brigham Young University  
³Department of Psychology, Brigham Young University

## Abstract

Manifold alignment (MA) involves a set of techniques for learning shared representations across domains, yet many traditional MA methods are incapable of performing out-of-sample extension, limiting their real-world applicability. We propose a guided representation learning framework leveraging a geometry-regularized twin autoencoder (AE) architecture to enhance MA while enabling generalization to unseen data. Our method enforces structured cross-modal mappings to maintain geometric fidelity in learned embeddings. By incorporating a pre-trained alignment model and a multitask learning formulation, we improve cross-domain generalization and representation robustness while maintaining alignment fidelity.

## Key Contributions

### 1. **Out-of-Sample Extension for Manifold Alignment**
- Addresses fundamental limitation of traditional MA methods that require recomputation for new data
- Enables generalization to unseen data points through parametric autoencoder functions
- Prevents data leakage in machine learning applications by separating training and test embeddings

### 2. **Geometry-Regularized Twin Autoencoder Architecture**
- Two independent autoencoders (AE_X, AE_Y) with encoder-decoder pairs
- Multitask learning combining reconstruction with pre-aligned embedding prediction
- Cross-domain translation via decoder swapping mechanism

### 3. **Pre-trained Alignment Integration**
- Incorporates existing MA models (JLMA, DTA, MASH, etc.) as guidance within framework
- Compatible with any semi-supervised alignment method for regularization
- Preserves alignment fidelity while enabling extension capabilities

### 4. **Clinical Application Validation**
- Alzheimer's disease assessment using ADNI data
- Translation between ADAS-Cog 13 and FAQ cognitive/functional evaluations
- Demonstrates practical utility in biomedical domain with asymmetric data costs

## Technical Innovation

### Loss Function Design
The twin autoencoder loss combines three key components:

1. **Reconstruction Loss**: ℒ_recon = (1/n_x) Σ ||x_i - g_X(f_X(x_i))||²
2. **Alignment Loss**: ℒ_align = (1/n_x) Σ ||f_X(x_k) - e_{x_k}||² (guides to pre-aligned embedding)
3. **Anchor Loss**: ℒ_anchor = (1/n_A) Σ ||f_X(x_j) - e_{a_{x_j}}||² (enforces known correspondences)

```
ℒ = ℒ_recon + λℒ_align + ℒ_anchor
```

### Architecture Benefits
- **Out-of-Sample Extension**: First MA approach enabling natural generalization to unseen data
- **Method Agnostic**: Compatible with any underlying MA method (JLMA, DTA, MASH, SPUD, etc.)
- **Direct Cross-Domain Mapping**: Decoder swapping enables A→B and B→A translation
- **Geometric Fidelity**: Maintains alignment quality while enabling extension capabilities

## Experimental Validation

### Datasets
- **UCI Repository**: 22 classification and 20 regression datasets for comprehensive evaluation
- **ADNI Clinical Data**: ADAS-Cog 13 and FAQ assessments from ~1,200 patients
- **Domain Splits**: Random, skewed importance, even importance, distortion, and rotation splits

### Performance Metrics
- **Mantel Correlation**: Measures preservation of pairwise distance structure (0.17-0.80 across methods)
- **Cross-Domain Mapping Error**: MSE between mapped and true corresponding points
- **Downstream Task Performance**: k-NN classification accuracy on aligned embeddings
- **Clinical Translation Accuracy**: RMSE for ADAS-Cog ↔ FAQ translation (1.12-1.48)

### Baseline Comparisons
Evaluation against state-of-the-art MA methods:
- **JLMA (Joint Laplacian Manifold Alignment)**: Best twin AE performance (r=0.80)
- **SPUD (Shortest Paths on Union of Domains)**: Strong performance (r=0.72)
- **MASH (Manifold Alignment via Stochastic Hopping)**: Good alignment (r=0.68)
- **MAGAN (Manifold Alignment GAN)**: Moderate performance (r=0.54)
- **DTA (Diffusion Transport Alignment)**: Lower correlation (r=0.25)
- **MAPA (Manifold Alignment via Procrustes)**: Weakest performance (r=0.17)

## Research Impact

### Scientific Contributions
- **Out-of-Sample Extension**: First MA framework enabling natural generalization to unseen data
- **Method Integration**: Novel approach combining pre-trained MA models with autoencoder learning
- **Cross-Domain Translation**: Direct mapping between domains without retraining underlying MA methods
- **Clinical Validation**: Demonstrated utility in Alzheimer's disease assessment translation

### Applications
- **Clinical Decision Support**: Predicting functional impairments from cognitive assessments
- **Biomedical Data Integration**: Multi-modal patient data alignment for enhanced diagnosis
- **Cost-Effective Assessment**: Leveraging expensive domain insights when only cheaper data available
- **Multimodal Machine Learning**: Enabling ML pipelines with proper train/test separation

## Workshop Presentation

**Venue**: 5th IEEE International Workshop on Multimodal AI (MMAI 2025)  
**Conference**: IEEE International Conference on Data Mining (ICDM) 2025  
**Focus**: Advancing Multimodal AI Research and Applications

The workshop provides an ideal platform for presenting our multimodal alignment approach, connecting with the broader AI community interested in cross-modal learning and data fusion.

## Implementation

This repository provides:
- Complete implementation of Twin Autoencoders with GRAE
- Comprehensive baseline method comparisons
- Experimental notebooks and demonstrations
- Evaluation tools and visualization utilities
- Documentation and usage examples

## Future Directions

- Extension to N>2 domains with multiple twin autoencoders
- Integration with modern deep learning architectures for complex data types
- Theoretical analysis of anchor percentage requirements and convergence properties
- Online/streaming alignment scenarios for dynamic data
- Large-scale deployment optimizations for clinical systems

## Keywords

manifold alignment, regularized autoencoders, out-of-sample extension, multimodal methods

---

*This research addresses a fundamental limitation of manifold alignment methods by enabling out-of-sample extension while maintaining alignment fidelity, with practical validation in Alzheimer's disease assessment.*
