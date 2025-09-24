# Graph Manifold Alignment

**Guided Manifold Al### **Novel Contributions**
- 🎯 **Out-of-Sample Extension**: Addresses key limitation of traditional MA methods that cannot generalize to unseen data
- 🔗 **Geometry-Regularized Architecture**: Twin autoencoders with multitask learning combining reconstruction and embedding prediction
- ⚓ **Pre-trained Alignment Integration**: Incorporates existing MA models as guidance within the autoencoder framework
- 🔄 **Cross-Domain Mapping**: Enables direct translation between domains through decoder swapping mechanism
- 📊 **Structured Regularization**: Three-component loss function: reconstruction, alignment, and anchor losses

### **Experimental Results**
- ✅ **Superior Embedding Consistency**: High Mantel correlations (0.68-0.80) for JLMA, SPUD, MASH regularization
- 🧠 **Clinical Validation**: Successfully translates between ADAS-Cog 13 and FAQ assessments (RMSE: 1.12-1.48)
- 📈 **Cross-Domain Mapping**: Outperforms MAGAN, DTA, and MASH in mapping accuracy across test datasets
- 🎯 **Predictive Enhancement**: Consistent performance improvements in downstream classification tasks

[![MMAI 2025](https://img.shields.io/badge/MMAI%202025-Accepted-brightgreen)](https://icdmw25mmai.github.io/)
[![IEEE ICDM](https://img.shields.io/badge/IEEE%20ICDM-Workshop-blue)](https://icdm2025.org/)

## 📄 Research Paper

**Accepted at MMAI 2025 Workshop @ IEEE ICDM 2025**

This repository contains the implementation of our paper "Guided Manifold Alignment with Geometry-Regularized Twin Autoencoders" accepted for presentation at the 5th IEEE International Workshop on Multimodal AI (MMAI) at ICDM 2025.

**Authors:** Jake S. Rhodes¹, Adam G. Rustad², Marshall S. Nielsen¹, Morgan McClellan¹, Dallan Gardner¹, Dawson Hedges³  
¹Department of Statistics, Brigham Young University  
²Department of Computer Science, Brigham Young University  
³Department of Psychology, Brigham Young University

**Abstract:** Manifold alignment (MA) involves a set of techniques for learning shared representations across domains, yet many traditional MA methods are incapable of performing out-of-sample extension, limiting their real-world applicability. We propose a guided representation learning framework leveraging a geometry-regularized twin autoencoder (AE) architecture to enhance MA while enabling generalization to unseen data. Our method enforces structured cross-modal mappings to maintain geometric fidelity in learned embeddings. By incorporating a pre-trained alignment model and a multitask learning formulation, we improve cross-domain generalization and representation robustness while maintaining alignment fidelity. We evaluate our approach using several MA methods, showing improvements in embedding consistency, information preservation, and cross-domain transfer. Additionally, we apply our framework to Alzheimer's disease diagnosis, demonstrating its ability to integrate multi-modal patient data and enhance predictive accuracy in cases limited to a single domain by leveraging insights from the multimodal problem.

**Keywords:** manifold alignment, regularized autoencoders, out-of-sample extension, multimodal methods

## Overview

This package provides a comprehensive implementation of **Twin Autoencoders** using **Geometry Regularized Autoencoders (GRAE)** for manifold alignment, along with multiple baseline alignment methods for comparison:

### 🎯 **Main Contribution: Guided Manifold Alignment**
- **Geometry-Regularized Twin Autoencoders**: Novel twin autoencoder architecture with geometry regularization for manifold alignment
- **Out-of-Sample Extension**: Enables generalization to unseen data points, addressing a key limitation of traditional MA methods
- **Guided Representation Learning**: Incorporates pre-trained alignment models within a multitask learning formulation
- **Cross-Modal Mappings**: Enforces structured mappings between domains while maintaining geometric fidelity
- **Robust Generalization**: Improves cross-domain generalization and representation robustness

### 🔧 **Baseline Alignment Methods**
- **DTA (Diffusion Transport Alignment)**: Integrates diffusion processes with regularized optimal transport
- **JLMA (Joint Laplacian Manifold Alignment)**: Joint graph Laplacian constructed from domain-specific similarity matrices
- **MAPA (Manifold Alignment via Procrustes Analysis)**: Uses Laplacian Eigenmaps with Procrustes analysis for alignment
- **MAGAN (Manifold Alignment GAN)**: Semi-supervised manifold alignment using generative adversarial networks
- **MASH (Manifold Alignment via Stochastic Hopping)**: Graph-based method using diffusion operators and random walks
- **SPUD (Shortest Paths on Union of Domains)**: Estimates geodesic distances via shortest paths in combined graphs
- **SSMA (Semi-Supervised Manifold Alignment)**: Classical semi-supervised approach using partial correspondences

### 📊 **Research Applications**
- **Alzheimer's Disease Assessment**: Translation between ADAS-Cog 13 and FAQ cognitive/functional assessments
- **Biomedical Data Integration**: Multi-modal patient data alignment for enhanced diagnosis
- **Cross-Domain Prediction**: Leveraging multimodal insights when only single-domain data is available
- **Clinical Decision Support**: Predicting functional impairments from cognitive evaluations

## 🌟 Research Highlights

### **Novel Contributions**
- 🎯 **Guided Alignment Framework**: Novel approach combining Geometry regularization with anchor guidance
- 🔗 **Geometry-Regularized Autoencoders**: Preserves local neighborhood structure during alignment
- ⚓ **Intelligent Guidance System**: Leverages known correspondences to improve alignment quality
- 🔄 **Twin Architecture**: Enables symmetric bidirectional domain transformation
- 📊 **Multimodal Integration**: Handles heterogeneous data types and dimensionalities

### **Experimental Results**
- ✅ **Superior Performance**: Outperforms traditional alignment methods on benchmark datasets
- 🧠 **Biomedical Applications**: Validated on ADNI neuroimaging data
- � **Scalability**: Efficient processing of high-dimensional multimodal data
- 🎯 **Robustness**: Stable performance across different data distributions

## Features

- 🔧 **State-of-the-art Methods**: Novel twin autoencoder approach + comprehensive baseline comparisons
- 📊 **Multimodal Data Support**: ADNI neuroimaging, tabular data, images, and custom formats
- 🧠 **Deep Learning Architecture**: PyTorch-based GRAE implementation with GPU acceleration
- 📈 **Comprehensive Evaluation**: Multiple metrics, visualization tools, and statistical analysis
- 🔬 **Research Reproducibility**: Complete experimental pipeline and notebook demonstrations
- ⚡ **High Performance**: Optimized for large-scale multimodal datasets

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Direct installation from GitHub

```bash
# Install latest version directly from GitHub
pip install git+https://github.com/JakeSRhodesLab/Graph-Manifold-Alignment.git

# Or install a specific branch/tag
pip install git+https://github.com/JakeSRhodesLab/Graph-Manifold-Alignment.git@main
```

**Note**: This will automatically install all dependencies including the `mashspud` package for MASH and SPUD functionality.

### Install from source (for development)

```bash
git clone https://github.com/JakeSRhodesLab/Graph-Manifold-Alignment.git
cd Graph-Manifold-Alignment
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Twin Autoencoders (GRAE)

```python
import numpy as np
from graph_manifold_alignment.autoencoder import GRAEAnchor
from sklearn.manifold import TSNE

# Load your data domains
domain_A = np.random.rand(100, 50)  # Source domain
domain_B = np.random.rand(100, 40)  # Target domain

# Compute shared embedding space
tsne = TSNE(n_components=2)
shared_embedding = tsne.fit_transform(np.vstack([domain_A, domain_B]))

# Define known anchor correspondences (if available)
anchors = np.array([[0, 0], [1, 1], [2, 2]])  # [source_idx, target_idx]

# Train Twin Autoencoders
autoencoder_A = GRAEAnchor(lam=100, anchor_lam=50, n_components=2)
autoencoder_A.fit(domain_A, shared_embedding[:100], anchors)

autoencoder_B = GRAEAnchor(lam=100, anchor_lam=50, n_components=2)
autoencoder_B.fit(domain_B, shared_embedding[100:], anchors)

# Transform data to aligned space
aligned_A = autoencoder_A.transform(domain_A)
aligned_B = autoencoder_B.transform(domain_B)
```

### Baseline Methods

```python
# Use traditional alignment methods for comparison
from graph_manifold_alignment.alignment_methods import DTA_andres, JLMA, MAGAN

# DTA (Domain Transfer Analysis)
dta = DTA_andres()
result = dta.fit_transform(domain_A, domain_B, anchors)

# JLMA (Joint Latent Manifold Alignment)
jlma = JLMA()
aligned_data = jlma.fit(domain_A, domain_B, anchors)
```

## Project Structure

```
Graph-Manifold-Alignment/
├── src/graph_manifold_alignment/    # Main package
│   ├── alignment_methods/           # Alignment algorithms
│   ├── helpers/                     # Utility functions
│   ├── adni/                       # ADNI-specific tools
│   └── autoencoder/                # Autoencoder models
├── notebooks/                       # Jupyter notebooks
│   ├── demonstrations/             # Method demonstrations
│   ├── experiments/                # Experimental analysis
│   └── analysis/                   # Data analysis
├── tests/                          # Unit tests
├── data/                           # Datasets
├── outputs/                        # Generated results
├── docs/                           # Documentation
└── scripts/                        # Utility scripts
```

## 🧠 Available Methods

### **Guided Manifold Alignment (Our Novel Approach)**

#### **Geometry-Regularized Twin Autoencoders**
- **Architecture**: Two independent autoencoders (AE_X, AE_Y) with encoder-decoder pairs (f_X, g_X) and (f_Y, g_Y)
- **Innovation**: Multitask learning combining reconstruction with pre-aligned embedding prediction
- **Loss Function**: ℒ = ℒ_recon + λℒ_align + ℒ_anchor
- **Applications**: Manifold alignment with out-of-sample extension capability

#### **Three-Component Loss System**
- **Reconstruction Loss**: Ensures accurate data reconstruction from embedding space
- **Alignment Loss**: Enforces consistency with pre-computed aligned embeddings
- **Anchor Loss**: Maintains alignment fidelity for known correspondences
- **Regularization**: λ parameter controls geometric alignment strength

#### **Cross-Domain Translation**
- **Mechanism**: Decoder swapping enables A→B and B→A domain translation
- **Process**: Encode with domain-specific encoder, decode with opposite decoder
- **Advantage**: Direct cross-modal mapping without retraining
- **Flexibility**: Compatible with any underlying MA method for guidance

### **Baseline Alignment Methods**
- **DTA (Diffusion Transport Alignment)**: Integrates diffusion processes with regularized optimal transport
- **JLMA (Joint Laplacian Manifold Alignment)**: Joint graph Laplacian from domain-specific similarity matrices
- **MAPA (Manifold Alignment via Procrustes Analysis)**: Laplacian Eigenmaps with Procrustes analysis
- **MAGAN (Manifold Alignment GAN)**: Generative adversarial networks with correspondence loss
- **MASH (Manifold Alignment via Stochastic Hopping)**: Diffusion-based approach with random walks
- **SPUD (Shortest Paths on Union of Domains)**: Geodesic distance estimation via shortest paths
- **SSMA (Semi-Supervised Manifold Alignment)**: Classical approach with partial correspondences

## 📚 Usage Examples & Documentation

### Notebooks
- **`notebooks/demonstrations/`** - Method demonstrations and tutorials
- **`notebooks/experiments/`** - Research experiments and comparisons  
- **`notebooks/analysis/`** - Data analysis and evaluation notebooks

### Documentation
- **[Paper Overview](docs/PAPER_OVERVIEW.md)** - Detailed research contribution summary
- **[Applications](docs/APPLICATIONS.md)** - Use cases across domains
- **[Research Paper Draft](TwinAEDraft.pdf)** - "Guided Manifold Alignment with Geometry-Regularized Twin Autoencoders" (MMAI 2025)

### Key Demonstrations
- **GRAE Tutorial**: Basic Geometry regularized autoencoder usage
- **Twin Architecture**: Bidirectional domain alignment examples
- **ADNI Analysis**: Neuroimaging data processing pipeline
- **Comparative Study**: Performance vs. baseline methods

## Data

The package supports various data formats including:
- Classification datasets (CSV format)
- Regression datasets 
- ADNI neuroimaging data
- Custom graph data formats

Sample datasets are provided in the `data/` directory.

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Citation

If you use this package in your research, please cite our MMAI 2025 paper:

CITATION TBD AFTER PUBLICATION


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- **Repository**: [Graph-Manifold-Alignment](https://github.com/JakeSRhodesLab/Graph-Manifold-Alignment)
- **Issues**: [GitHub Issues](https://github.com/JakeSRhodesLab/Graph-Manifold-Alignment/issues)

## 🙏 Acknowledgments

- **Jake Rhodes Lab** - Primary research and development
- **MMAI 2025 Workshop** - Platform for presenting this research  
- **IEEE ICDM 2025** - Conference support and academic venue
- **Multimodal AI Community** - Feedback and collaboration
- **ADNI Initiative** - Neuroimaging data for biomedical applications
- **Open Source Community** - Supporting libraries and frameworks

### Workshop Information
- **Workshop**: [5th IEEE International Workshop on Multimodal AI (MMAI)](https://icdmw25mmai.github.io/)
- **Conference**: [IEEE International Conference on Data Mining (ICDM) 2025](https://icdm2025.org/)
- **Focus**: Advancing Multimodal AI Research and Applications
- **Topics**: Multimodal learning, data fusion, cross-modal applications
