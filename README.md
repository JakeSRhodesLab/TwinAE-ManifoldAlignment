# Graph Manifold Alignment

**Guided Manifold Alignment with Geometry-Regularized Twin Autoencoders** - A novel approach using Geometric Regularized Autoencoders (GRAE) for multimodal data alignment.

[![MMAI 2025](https://img.shields.io/badge/MMAI%202025-Accepted-brightgreen)](https://icdmw25mmai.github.io/)
[![IEEE ICDM](https://img.shields.io/badge/IEEE%20ICDM-Workshop-blue)](https://icdm2025.org/)

## 📄 Research Paper

**Accepted at MMAI 2025 Workshop @ IEEE ICDM 2025**

This repository contains the implementation of our paper "Guided Manifold Alignment with Geometry-Regularized Twin Autoencoders" accepted for presentation at the 5th IEEE International Workshop on Multimodal AI (MMAI) at ICDM 2025.

## Overview

This package provides a comprehensive implementation of **Twin Autoencoders** using **Geometric Regularized Autoencoders (GRAE)** for manifold alignment, along with multiple baseline alignment methods for comparison:

### 🎯 **Main Contribution: Guided Manifold Alignment**
- **GRAE (Geometry-Regularized Autoencoders)**: Novel autoencoder architecture with geometric regularization
- **Anchor Guidance**: Leverages known correspondences to guide the alignment process
- **Twin Architecture**: Paired autoencoders for bidirectional domain translation
- **Geometric Preservation**: Maintains local neighborhood structure during alignment
- **Multimodal Integration**: Support for various data modalities (text, images, tabular data)

### 🔧 **Baseline Alignment Methods**
- **DTA (Domain Transfer Analysis)**: Statistical domain adaptation
- **JLMA (Joint Latent Manifold Alignment)**: Joint embedding space learning
- **Procrustes**: Classical orthogonal transformation alignment
- **MAGAN**: Manifold Alignment with Generative Adversarial Networks
- **MALI**: Manifold Alignment Learning Interface
- **MASH**: Manifold Alignment via Stochastic Hashing
- **SSMA**: Semi-Supervised Manifold Alignment

### 📊 **Research Applications**
- **ADNI Dataset Analysis**: Alzheimer's Disease Neuroimaging Initiative data processing
- **Multimodal Data Fusion**: Cross-modal learning and alignment
- **Biomedical Applications**: Neuroimaging and clinical data integration

## 🌟 Research Highlights

### **Novel Contributions**
- 🎯 **Guided Alignment Framework**: Novel approach combining geometric regularization with anchor guidance
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

### Install from source

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

#### **Geometry-Regularized Autoencoders (GRAE)**
- **Architecture**: Deep autoencoder with geometric regularization in latent space
- **Innovation**: Enforces geometric consistency between original and embedded spaces
- **Loss Function**: Combined reconstruction + geometric regularization
- **Applications**: Manifold learning with preserved geometric structure

#### **Anchor Guidance System**
- **Framework**: Intelligent use of known correspondences to guide alignment
- **Advantage**: Leverages partial correspondence information for improved quality
- **Implementation**: GRAEAnchor class with anchor-specific loss terms
- **Performance**: Superior alignment quality through guided learning

#### **Twin Architecture**
- **Concept**: Paired geometry-regularized autoencoders for bidirectional translation
- **Benefit**: Enables symmetric A↔B domain transformations
- **Consistency**: Cycle consistency loss for stable alignment
- **Robustness**: Handles domain asymmetries effectively

### **Baseline Alignment Methods**
- **DTA (Domain Transfer Analysis)**: Statistical alignment with optimal transport
- **JLMA (Joint Latent Manifold Alignment)**: Spectral embedding alignment
- **Procrustes**: Classical orthogonal transformation alignment
- **MAGAN**: Generative adversarial approach to manifold alignment
- **MALI**: Semi-supervised manifold alignment interface
- **MASH**: Stochastic hashing for scalable alignment
- **SSMA**: Semi-supervised approach with unlabeled data utilization

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
- **GRAE Tutorial**: Basic geometric regularized autoencoder usage
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

```bibtex
@inproceedings{rhodes2025guided,
  title={Guided Manifold Alignment with Geometry-Regularized Twin Autoencoders},
  author={Rhodes, Jake and [Additional Authors]},
  booktitle={Proceedings of the 5th IEEE International Workshop on Multimodal AI (MMAI)},
  year={2025},
  organization={IEEE},
  venue={ICDM 2025 Workshop},
  url={https://github.com/JakeSRhodesLab/Graph-Manifold-Alignment}
}
```

For the software package:

```bibtex
@software{graph_manifold_alignment,
  author = {Jake Rhodes Lab},
  title = {Graph Manifold Alignment: Geometry-Regularized Twin Autoencoders Implementation},
  year = {2025},
  url = {https://github.com/JakeSRhodesLab/Graph-Manifold-Alignment},
  note = {Implementation of "Guided Manifold Alignment with Geometry-Regularized Twin Autoencoders" - MMAI 2025}
}
```

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
