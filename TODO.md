# TODO: Manual Repository Finalization Tasks

This document outlines the remaining manual tasks needed to complete the repository setup and prepare it for publication alongside the MMAI 2025 paper.

## 🔧 **Immediate Technical Tasks**

### 1. **Move Alignment Method Files**
The alignment method files need to be moved from the old structure to the new package structure:

```bash
# Move alignment methods to new structure
cp Python_Files/AlignmentMethods/*.py src/graph_manifold_alignment/alignment_methods/

# Verify all files are moved:
# - DTA_andres.py
# - jlma.py  
# - ma_procrustes.py
# - MAGAN.py
# - mali.py
# - MASH_MD.py
# - ssma.py
```

### 2. **Fix Import Statements**
Update all import statements in the moved files to use the new package structure:

**Files to update:**
- All files in `src/graph_manifold_alignment/alignment_methods/`
- All files in `src/graph_manifold_alignment/helpers/`
- All files in `src/graph_manifold_alignment/adni/`
- All files in `src/graph_manifold_alignment/autoencoder/`

**Example changes needed:**
```python
# OLD imports to change:
from Main.Pipeline import pipe
from Helpers.utils import some_function
from AlignmentMethods.DTA_andres import DTA

# NEW imports should be:
from ..helpers.Pipeline_Helpers import pipe
from ..helpers.utils import some_function
from .DTA_andres import DTA
```

### 3. **Test the Package Installation**
```bash
# Install in development mode
pip install -e .

# Test basic imports
python -c "import graph_manifold_alignment; print('Success!')"
python -c "from graph_manifold_alignment.autoencoder import GRAEAnchor; print('Autoencoder import works!')"
```

## 📝 **Documentation Updates Needed**

### 4. **Update Author Information**
Update the author information in these files with actual names:
- `setup.py` (line 17): Replace `"your.email@institution.edu"` with actual email
- `pyproject.toml` (line 8): Replace `"your.email@institution.edu"` with actual email
- Citation BibTeX entries: Replace `"[Additional Authors]"` with actual co-authors

### 5. **Complete Paper Abstract**
Add the actual paper abstract to:
- `docs/PAPER_OVERVIEW.md` - Replace the placeholder abstract with the real one
- Consider adding key results, figures, or methodology details

### 6. **Add Missing Documentation**
Create these missing documentation files:
- `docs/api/` - API documentation (consider using Sphinx)
- `docs/tutorials/` - Step-by-step tutorials
- Examples section with real data usage scenarios

## 🧪 **Code Quality & Testing**

### 7. **Implement Unit Tests**
Create proper unit tests in the `tests/` directory:
- `tests/test_grae.py` - Test GRAE functionality
- `tests/test_alignment_methods.py` - Test baseline methods
- `tests/test_helpers.py` - Test utility functions
- `tests/test_integration.py` - End-to-end workflow tests

### 8. **Add Type Hints**
Add proper type hints to all functions, especially in:
- `src/graph_manifold_alignment/autoencoder/AutoEncoders.py`
- `src/graph_manifold_alignment/alignment_methods/*.py`
- `src/graph_manifold_alignment/helpers/*.py`

### 9. **Code Formatting & Linting**
```bash
# Install development dependencies
pip install black flake8 mypy

# Format code
black src/ tests/ scripts/

# Check for issues
flake8 src/ tests/ scripts/
mypy src/
```

## 📊 **Data & Examples**

### 10. **Add Sample Datasets**
Create sample datasets for demonstrations:
- `data/samples/` - Small demo datasets
- `data/synthetic/` - Generated synthetic data for testing
- Update `.gitignore` to exclude large data files

### 11. **Complete Notebook Demonstrations**
Finish the demonstration notebooks:
- `notebooks/demonstrations/basic_usage.ipynb` - Basic GRAE usage
- `notebooks/demonstrations/comparison_study.ipynb` - Compare with baselines
- `notebooks/demonstrations/adni_analysis.ipynb` - Real biomedical data example
- Ensure all notebooks run without errors

### 12. **Add Visualization Examples**
Create visualization utilities and examples:
- Before/after alignment plots
- Embedding space visualizations
- Performance comparison charts
- Loss curve plots

## 🚀 **Deployment & Publication**

### 13. **Prepare for PyPI (Optional)**
If you want to publish to PyPI:
- Test package building: `python -m build`
- Test installation from built package
- Consider using TestPyPI first

### 14. **GitHub Repository Setup**
- Add repository description
- Set up GitHub topics/tags: `manifold-alignment`, `deep-learning`, `multimodal-ai`, `pytorch`
- Create release for MMAI 2025 paper
- Add link to paper when published

### 15. **CI/CD Pipeline (Optional)**
Set up GitHub Actions:
- `.github/workflows/tests.yml` - Automated testing
- `.github/workflows/docs.yml` - Documentation building
- Code quality checks on pull requests

## 🧹 **Cleanup Tasks**

### 16. **Remove Redundant Files**
After verifying everything works, remove old directories:
```bash
# BACKUP FIRST, then remove:
rm -rf Python_Files/
rm -rf "Output Graphs/"
rm -rf TwinAE-MA/  # Keep Twin_Autoencoder_Structure, remove duplicate
rm -rf Resources/  # After moving useful files
```

### 17. **Update .gitignore**
Add patterns for:
- Large model files (`*.pth`, `*.pkl`)
- Generated outputs (`outputs/**/*.png`, `outputs/**/*.pdf`)
- Notebook checkpoints
- Environment-specific files

### 18. **Virtual Environment**
Move or remove the current `myvenv/` directory:
- Either add to `.gitignore` 
- Or move outside the repository
- Document environment setup in README

## 📋 **Content Verification**

### 19. **Verify All Links Work**
Check all links in documentation:
- Workshop website links
- Internal documentation links
- GitHub repository references

### 20. **Final Testing Checklist**
- [ ] All imports work correctly
- [ ] Basic usage examples run
- [ ] Package installs cleanly
- [ ] Documentation builds without errors
- [ ] No broken links or references
- [ ] All notebooks execute successfully

## 🎯 **Priority Order**

**High Priority (Essential):**
1. Move alignment method files (#1)
2. Fix import statements (#2)
3. Test package installation (#3)
4. Update author information (#4)

**Medium Priority (Important):**
5. Add unit tests (#7)
6. Complete notebook demonstrations (#11)
7. Add sample datasets (#10)
8. Clean up old directories (#16)

**Low Priority (Nice to have):**
9. Add API documentation (#6)
10. Set up CI/CD (#15)
11. Prepare for PyPI (#13)

## 📞 **Support**

If you encounter issues with any of these tasks:
1. Check the import statements carefully - this is the most common issue
2. Verify file paths are correct after moves
3. Test with a fresh virtual environment to catch dependency issues
4. Consider the package structure when writing relative imports

---

**Estimated Time:** 4-8 hours for high/medium priority tasks, depending on complexity of import fixes and testing thoroughness.
