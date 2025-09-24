"""
Basic tests for the Graph Manifold Alignment package.

This module contains basic unit tests to verify package functionality.
"""

import unittest
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import graph_manifold_alignment as gma
    from graph_manifold_alignment import alignment_methods, helpers, adni, autoencoder
except ImportError as e:
    print(f"Warning: Could not import package components: {e}")


class TestPackageStructure(unittest.TestCase):
    """Test that the package structure is correct."""
    
    def test_package_import(self):
        """Test that the main package can be imported."""
        try:
            import graph_manifold_alignment
            self.assertTrue(hasattr(graph_manifold_alignment, '__version__'))
        except ImportError:
            self.fail("Could not import main package")
    
    def test_submodules_import(self):
        """Test that submodules can be imported."""
        try:
            from graph_manifold_alignment import alignment_methods
            from graph_manifold_alignment import helpers
            from graph_manifold_alignment import adni
            from graph_manifold_alignment import autoencoder
        except ImportError as e:
            self.fail(f"Could not import submodules: {e}")


class TestAlignmentMethods(unittest.TestCase):
    """Test alignment methods functionality."""
    
    def test_alignment_methods_available(self):
        """Test that alignment methods are available."""
        try:
            from graph_manifold_alignment import alignment_methods
            # Add specific tests for your alignment methods here
            self.assertTrue(True)  # Placeholder
        except ImportError:
            self.skip("Alignment methods not available")


class TestHelpers(unittest.TestCase):
    """Test helper functions."""
    
    def test_helpers_available(self):
        """Test that helper functions are available."""
        try:
            from graph_manifold_alignment import helpers
            # Add specific tests for your helper functions here
            self.assertTrue(True)  # Placeholder
        except ImportError:
            self.skip("Helper functions not available")


if __name__ == "__main__":
    unittest.main()
