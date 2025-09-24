#!/usr/bin/env python3
"""
Main pipeline runner script for Graph Manifold Alignment experiments.

This script provides a command-line interface to run various alignment experiments.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the src directory to the path so we can import our package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from graph_manifold_alignment import alignment_methods
    from graph_manifold_alignment import helpers
except ImportError as e:
    print(f"Error importing package: {e}")
    print("Make sure you've installed the package with: pip install -e .")
    sys.exit(1)


def main():
    """Main entry point for the pipeline runner."""
    parser = argparse.ArgumentParser(
        description="Graph Manifold Alignment Pipeline Runner"
    )
    
    parser.add_argument(
        "--method",
        choices=["dta", "jlma", "procrustes", "magan", "mali", "mash", "ssma"],
        default="dta",
        help="Alignment method to use"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the input data"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    print(f"Running {args.method} alignment on {args.data_path}")
    print(f"Results will be saved to {args.output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # TODO: Implement actual pipeline logic
    print("Pipeline execution would start here...")
    print("This is a template - implement the actual pipeline logic based on your existing Pipeline.py")


if __name__ == "__main__":
    main()
