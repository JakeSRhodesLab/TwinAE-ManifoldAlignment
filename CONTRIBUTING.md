# Contributing to Graph Manifold Alignment

Thank you for your interest in contributing to the Graph Manifold Alignment project! This document provides guidelines for contributors.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/Graph-Manifold-Alignment.git
   cd Graph-Manifold-Alignment
   ```
3. **Set up the development environment**:
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Use meaningful variable and function names

### Code Formatting

We use `black` for code formatting:

```bash
black src/ tests/ scripts/
```

### Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting:
  ```bash
  python -m pytest tests/
  ```
- Aim for good test coverage

### Documentation

- Update docstrings for any modified functions/classes
- Add examples in docstrings when helpful
- Update the README.md if adding new features

## Submitting Changes

1. **Create a new branch** for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the guidelines above

3. **Test your changes**:
   ```bash
   python -m pytest tests/
   black --check src/ tests/ scripts/
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add descriptive commit message"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

## Pull Request Guidelines

- Provide a clear description of what your PR does
- Reference any related issues
- Ensure all tests pass
- Keep PRs focused on a single feature or fix
- Update documentation if needed

## Code Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged

## Reporting Issues

- Use GitHub Issues to report bugs or request features
- Provide clear descriptions and reproduction steps
- Include relevant code snippets or error messages

## Questions?

Feel free to open an issue for any questions about contributing.

Thank you for contributing to Graph Manifold Alignment!
