# Contributing to LiDAR Converter

Thank you for your interest in contributing to LiDAR Converter! We appreciate your effort.

## Development Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Setup Instructions

1. **Fork and clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/lidar-converter.git
cd lidar-converter
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in editable mode with dev dependencies:**
```bash
pip install -e ".[dev]"
```

## Running Tests

Run all tests:
```bash
pytest
```

Run with coverage report:
```bash
pytest --cov=Lidar_Converter --cov-report=html
```

## Code Quality

We maintain consistent code quality using:

### Format with Black
```bash
black Lidar_Converter/
black tests/
```

### Lint with Flake8
```bash
flake8 Lidar_Converter/
flake8 tests/
```

### Type checking with mypy
```bash
mypy Lidar_Converter/
```

### Run all checks
```bash
black Lidar_Converter/ tests/ && flake8 Lidar_Converter/ tests/ && mypy Lidar_Converter/ && pytest
```

## Making Changes

1. **Create a feature branch:**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes and commit:**
```bash
git add .
git commit -m "feat: description of your changes"
```

Follow conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `test:` for test additions
- `refactor:` for code refactoring
- `style:` for formatting changes

3. **Push and create a Pull Request:**
```bash
git push origin feature/your-feature-name
```

## Pull Request Guidelines

- Include a clear description of changes
- Reference related issues (e.g., "Fixes #123")
- Ensure all tests pass locally
- Update documentation if needed
- Follow code style guidelines

## Questions?

Feel free to open an issue or discussion in the repository.
