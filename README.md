# Stationary Detector

A Python project template using Poetry for dependency management.

## Setup

1. Create virtual environment:
   ```bash
   python3 -m venv .venv
   ```

2. Activate virtual environment:
   ```bash
   source .venv/bin/activate
   ```

3. Install Poetry:
   ```bash
   pip3 install poetry
   ```

4. Install dependencies:
   ```bash
   poetry install
   ```

## Run

## Project Structure

```
├── src/
│   └── stationary_detector/    # Main package
│       ├── __init__.py         # Package initialization
│       └── main.py             # Main application entry point
├── pyproject.toml              # Poetry configuration and dependencies
├── README.md                   # This file
└── .gitignore                  # Git ignore patterns
```

## Dependencies

This project includes various packages for computer vision, machine learning, and data analysis:
- OpenCV for computer vision
- PyTorch for deep learning
- Ultralytics for YOLO models
- Deep Sort for object tracking
- Pandas/NumPy for data manipulation
- And many more...

See `pyproject.toml` for the complete list of dependencies.