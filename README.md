# Axios

Vision-based humanoid motion imitation and autonomous boxing AI.

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### macOS ARM64 (Apple Silicon) - PyBullet Installation

PyBullet does not provide pre-built wheels for macOS ARM64. Install via conda:

```
brew install miniforge
conda create -n axios python=3.12
conda activate axios
conda install -c conda-forge pybullet
pip install -r requirements.txt
```

Or install system dependencies and build from source:

```
brew install cmake
pip install pybullet
```

## Project Structure

```
src/           Source code
config/        Configuration files
scripts/       Entry point scripts
outputs/       Generated data and recordings
```

## Usage

Manual control test:
```
python scripts/run_manual_control.py
```

Mirror mode:
```
python scripts/run_mirror.py
```
