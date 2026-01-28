# Axios

Vision-based humanoid motion imitation and autonomous boxing AI.

## Setup

Requires conda (miniforge) for pybullet on macOS ARM64.

```
conda activate axios
pip install -r requirements.txt
```

If starting fresh:

```
brew install miniforge
conda create -n axios python=3.12 -y
conda activate axios
conda install -c conda-forge pybullet -y
pip install -r requirements.txt
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
