# Semantic MMADDPG Simulator (Multi-Agent OFDMA Resource Allocation)

This repository implements a modular simulation and training pipeline for semantic-aware resource allocation using a modified MADDPG (MMADDPG) approach.

This project implements a multi-agent reinforcement learning framework for
OFDMA resource allocation with orthogonal RB assignment, power control, and modulation adaptation.

## Features
- Multi-agent TD3 with local actors and critics
- Centralized twin global critics
- Orthogonal RB allocation with collision penalties
- SINR-aware environment
- Full training and evaluation pipeline
- Reproducible experiments

## Project Structure
- Classes/
  - agent.py
  - global_controller.py
  - networks.py
  - g_network.py
  - buffer.py
- environment.py
- main.py
- config.py
- logs/
- checkpoints/

## Training
Set in config.py:
```python
mode = "train"
