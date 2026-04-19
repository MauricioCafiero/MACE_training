# Molecular Dynamics with UMA ML Potential

This project converts SMILES strings to 3D structures, runs molecular dynamics using the UMA ML potential, and produces trajectory files with gradients.

## Overview

The pipeline consists of two segments:

1. **SMILES to XYZ** (`code/smiles_to_xyz.py`): Converts a SMILES string to a 3D XYZ file using RDKit
2. **Run Dynamics** (`code/run_dynamics.py`): Runs molecular dynamics using ASE with the Fairchem UMA calculator

## Requirements

- Python 3.8+
- RDKit
- ASE (Atomic Simulation Environment)
- Fairchem (fairchem-core)
- PyTorch
- Py3Dmol (for visualization)

## Installation

```bash
pip install rdkit ase fairchem-core torch py3Dmol
```

## Setup

Set your HuggingFace API token as an environment variable:

```bash
export HF_TOKEN="your_token_here"
```

On Windows:
```cmd
set HF_TOKEN=your_token_here
```

## Usage

### Step 1: Generate 3D Structure from SMILES

```bash
python code/smiles_to_xyz.py "CCO" -o results/ethanol.xyz
```

This will:
- Parse the SMILES string
- Add hydrogens
- Embed in 3D and optimize geometry with MMFF94
- Save to `results/ethanol.xyz`

### Step 2: Run Molecular Dynamics

```bash
python code/run_dynamics.py results/ethanol.xyz -o results/trajectory.xyz --steps 1000 --temperature 300
```

Options:
- `--steps`: Number of MD steps (default: 1000)
- `--timestep`: Timestep in femtoseconds (default: 1.0)
- `--temperature`: Temperature in Kelvin (default: 300)
- `--md-type`: `langevin` (NVT) or `verlet` (NVE) (default: langevin)

### Full Pipeline Example

```bash
# Ethanol example
python code/smiles_to_xyz.py "CCO" -o results/ethanol.xyz
python code/run_dynamics.py results/ethanol.xyz -o results/ethanol_trajectory.xyz --steps 500 --temperature 300
```

## Output

The final trajectory file contains:
- All frames from the MD simulation
- Atomic forces (gradients) for each frame in extended XYZ format

## Project Structure

```
.
├── code/
│   ├── smiles_to_xyz.py    # Segment 1: SMILES to 3D XYZ
│   └── run_dynamics.py     # Segment 2: MD with UMA potential
├── results/                 # Output files
└── README.md
```
