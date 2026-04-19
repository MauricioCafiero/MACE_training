# Project Memory: MACE Training

## Current State

### Project Overview
3D XYZ structure generation from SMILES strings → molecular dynamics with UMA ML potential → trajectory XYZ with gradients

### Completed Files

1. **code/smiles_to_xyz.py** - Segment 1: SMILES to 3D XYZ
   - Uses RDKit to parse SMILES, add Hs, embed 3D, optimize with MMFF94
   - Fixed issue with EmbedMolecule params (useRandomCoords on ETKDGv3 object)
   - Working status: Verified

2. **code/run_dynamics.py** - Segment 2: MD with UMA potential
   - Uses ASE with Fairchem FAIRChemCalculator
   - UMA predictor: `pretrained_mlip.get_predict_unit("uma-s-1", device=device)`
   - Supports Langevin (NVT) and Verlet (NVE) integrators
   - Writes extended XYZ with forces/gradients

3. **README.md** - Project documentation

### Project Structure
```
Mace_training/
├── code/
│   ├── smiles_to_xyz.py
│   └── run_dynamics.py
├── results/          # Output directory
├── memory/           # Session persistence
└── README.md
```

### Usage
```bash
# Step 1: SMILES to XYZ
python code/smiles_to_xyz.py "CCO" -o results/ethanol.xyz

# Step 2: Run MD
python code/run_dynamics.py results/ethanol.xyz -o results/trajectory.xyz --steps 1000
```

### Environment Requirements
- `HF_TOKEN` environment variable must be set for HuggingFace

### Last Updated
2026-04-19
