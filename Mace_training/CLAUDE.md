# CLAUDE.md

## Project Overview
This is code to generate 3D XYZ structures from SMILES strings, run dynamics on them, and produce a final XYZ file with the dynamics trajectory including gradients.

## Tech Stack
- Language: Python
- packages: 
  * RDKit for general utility
  * ASE for running dynamics and generating the final XYZ trajectory file
  * Fairchem for the ASE calculator: use the UMA ML potential
  * Py3DMol for visualization

## Project Structure
code/          # all code
results/       # any new produced files

## Other information
- the UMA weights will be pulled from HuggingFace; assume that the HuggingFace API key is an environment variable
