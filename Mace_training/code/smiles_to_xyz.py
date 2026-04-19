#!/usr/bin/env python3
"""
Segment 1: Convert SMILES string to 3D XYZ file using RDKit.

This script takes a SMILES string, adds hydrogens, embeds it in 3D,
performs a geometry optimization, and writes the result to an XYZ file.
"""

import argparse
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_xyz(smiles: str, output_path: str) -> None:
    """
    Convert a SMILES string to an XYZ file.

    Args:
        smiles: The SMILES string representing the molecule.
        output_path: Path to write the output XYZ file.
    """
    # Create molecule from SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Add hydrogens
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    # useRandomCoords helps with larger molecules
    params = AllChem.ETKDGv3()
    params.useRandomCoords = True
    embed_result = AllChem.EmbedMolecule(mol, params)
    if embed_result != 0:
        raise RuntimeError("Failed to embed molecule in 3D")

    # Optimize geometry using MMFF94 force field
    AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94")

    # Get atomic positions
    conf = mol.GetConformer()
    positions = conf.GetPositions()

    # Get atomic symbols
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]

    # Write XYZ file
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        # First line: number of atoms
        f.write(f"{len(atoms)}\n")
        # Second line: comment (SMILES)
        f.write(f"Generated from SMILES: {smiles}\n")
        # Atomic positions
        for symbol, pos in zip(atoms, positions):
            f.write(f"{symbol:2s} {pos[0]:15.8f} {pos[1]:15.8f} {pos[2]:15.8f}\n")

    print(f"XYZ file written to: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SMILES string to XYZ file"
    )
    parser.add_argument(
        "smiles",
        help="SMILES string of the molecule",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="results/output.xyz",
        help="Output XYZ file path (default: results/output.xyz)",
    )

    args = parser.parse_args()

    try:
        smiles_to_xyz(args.smiles, args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
