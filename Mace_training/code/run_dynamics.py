#!/usr/bin/env python3
"""
Segment 2: Run molecular dynamics using ASE with Fairchem UMA calculator.

This script loads an XYZ file, sets up the UMA ML potential from Fairchem,
runs MD simulation, and writes a trajectory XYZ file with gradients included.
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import read, write
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase import units

# Fairchem calculator for UMA potential
from fairchem.core import FAIRChemCalculator, pretrained_mlip


def run_dynamics(
    input_xyz: str,
    output_xyz: str,
    steps: int = 1000,
    timestep: float = 1.0,
    temperature: float = 300.0,
    md_type: str = "langevin",
) -> None:
    """
    Run molecular dynamics on a structure using UMA ML potential.

    Args:
        input_xyz: Path to input XYZ file.
        output_xyz: Path to write output trajectory XYZ file.
        steps: Number of MD steps to run.
        timestep: Timestep in femtoseconds.
        temperature: Temperature in Kelvin.
        md_type: Type of MD ('verlet' or 'langevin').
    """
    # Load the structure
    atoms = read(input_xyz)
    print(f"Loaded structure with {len(atoms)} atoms")

    # Set charge and spin for the UMA calculator (avoids warnings and potential errors)
    atoms.info['charge'] = 0
    atoms.info['spin'] = 1

    # Set up the UMA calculator from Fairchem
    # Uses HuggingFace API token from environment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device=device)
    calculator = FAIRChemCalculator(predictor, task_name="omol")
    atoms.calc = calculator

    # Verify calculator works by computing initial energy
    print(f"Initial energy: {atoms.get_potential_energy():.4f} eV")

    # Set up MD
    timestep_seconds = timestep * units.fs

    if md_type == "langevin":
        # Langevin dynamics for NVT ensemble
        dyn = Langevin(
            atoms,
            timestep=timestep_seconds,
            temperature_K=temperature,
            friction=0.02,  # 1/fs, typical for MD
        )
    elif md_type == "verlet":
        # Velocity Verlet for NVE ensemble
        dyn = VelocityVerlet(atoms, timestep=timestep_seconds)
        # Set initial velocities if needed
        ke = atoms.get_kinetic_energy()
        if np.allclose(ke, 0):
            from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
            MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    else:
        raise ValueError(f"Unknown MD type: {md_type}")

    # Create trajectory storage
    trajectory = []

    def write_step():
        """Callback to save each step with forces."""
        # Get forces (negative gradients)
        forces = atoms.get_forces()
        # Store forces in arrays (not info) to avoid comparison issues
        atoms.arrays['forces'] = forces
        trajectory.append(atoms.copy())

    # Attach callback
    dyn.attach(write_step, interval=1)

    # Run MD
    print(f"Running {md_type} MD for {steps} steps at {temperature}K")
    print(f"Timestep: {timestep} fs")
    dyn.run(steps)

    # Write trajectory to XYZ with extended format including forces
    output = Path(output_xyz)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Write as extended XYZ with positions and forces only
    for frame in trajectory:
        # Remove momentum array to keep output clean
        if 'momenta' in frame.arrays:
            del frame.arrays['momenta']

    # Write all frames - forces array will be written as fx, fy, fz columns
    write(output_xyz, trajectory, format='extxyz')

    print(f"Trajectory written to: {output_xyz}")
    print(f"Total frames: {len(trajectory)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run MD using UMA potential with ASE"
    )
    parser.add_argument(
        "input_xyz",
        help="Input XYZ file from segment 1",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="results/trajectory.xyz",
        help="Output trajectory XYZ file (default: results/trajectory.xyz)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of MD steps (default: 1000)",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=1.0,
        help="Timestep in femtoseconds (default: 1.0)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature in Kelvin (default: 300)",
    )
    parser.add_argument(
        "--md-type",
        choices=["verlet", "langevin"],
        default="langevin",
        help="MD integrator type (default: langevin)",
    )

    args = parser.parse_args()

    # Check for HF token
    if not os.environ.get("HF_TOKEN"):
        print(
            "Warning: HF_TOKEN environment variable not set. "
            "Fairchem may fail to download UMA weights.",
            file=sys.stderr,
        )

    try:
        run_dynamics(
            args.input_xyz,
            args.output,
            args.steps,
            args.timestep,
            args.temperature,
            args.md_type,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
