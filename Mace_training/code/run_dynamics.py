#!/usr/bin/env python3
"""
Segment 2: Run molecular dynamics using ASE with Fairchem UMA calculator.

This script loads an XYZ file, sets up the UMA ML potential from Fairchem,
runs MD simulation, and writes a trajectory XYZ file with gradients included.
"""

import argparse
import os
import sys
from pathlib import Path

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
        if atoms.get_kinetic_energy() == 0:
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
        # Store in atoms object info for writing
        atoms.info['forces'] = forces
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

    # Write as extended XYZ with forces as additional columns
    # ASE's write will include forces if present in atoms.arrays
    for i, frame in enumerate(trajectory):
        frame.arrays['fx'] = frame.info['forces'][:, 0]
        frame.arrays['fy'] = frame.info['forces'][:, 1]
        frame.arrays['fz'] = frame.info['forces'][:, 2]

    # Write all frames
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
        sys.exit(1)


if __name__ == "__main__":
    main()
