from ase.io import read, write
import warnings
warnings.filterwarnings("ignore")
from mace.cli.run_train import main as mace_run_train_main
import sys
import logging
import glob
import os
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution
from aseMolec import extAtoms as ea
from aseMolec import anaAtoms as aa
import time
import numpy as np
import pylab as pl
from IPython import display
from mace.calculators import MACECalculator
from xtb.ase.calculator import XTB
from matplotlib import pyplot as plt
from tqdm import tqdm
from ase import units
import random
from xtb.ase.calculator import XTB
from mace.calculators import mace_mp
import py3Dmol

def train_mace(config_file_path):
    logging.getLogger().handlers.clear()
    sys.argv = ["program", "--config", config_file_path]
    mace_run_train_main()

def simpleMD(init_conf, temp, calc, fname, s, T):
    '''
    Runs a simple molecular dynamics simulation.
    '''
    init_conf.set_calculator(calc)

    #initialize the temperature

    MaxwellBoltzmannDistribution(init_conf, temperature_K=300) #initialize temperature at 300
    Stationary(init_conf)
    ZeroRotation(init_conf)

    dyn = Langevin(init_conf, 1.0*units.fs, temperature_K=temp, friction=0.1) #drive system to desired temperature

    #%matplotlib inline

    time_fs = []
    temperature = []
    energies = []

    #remove previously stored trajectory with the same name
    os.system('rm -rfv '+fname)

    fig, ax = pl.subplots(2, 1, figsize=(6,6), sharex='all', gridspec_kw={'hspace': 0, 'wspace': 0})

    def write_frame():
            dyn.atoms.info['energy_mace'] = dyn.atoms.get_potential_energy()
            dyn.atoms.arrays['force_mace'] = dyn.atoms.calc.get_forces()
            dyn.atoms.write(fname, append=True)
            time_fs.append(dyn.get_time()/units.fs)
            temperature.append(dyn.atoms.get_temperature())
            energies.append(dyn.atoms.get_potential_energy()/len(dyn.atoms))

            ax[0].plot(np.array(time_fs), np.array(energies), color="b")
            ax[0].set_ylabel('E (eV/atom)')

            # plot the temperature of the system as subplots
            ax[1].plot(np.array(time_fs), temperature, color="r")
            ax[1].set_ylabel('T (K)')
            ax[1].set_xlabel('Time (fs)')

            display.clear_output(wait=True)
            display.display(pl.gcf())
            time.sleep(0.01)

    dyn.attach(write_frame, interval=s)
    t0 = time.time()
    dyn.run(T)
    t1 = time.time()
    print("MD finished in {0:.2f} minutes!".format((t1-t0)/60))

def view_traj(filepath: str):
    '''
    Receives a filepath to an xyz trajectory file and visualizes it with py3Dmol.
    '''
    f = open(filepath, "r")
    lines = f.readlines()
    f.close()
    xyz_string = ''.join(lines)

    viewer = py3Dmol.view(width=800, height=400)
    viewer.addModelsAsFrames(xyz_string, "xyz")
    viewer.setStyle({"stick": {}, "sphere": {"radius": 0.5}})
    viewer.animate({'loop': 'forward', 'reps': 2})
    viewer.zoomTo()
    viewer.show()

def make_train_file(filename: str, ft: bool = False, max_L: int = 0, 
                    r_max: float = 4.0, name: str = "mace_model", 
                    max_num_epochs: int = 300, batch_size: int = 10,
                    swa: bool = True, energy_key: str = 'energy_xtb',
                    forces_key: str = 'forces_xtb', train_file: str = "data/train_file.xyz",
                    valid_file: str = "data/val_file.xyz", test_file: str = "data/test_file.xyz",
                    valid_fraction: float = 0.1, seed: int = 42, stress_weight: float = 0.0,
                    energy_weight: float = 1.0, forces_weight: float = 10.0, foundation_model: str = 'small',
                    num_samples_pt: int = 300):
    '''
    Creates a yaml file for training a MACE model. If ft is True, creates a yaml file for fine-tuning a MACE model. 
    Otherwise, creates a yaml file for training a MACE model from scratch.
    '''
    if ft:
        f = open('data/ft_template.yml', 'r')
    else:
        f = open('data/template.yml', 'r') 
    template = f.readlines()
    f.close()   

    variables = [max_L, r_max, name, max_num_epochs, batch_size, swa, energy_key, forces_key, train_file, 
                 valid_file, test_file, valid_fraction, seed, stress_weight, energy_weight, forces_weight,
                 foundation_model, num_samples_pt]
    
    variable_names = ['max_L', 'r_max', 'name', 'max_num_epochs', 'batch_size', 'swa', 'energy_key', 'forces_key', 
                      'train_file', 'valid_file', 'test_file', 'valid_fraction', 'seed', 'stress_weight', 
                      'energy_weight', 'forces_weight', 'foundation_model', 'num_samples_pt']

    for name in variable_names:
        for line in template:
            if name in line:
                idx = template.index(line)
                template[idx] = f"{name}: {variables[variable_names.index(name)]}\n"
    with open(filename, 'w') as f:
        f.writelines(template)
        f.close()
