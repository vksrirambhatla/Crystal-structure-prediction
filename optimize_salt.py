import psi4
import os
from pathlib import Path

# Print Psi4 version
print(f"Psi4 version: {psi4.__version__}")

# Set Psi4 memory and output file
psi4.set_memory('8 GB')  # Reduced to avoid potential memory issues
psi4.core.set_output_file('optimization_output.dat', False)

# Function to read PDB and convert to XYZ-like format for Psi4
def pdb_to_xyz(pdb_file, shift_x=0.0, shift_y=0.0, shift_z=0.0):
    coordinates = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom = line[76:78].strip() or line[12:16].strip()
                x = float(line[30:38]) + shift_x
                y = float(line[38:46]) + shift_y
                z = float(line[46:54]) + shift_z
                coordinates.append(f"{atom} {x} {y} {z}")
    return "\n".join(coordinates)

# Define optimization parameters with PCM solvent
opt_settings = {
    'basis': '3-21G',  # Small basis set for faster testing
    'scf_type': 'df',
    'e_convergence': 1e-5,  # Loosened for faster convergence
    'd_convergence': 1e-5,
    'maxiter': 100,  # Reduced SCF iterations per displacement
    'geom_maxiter': 10,  # Even fewer steps for testing
    'opt_coordinates': 'cartesian',
    'g_convergence': 'gau_loose',  # Looser convergence criteria
    'pcm': True,
    'pcm_scf_type': 'total'
}

# Configure PCM solvent with adjusted cavity parameters
psi4.pcm_helper("""
Units = Angstrom
Medium {
    SolverType = IEFPCM
    Solvent = Water
}
Cavity {
    Type = GePol
    Scaling = False
    RadiiSet = Bondi
    Mode = Implicit
    Area = 0.1
    MinRadius = 0.3
}
""")

# Single-point energy settings
sp_settings = {
    'method': 'b3lyp-d3mbj',
    'basis': '6-31G*',
    'scf_type': 'df',
    'e_convergence': 1e-6,
    'd_convergence': 1e-6,
    'maxiter': 200,
    'pcm': True,
    'pcm_scf_type': 'total'
}

# Load PDB files
cation_file = 'cation.pdb'
anion_file = 'anion.pdb'

for pdb_file in [cation_file, anion_file]:
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB file {pdb_file} not found!")

# Convert PDBs to XYZ strings
cation_geom = pdb_to_xyz(cation_file)
anion_geom = pdb_to_xyz(anion_file, shift_x=4.0, shift_y=0.0, shift_z=0.0)

# Debug: Print geometries
print("Cation geometry:\n", cation_geom)
print("Anion geometry:\n", anion_geom)

if not cation_geom.strip() or not anion_geom.strip():
    raise ValueError("One or both PDB files resulted in empty geometry strings!")

# Create separate molecule objects for cation and anion
cation = psi4.geometry(f"""
1 1
{cation_geom}
units angstrom
no_reorient
no_com
""")

anion = psi4.geometry(f"""
-1 1
{anion_geom}
units angstrom
no_reorient
no_com
""")

# Combine fragments manually into a single molecule
cation_xyz = cation.save_string_xyz_file().splitlines()[2:]  # Skip header
anion_xyz = anion.save_string_xyz_file().splitlines()[2:]
combined_xyz = "\n".join(cation_xyz + anion_xyz)

geometry_string = f"""
0 1
units angstrom
{combined_xyz}
no_reorient
no_com
"""

# Debug: Print the combined geometry string
print("Combined geometry string:\n", geometry_string)

# Create the final molecule object
molecule = psi4.geometry(geometry_string)

# Set optimization options
psi4.set_options(opt_settings)

# Perform geometry optimization with progress monitoring
print("Starting geometry optimization of the organic ion pair with PCM solvent...")
opt_energy, opt_history = psi4.optimize('scf', molecule=molecule, return_history=True)
print("Optimization completed successfully!")
print(f"Optimized energy: {opt_energy:.6f} Hartree")

# Debug: Print optimization history
if opt_history:
    print(f"Number of optimization steps: {len(opt_history['energy'])}")
    print(f"Energy history: {opt_history['energy']}")

# Save optimized geometry
optimized_xyz = molecule.save_string_xyz_file()
with open('optimized_salt_geometry.xyz', 'w') as f:
    f.write(optimized_xyz)
print("Optimized geometry saved to 'optimized_salt_geometry.xyz'")

# Compute single-point energy
conformers = [{'geometry': molecule.clone(), 'opt_energy': opt_energy}]
psi4.set_options(sp_settings)
energies = []
for i, conf in enumerate(conformers):
    conf['geometry'].update_geometry()
    energy = psi4.energy(sp_settings['method'], molecule=conf['geometry'])
    energies.append({'index': i, 'energy': energy, 'opt_energy': conf['opt_energy']})
    print(f"Conformer {i}: Single-point energy = {energy:.6f} Hartree")

# Filter conformers within 10 kcal/mol
hartree_to_kcal = 627.509
min_energy = min(e['energy'] for e in energies)
filtered_conformers = []

for conf in energies:
    rel_energy = (conf['energy'] - min_energy) * hartree_to_kcal
    if rel_energy <= 10.0:
        filtered_conformers.append({
            'index': conf['index'],
            'energy_hartree': conf['energy'],
            'rel_energy_kcal': rel_energy
        })