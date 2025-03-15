import psi4
import os
from pathlib import Path

# Set Psi4 memory and output file
psi4.set_memory('8 GB')  # Adjust based on your system
psi4.core.set_output_file('optimization_output.dat', False)

# Function to read PDB and convert to XYZ-like format for Psi4
def pdb_to_xyz(pdb_file):
    """
    Convert PDB file to a simple XYZ-like string for Psi4.
    Assumes PDB has ATOM records with x, y, z coordinates.
    """
    coordinates = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom = line[76:78].strip() or line[12:16].strip()  # Element symbol
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coordinates.append(f"{atom} {x} {y} {z}")
    return "\n".join(coordinates)

# Define optimization parameters
opt_settings = {
    'basis': '6-31G(d)',       # Basis set
    'scf_type': 'df',          # Density fitting for SCF
    'e_convergence': 1e-6,     # Energy convergence criterion
    'd_convergence': 1e-6,     # Density convergence criterion
    'maxiter': 200,            # Maximum SCF iterations
    'geom_maxiter': 200,       # Maximum geometry optimization iterations
    'opt_coordinates': 'cartesian',  # Use Cartesian coordinates
    'g_convergence': 'qchem'   # Convergence criterion
}

# Single-point energy settings (B3LYP-D3MBJ/6-31G*)
sp_settings = {
    'method': 'b3lyp-d3mbj',   # B3LYP with D3 dispersion and Becke-Johnson damping
    'basis': '6-31G*',         # Basis set for single-point energy
    'scf_type': 'df',          # Density fitting
    'e_convergence': 1e-6,
    'd_convergence': 1e-6,
    'maxiter': 200
}

# Load PDB file (replace 'molecule.pdb' with your file path)
pdb_file = 'molecule.pdb'
if not os.path.exists(pdb_file):
    raise FileNotFoundError(f"PDB file {pdb_file} not found!")

# Convert PDB to Psi4 geometry
geom_string = pdb_to_xyz(pdb_file)
molecule = psi4.geometry(f"""
0 1  # Neutral molecule, singlet (adjust charge/multiplicity if needed)
{geom_string}
symmetry c1
""")

# Set optimization options
psi4.set_options(opt_settings)

# Perform geometry optimization
print("Starting geometry optimization...")
opt_energy, opt_history = psi4.optimize('scf', molecule=molecule, return_history=True)
print(f"Optimized energy: {opt_energy:.6f} Hartree")

# Save optimized geometry
optimized_xyz = molecule.save_string_xyz_file()
with open('optimized_geometry.xyz', 'w') as f:
    f.write(optimized_xyz)

# For this example, we assume one conformer from the PDB.
# If you have multiple conformers, you'd need to loop over them.
# Here, we compute single-point energy for the optimized geometry.
conformers = [{'geometry': molecule.clone(), 'opt_energy': opt_energy}]

# Compute single-point energies with B3LYP-D3MBJ/6-31G*
psi4.set_options(sp_settings)
energies = []
for i, conf in enumerate(conformers):
    conf['geometry'].update_geometry()
    energy = psi4.energy(sp_settings['method'], molecule=conf['geometry'])
    energies.append({'index': i, 'energy': energy, 'opt_energy': conf['opt_energy']})
    print(f"Conformer {i}: Single-point energy = {energy:.6f} Hartree")

# Convert energies to kcal/mol and filter within 10 kcal/mol of minimum
hartree_to_kcal = 627.509  # Conversion factor
min_energy = min(e['energy'] for e in energies)  # Global minimum (single-point energy)
filtered_conformers = []

for conf in energies:
    rel_energy = (conf['energy'] - min_energy) * hartree_to_kcal
    if rel_energy <= 10.0:
        filtered_conformers.append({
            'index': conf['index'],
            'energy_hartree': conf['energy'],
            'rel_energy_kcal': rel_energy
        })
        print(f"Conformer {conf['index']}: Relative energy = {rel_energy:.2f} kcal/mol (retained)")
    else:
        print(f"Conformer {conf['index']}: Relative energy = {rel_energy:.2f} kcal/mol (discarded)")

# Summary
print(f"\nRetained {len(filtered_conformers)} conformers within 10 kcal/mol of the global minimum.")