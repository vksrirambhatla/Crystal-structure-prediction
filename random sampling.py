import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import pymatgen.core as pmg
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from multiprocessing import Pool
import random
import os

# Load the geometry-optimized structure from PDB
pdb_file = "Aurora_00001.pdb"  # Ensure this file is in your working directory
mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)  # Keep hydrogens
if mol is None:
    raise ValueError("Failed to load PDB file. Check format and path.")
conf = mol.GetConformer()

# Van der Waals radii (in Å, from RDKit or literature)
vdw_radii = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'H': 1.2, 'F': 1.47}

# Calculate molecular volume (approximate from coordinates)
def calculate_molecular_volume(conf):
    coords = conf.GetPositions()
    dimensions = np.max(coords, axis=0) - np.min(coords, axis=0)
    return np.prod(dimensions) * 0.75  # Rough estimate, adjust factor as needed

mol_volume = calculate_molecular_volume(conf)
print(f"Estimated molecular volume: {mol_volume:.2f} Å³")

# Commonly observed space groups (assuming chiral molecule, adjust if achiral)
space_groups = {
    "chiral": ["P212121", "P21212", "C2221", "P22121", "P2122", "P2212", "P2221", "P422", "P4212", "P3221"],
}
selected_space_groups = space_groups["chiral"]  # Adjust based on your molecule's chirality

def random_unit_cell_params(space_group, mol_volume):
    orientation = np.random.uniform(0, 2 * np.pi, 3)
    coords = conf.GetPositions()
    proj_a, proj_b, proj_c = np.max(coords, axis=0) - np.min(coords, axis=0)
    a = random.uniform(proj_a * 0.9, proj_a * 1.1)
    b = random.uniform(proj_b * 0.9, proj_b * 1.1)
    c = random.uniform(proj_c * 0.9, proj_c * 1.1)
    
    if space_group in ["P1", "P1̅"]:
        v_star = random.uniform(0.55, 0.85) * mol_volume / (a * b * c)
        alpha, beta, gamma = np.random.uniform(60, 120, 3)  # Degrees
    else:
        alpha, beta, gamma = 90, 90, 90  # Example for orthorhombic
    return a, b, c, alpha, beta, gamma, orientation

def check_packing_criteria(conf, a, b, c, alpha, beta, gamma, vdw_radii):
    volume = a * b * c * np.sin(np.radians(alpha)) * np.sin(np.radians(beta)) * np.sin(np.radians(gamma))
    packing_fraction = mol_volume / volume
    
    coords = conf.GetPositions()
    min_dist = np.inf
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = np.linalg.norm(coords[i] - coords[j])
            min_vdw = vdw_radii[conf.GetAtomWithIdx(i).GetSymbol()] + vdw_radii[conf.GetAtomWithIdx(j).GetSymbol()]
            min_dist = min(min_dist, dist / min_vdw)
    return 0.55 <= packing_fraction <= 0.85 and min_dist >= 0.7

def generate_packing(args):
    space_group, conf = args
    sg = pmg.SpaceGroup(space_group)
    max_attempts = 1000
    for _ in range(max_attempts):
        a, b, c, alpha, beta, gamma, orientation = random_unit_cell_params(space_group, mol_volume)
        if check_packing_criteria(conf, a, b, c, alpha, beta, gamma, vdw_radii):
            return (space_group, a, b, c, alpha, beta, gamma, orientation)
    return None

# Parallel execution with folder output
if __name__ == '__main__':
    num_packings = 1000
    os.makedirs("crystal_packings", exist_ok=True)  # Create root directory if it doesn't exist
    
    with Pool(processes=4) as pool:
        results = pool.map(generate_packing, [(sg, conf) for sg in selected_space_groups for _ in range(num_packings // len(selected_space_groups))])
    
    # Organize packings by space group
    space_group_packings = {}
    for result in results:
        if result is not None:
            space_group, a, b, c, alpha, beta, gamma, orientation = result
            if space_group not in space_group_packings:
                space_group_packings[space_group] = []
                os.makedirs(f"crystal_packings/{space_group}", exist_ok=True)  # Create space group folder
            space_group_packings[space_group].append((a, b, c, alpha, beta, gamma, orientation))
    
    # Save packings to files
    for space_group, packings in space_group_packings.items():
        for i, (a, b, c, alpha, beta, gamma, orientation) in enumerate(packings):
            with open(f"crystal_packings/{space_group}/packing_{i}.txt", "w") as f:
                f.write(f"Packing {i}:\n")
                f.write(f"a={a:.2f} Å, b={b:.2f} Å, c={c:.2f} Å, α={alpha:.1f}°, β={beta:.1f}°, γ={gamma:.1f}°\n")
                f.write(f"Orientation: {orientation}\n")
    
    total_successful = sum(len(packings) for packings in space_group_packings.values())
    print(f"Total successful packings: {total_successful}/{num_packings}")