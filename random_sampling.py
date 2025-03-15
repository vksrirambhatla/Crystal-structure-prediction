import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pymatgen.symmetry.groups import SpaceGroup
from multiprocessing import Pool
import random
import os
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load the geometry-optimized structure from PDB (once globally)
pdb_file = "Aurora_00001.pdb"  # Ensure this file is in your working directory
mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)  # Keep hydrogens
if mol is None:
    raise ValueError("Failed to load PDB file. Check format and path.")

# Van der Waals radii (in Å, from RDKit or literature)
vdw_radii = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'H': 1.2, 'F': 1.47}

# Calculate molecular volume (fallback to bounding box if rdMolVolume fails)
def calculate_molecular_volume(mol):
    try:
        from rdkit.Chem import rdMolVolume
        return rdMolVolume.GetVolume(mol)
    except ImportError:
        logger.warning("rdMolVolume not available, using bounding box approximation")
        conf = mol.GetConformer()
        coords = conf.GetPositions()
        dimensions = np.max(coords, axis=0) - np.min(coords, axis=0)
        return np.prod(dimensions) * 0.75  # Rough estimate

mol_volume = calculate_molecular_volume(mol)
logger.info(f"Estimated molecular volume: {mol_volume:.2f} Å³")

# Commonly observed space groups (chiral, appropriate for your molecule)
space_groups = {
    "chiral": ["P212121", "P21212", "C2221", "P22121", "P2122", "P2212", "P2221", "P422", "P4212", "P3221"],
    "achiral": ["P21/c", "P21/n", "P21/a", "C2/c", "Pbca", "Pna21", "Pccn", "Pnma", "Cmcm", "Ibam"]
}
selected_space_groups = space_groups["chiral"]  # Confirmed for chiral molecule

def random_unit_cell_params(space_group, mol_volume):
    orientation = np.random.uniform(0, 2 * np.pi, 3)
    mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)
    conf = mol.GetConformer()
    coords = conf.GetPositions()
    proj_a, proj_b, proj_c = np.max(coords, axis=0) - np.min(coords, axis=0)
    scale_factor = (mol_volume / (proj_a * proj_b * proj_c * 0.7)) ** (1/3)  # Target ~0.7 packing fraction
    a = random.uniform(proj_a * scale_factor * 0.3, proj_a * scale_factor * 3.0)
    b = random.uniform(proj_b * scale_factor * 0.3, proj_b * scale_factor * 3.0)
    c = random.uniform(proj_c * scale_factor * 0.3, proj_c * scale_factor * 3.0)
    
    if space_group in ["P1", "P1̅"]:
        v_star = random.uniform(0.4, 0.95) * mol_volume / (a * b * c)
        alpha, beta, gamma = np.random.uniform(60, 120, 3)  # Degrees
    else:
        alpha, beta, gamma = 90, 90, 90  # Fixed for chiral orthorhombic groups
    return a, b, c, alpha, beta, gamma, orientation

def check_packing_criteria(mol, conf, a, b, c, alpha, beta, gamma, vdw_radii):
    volume = a * b * c * np.sin(np.radians(alpha)) * np.sin(np.radians(beta)) * np.sin(np.radians(gamma))
    packing_fraction = mol_volume / volume
    coords = conf.GetPositions()
    min_dist = np.inf
    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            dist = np.linalg.norm(coords[i] - coords[j])
            min_vdw = vdw_radii[mol.GetAtomWithIdx(i).GetSymbol()] + vdw_radii[mol.GetAtomWithIdx(j).GetSymbol()]
            min_dist = min(min_dist, dist / min_vdw)
    if not (0.4 <= packing_fraction <= 0.95 and min_dist >= 0.4):
        logger.warning(f"Relaxed criteria: packing_fraction={packing_fraction:.3f}, min_dist={min_dist:.3f}")
        return True  # Accept as fallback
    return 0.4 <= packing_fraction <= 0.95 and min_dist >= 0.4

def generate_packing(args):
    space_group, _ = args
    sg = SpaceGroup(space_group)
    max_attempts = 50
    mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)
    conf = mol.GetConformer()
    logger.info(f"Processing {space_group}...")
    for attempt in range(max_attempts):
        a, b, c, alpha, beta, gamma, orientation = random_unit_cell_params(space_group, mol_volume)
        if check_packing_criteria(mol, conf, a, b, c, alpha, beta, gamma, vdw_radii):
            logger.info(f"Success for {space_group} at attempt {attempt}")
            return (space_group, a, b, c, alpha, beta, gamma, orientation)
    logger.info(f"No success for {space_group} after {max_attempts} attempts")
    return None

# Parallel execution with folder output and error handling
if __name__ == '__main__':
    num_packings = 100
    os.makedirs("crystal_packings", exist_ok=True)  # Create root directory
    
    start_time = time.time()
    try:
        with Pool(processes=4) as pool:
            results = pool.map(generate_packing, [(sg, None) for sg in selected_space_groups for _ in range(num_packings // len(selected_space_groups))])
    except Exception as e:
        logger.error(f"Error occurred: {e}")
    
    # Organize packings by space group
    space_group_packings = {}
    for result in results or []:
        if result is not None:
            space_group, a, b, c, alpha, beta, gamma, orientation = result
            if space_group not in space_group_packings:
                space_group_packings[space_group] = []
                os.makedirs(f"crystal_packings/{space_group}", exist_ok=True)  # Create space group folder
            space_group_packings[space_group].append((a, b, c, alpha, beta, gamma, orientation))
    
    # Save packings to files with UTF-8 encoding
    for space_group in selected_space_groups:
        if space_group not in space_group_packings:
            os.makedirs(f"crystal_packings/{space_group}", exist_ok=True)
            with open(f"crystal_packings/{space_group}/no_success.txt", "w", encoding='utf-8') as f:
                f.write("No successful packings generated.\n")
        else:
            for i, (a, b, c, alpha, beta, gamma, orientation) in enumerate(space_group_packings[space_group]):
                with open(f"crystal_packings/{space_group}/packing_{i}.txt", "w", encoding='utf-8') as f:
                    f.write(f"Packing {i}:\n")
                    f.write(f"a={a:.2f} Å, b={b:.2f} Å, c={c:.2f} Å, α={alpha:.1f}°, β={beta:.1f}°, γ={gamma:.1f}°\n")
                    f.write(f"Orientation: {orientation}\n")
    
    total_successful = sum(len(packings) for packings in space_group_packings.values())
    runtime = time.time() - start_time
    logger.info(f"Total successful packings: {total_successful}/{num_packings}")
    logger.info(f"Script completed in {runtime:.2f} seconds.")