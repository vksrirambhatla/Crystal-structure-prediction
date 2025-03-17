import numpy as np
from scipy.linalg import eig
from scipy.optimize import minimize
from ase import Atoms
from ase.io import write
import psi4
import random
import os
import time

# Step 1: Define Initial Molecule from PDB (Neutral)
symbols = [
    'C', 'N', 'C', 'C', 'N', 'C', 'C', 'C', 'C', 'N', 'C', 'C', 'O', 'C', 'H',
    'C', 'C', 'C', 'H', 'C', 'C', 'F', 'C', 'C', 'C', 'C', 'O', 'C', 'C', 'O',
    'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
    'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
    'H', 'H', 'H'
]
positions = np.array([
    [1.451, 3.452, -0.331], [1.316, 3.635, 0.786], [1.594, 3.287, -1.785], [0.710, 2.146, -2.311],
    [1.060, 0.808, -1.817], [0.395, -0.373, -2.392], [1.061, -0.723, -3.728], [0.591, -2.051, -4.330],
    [0.693, -3.147, -3.291], [-0.148, -2.672, -2.192], [0.569, -1.595, -1.481], [0.245, -1.440, -0.002],
    [0.929, -2.338, 0.898], [0.755, -3.769, 0.996], [1.296, -4.067, 1.906], [-0.684, -4.183, 1.195],
    [-0.770, -5.679, 1.507], [-0.008, -6.622, 0.559], [0.228, -7.469, 1.217], [-0.774, -7.289, -0.596],
    [-0.904, -8.676, -0.666], [-0.350, -9.444, 0.300], [-1.569, -9.356, -1.659], [-2.166, -8.635, -2.674],
    [-2.073, -7.257, -2.673], [-1.390, -6.603, -1.654], [-1.260, -5.248, -1.605], [-1.875, -4.372, -2.543],
    [-1.446, -3.022, -2.009], [-2.237, -2.413, -1.304], [1.381, -6.045, 0.242], [1.386, -4.562, -0.125],
    [1.189, 4.605, -2.450], [2.646, 3.075, -2.032], [0.798, 2.129, -3.405], [-0.338, 2.361, -2.072],
    [1.003, 0.784, -0.800], [-0.676, -0.168, -2.546], [0.893, 0.088, -4.447], [2.143, -0.791, -3.559],
    [-0.452, -1.958, -4.656], [1.198, -2.302, -5.209], [0.368, -4.110, -3.699], [1.732, -3.251, -2.955],
    [1.642, -1.837, -1.520], [0.565, -0.433, 0.290], [-0.831, -1.470, 0.191], [-1.108, -3.622, 2.037],
    [-1.284, -3.935, 0.317], [-1.820, -5.991, 1.576], [-0.338, -5.815, 2.506], [-1.627, -10.439, -1.644],
    [-2.700, -9.146, -3.468], [-2.536, -6.685, -3.471], [-2.967, -4.475, -2.522], [-1.523, -4.550, -3.566],
    [2.005, -6.162, 1.136], [1.852, -6.633, -0.556], [0.859, -4.420, -1.068], [2.421, -4.230, -0.270],
    [0.153, 4.849, -2.224], [1.294, 4.540, -3.531], [1.815, 5.423, -2.098]
])

initial_molecule = Atoms(symbols=symbols, positions=positions)

# Step 2: DFT Geometry Optimization and Conformer Generation with Psi4 Directly
def optimize_geometry(molecule, method='b3lyp', basis='6-31g*', timeout=7200):
    mol = molecule.copy()
    print(f"Initial positions (first 5 atoms): {mol.positions[:5]}")
    geom = '\n'.join(f"{s} {x} {y} {z}" for s, (x, y, z) in zip(mol.symbols, mol.positions))
    psi4.core.clean()
    psi4.geometry(f"0 1\n{geom}")
    psi4.set_options({
        'basis': basis,
        'maxiter': 100,          # SCF iterations
        'geom_maxiter': 100,     # Geometry iterations
        'scf_type': 'df',        # Density fitting for speed
        'print': 2               # Higher verbosity
    })
    psi4.set_num_threads(4)
    start_time = time.time()
    try:
        wfn = psi4.optimize(method, return_wfn=True)[1]
        optimized_positions = wfn.molecule().geometry().np
        elapsed = time.time() - start_time
        print(f"Optimization completed in {elapsed/60:.2f} minutes")
        mol.set_positions(optimized_positions)
        print(f"Optimized positions (first 5 atoms): {mol.positions[:5]}")
        if elapsed > timeout:
            print(f"Warning: Optimization took longer than {timeout/3600} hours")
        return mol
    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        return None

def generate_conformers(molecule, n_conformers=3):
    conformers = [optimize_geometry(molecule)]
    if conformers[0] is None:
        print("Initial optimization failed, stopping conformer generation")
        return conformers
    for i in range(n_conformers - 1):
        mol = molecule.copy()
        pivot = random.randint(0, len(mol) - 1)
        axis = np.random.randn(3)
        angle = random.choice([-30, 30]) * np.pi / 180
        mol.rotate(angle, axis, center=mol.positions[pivot])
        print(f"Generating conformer {i+2}/{n_conformers}")
        optimized = optimize_geometry(mol)
        if optimized is None:
            print(f"Conformer {i+2} optimization failed, skipping")
            continue
        conformers.append(optimized)
    return conformers

def compute_single_point_energy(molecule, method='b3lyp', basis='6-31g*'):
    mol = molecule.copy()
    geom = '\n'.join(f"{s} {x} {y} {z}" for s, (x, y, z) in zip(mol.symbols, mol.positions))
    psi4.core.clean()
    psi4.geometry(f"0 1\n{geom}")
    psi4.set_options({'basis': basis, 'maxiter': 100, 'print': 2})
    psi4.set_num_threads(4)
    energy_hartree = psi4.energy(method)
    energy_ev = energy_hartree * 27.2114  # Convert Hartree to eV
    return energy_ev * 23.0605  # Convert eV to kcal/mol

print("Optimizing initial geometry with Psi4...")
base_conformer = optimize_geometry(initial_molecule)
if base_conformer is None:
    print("Initial optimization failed, exiting")
    exit(1)

print("Generating and optimizing conformers with Psi4...")
conformers = generate_conformers(base_conformer, n_conformers=3)
print("Computing single-point energies with Psi4...")
energies = []
for i, conf in enumerate(conformers):
    if conf is not None:
        energy = compute_single_point_energy(conf)
        energies.append(energy)
    else:
        energies.append(float('inf'))  # Mark failed conformers
min_energy = min(energies)
filtered_conformers = [conf for conf, energy in zip(conformers, energies) if energy - min_energy <= 10 and conf is not None]
print(f"Retained {len(filtered_conformers)} conformers within 10 kcal/mol of minimum ({min_energy:.2f} kcal/mol)")

# Step 3: CrystalMath Prediction with Conformers
def compute_inertial_tensor(molecule):
    masses = {'C': 12.01, 'H': 1.008, 'N': 14.01, 'O': 16.00, 'F': 19.00}
    coords = molecule.positions
    mass_array = np.array([masses[s] for s in molecule.symbols])
    com = np.sum(coords * mass_array[:, np.newaxis], axis=0) / np.sum(mass_array)
    coords -= com
    I = np.zeros((3, 3))
    for i in range(len(coords)):
        r = coords[i]
        I += mass_array[i] * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    eigenvalues, eigenvectors = eig(I)
    return eigenvectors.T

n_max = 5
directions = []
for nu in range(-n_max, n_max + 1):
    for nv in range(-n_max, n_max + 1):
        for nw in range(-n_max, n_max + 1):
            if nu * nv * nw == 0 and max(abs(nu), abs(nv), abs(nw)) == n_max:
                directions.append([nu, nv, nw])

def solve_orthogonality_p21(principal_axes, direction):
    n_c = np.array(direction, dtype=float)
    e1, e2, e3 = principal_axes
    extents = np.max(positions, axis=0) - np.min(positions, axis=0)
    initial_guess = [extents[0] * 2.0, extents[1] * 2.0, extents[2] * 2.0, 100.0]

    def orthogonality_error(params):
        a, b, c, beta_deg = params
        beta = np.radians(beta_deg)
        H = np.array([[a, 0, 0], [0, b, 0], [c * np.cos(beta), 0, c * np.sin(beta)]])
        ortho_error = sum(np.dot(e, H @ n_c)**2 for e in [e1, e2, e3])
        volume = a * b * c * np.sin(beta)
        volume_penalty = 1000 * max(0, 5000 - volume)
        return ortho_error + volume_penalty

    result = minimize(orthogonality_error, initial_guess, bounds=[(10, 25), (10, 30), (10, 25), (90, 120)])
    a, b, c, beta_deg = result.x
    beta = np.radians(beta_deg)
    H = np.array([[a, 0, 0], [0, b, 0], [c * np.cos(beta), 0, c * np.sin(beta)]])
    print(f"P2_1 Optimized H: a={a:.2f}, b={b:.2f}, c={c:.2f}, beta={beta_deg:.2f}, error={result.fun:.6f}")
    return H, result.fun

def solve_orthogonality_p212121(principal_axes, direction):
    n_c = np.array(direction, dtype=float)
    e1, e2, e3 = principal_axes
    extents = np.max(positions, axis=0) - np.min(positions, axis=0)
    initial_guess = [extents[0] * 3.0, extents[1] * 3.0, extents[2] * 3.0]

    def orthogonality_error(params):
        a, b, c = params
        H = np.array([[a, 0, 0], [0, b, 0], [0, 0, c]])
        ortho_error = sum(np.dot(e, H @ n_c)**2 for e in [e1, e2, e3])
        volume = a * b * c
        volume_penalty = 1000 * max(0, 10000 - volume)
        return ortho_error + volume_penalty

    result = minimize(orthogonality_error, initial_guess, bounds=[(15, 30), (20, 40), (15, 30)])
    a, b, c = result.x
    H = np.array([[a, 0, 0], [0, b, 0], [0, 0, c]])
    print(f"P2_12_12_1 Optimized H: a={a:.2f}, b={b:.2f}, c={c:.2f}, error={result.fun:.6f}")
    return H, result.fun

all_structures_p21 = []
all_structures_p212121 = []

for conf_idx, conformer in enumerate(filtered_conformers):
    print(f"Processing conformer {conf_idx + 1}/{len(filtered_conformers)}")
    principal_axes = compute_inertial_tensor(conformer)
    initial_structures_p21 = []
    initial_structures_p212121 = []
    for _ in range(10):
        dir_idx = random.randint(0, len(directions) - 1)
        H_p21, error_p21 = solve_orthogonality_p21(principal_axes, directions[dir_idx])
        H_p212121, error_p212121 = solve_orthogonality_p212121(principal_axes, directions[dir_idx])
        initial_structures_p21.append({'H': H_p21, 'ortho_error': error_p21, 'orientation': principal_axes})
        initial_structures_p212121.append({'H': H_p212121, 'ortho_error': error_p212121, 'orientation': principal_axes})

    def apply_p21_symmetry(positions, H):
        shifted_positions = positions - np.mean(positions, axis=0) + np.array([0, H[1, 1] * 0.25, 0])
        frac_coords = np.linalg.inv(H) @ shifted_positions.T
        sym_ops = [lambda x: x, lambda x: np.array([-x[0], x[1] + 0.5, -x[2]])]
        all_positions = []
        for op in sym_ops:
            sym_coords = np.apply_along_axis(op, 0, frac_coords)
            cart_coords = (H @ sym_coords).T
            all_positions.append(cart_coords)
        return np.vstack(all_positions)

    def apply_p212121_symmetry(positions, H):
        shifted_positions = positions - np.mean(positions, axis=0) + np.array([H[0, 0] * 0.25, H[1, 1] * 0.25, 0])
        frac_coords = np.linalg.inv(H) @ shifted_positions.T
        sym_ops = [
            lambda x: x,
            lambda x: np.array([x[0] + 0.5, -x[1] + 0.5, -x[2]]),
            lambda x: np.array([-x[0], x[1] + 0.5, -x[2] + 0.5]),
            lambda x: np.array([-x[0] + 0.5, -x[1], x[2] + 0.5])
        ]
        all_positions = []
        for op in sym_ops:
            sym_coords = np.apply_along_axis(op, 0, frac_coords)
            cart_coords = (H @ sym_coords).T
            all_positions.append(cart_coords)
        return np.vstack(all_positions)

    def calculate_vdw_volume(molecule):
        vdw_radii = {'C': 1.7, 'H': 1.2, 'N': 1.55, 'O': 1.52, 'F': 1.47}
        volume = 0
        for atom in molecule:
            volume += (4/3) * np.pi * vdw_radii[atom.symbol] ** 3
        return volume

    vdw_vol = calculate_vdw_volume(conformer)
    filtered_structures_p21 = []
    for struct in initial_structures_p21:
        H = struct['H']
        cell_volume = np.linalg.det(H)
        vdwfv = (vdw_vol * 2) / cell_volume * 100
        if 25 < vdwfv < 35:
            positions = apply_p21_symmetry(conformer.positions, H)
            total_score = struct['ortho_error'] + 10 * abs(vdwfv - 30)
            filtered_structures_p21.append({'H': H, 'positions': positions, 'vdwfv': vdwfv, 'score': total_score, 'conformer_idx': conf_idx})

    filtered_structures_p212121 = []
    for struct in initial_structures_p212121:
        H = struct['H']
        cell_volume = np.linalg.det(H)
        vdwfv = (vdw_vol * 4) / cell_volume * 100
        if 25 < vdwfv < 35:
            positions = apply_p212121_symmetry(conformer.positions, H)
            total_score = struct['ortho_error'] + 10 * abs(vdwfv - 30)
            filtered_structures_p212121.append({'H': H, 'positions': positions, 'vdwfv': vdwfv, 'score': total_score, 'conformer_idx': conf_idx})

    all_structures_p21.extend(filtered_structures_p21)
    all_structures_p212121.extend(filtered_structures_p212121)

# Sort and Output
all_structures_p21.sort(key=lambda x: x['score'])
all_structures_p212121.sort(key=lambda x: x['score'])

for i, struct in enumerate(all_structures_p21[:5]):
    crystal = Atoms(symbols=symbols*2, positions=struct['positions'], cell=struct['H'])
    a, b, c = np.linalg.norm(struct['H'], axis=1)
    beta = np.degrees(np.arccos(struct['H'][2, 0] / c))
    print(f"P2_1 Structure {i} (Conformer {struct['conformer_idx']}): a={a:.2f} Å, b={b:.2f} Å, c={c:.2f} Å, beta={beta:.2f}°, vdWFV={struct['vdwfv']:.2f}%, score={struct['score']:.2f}")
    write(f'crystal_p21_{i}.cif', crystal)

for i, struct in enumerate(all_structures_p212121[:5]):
    crystal = Atoms(symbols=symbols*4, positions=struct['positions'], cell=struct['H'])
    a, b, c = np.linalg.norm(struct['H'], axis=1)
    print(f"P2_12_12_1 Structure {i} (Conformer {struct['conformer_idx']}): a={a:.2f} Å, b={b:.2f} Å, c={c:.2f} Å, vdWFV={struct['vdwfv']:.2f}%, score={struct['score']:.2f}")
    write(f'crystal_p212121_{i}.cif', crystal)

print(f"Generated {len(all_structures_p21)} structures in P2_1 space group.")
print(f"Generated {len(all_structures_p212121)} structures in P2_12_12_1 space group.")