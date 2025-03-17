import numpy as np
from scipy.linalg import eig
from scipy.optimize import minimize
from ase import Atoms
from ase.io import write, read
from ase.calculators.orca import ORCA
from ase.optimize import BFGS
import random
import os

# Step 1: Define Initial Molecule
symbols = [
    'C', 'N', 'C', 'C', 'N', 'C', 'C', 'C', 'C', 'N', 'C', 'C', 'O', 'C', 'H',
    'C', 'C', 'C', 'H', 'C', 'C', 'F', 'C', 'C', 'C', 'C', 'O', 'C', 'C', 'O',
    'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
    'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
    'H', 'H', 'H', 'H'
]
positions = np.array([
    [1.805, 2.843, 0.142], [1.555, 3.044, 1.235], [2.116, 2.645, -1.284], [1.031, 1.834, -1.989],
    [0.858, 0.417, -1.540], [-0.229, -0.367, -2.222], [0.094, -0.464, -3.714], [1.344, -1.308, -3.935],
    [1.200, -2.679, -3.324], [0.709, -2.621, -1.957], [-0.364, -1.726, -1.601], [-1.674, -2.365, -1.986],
    [-1.619, -2.884, -3.299], [-2.278, -4.120, -3.678], [-3.411, -3.909, -3.814], [-2.117, -5.115, -2.549],
    [-0.737, -5.686, -2.318], [0.179, -5.878, -3.513], [0.879, -5.009, -3.602], [1.159, -7.024, -3.226],
    [1.325, -8.114, -4.071], [0.696, -8.070, -5.271], [2.209, -9.139, -3.838], [2.989, -9.096, -2.694],
    [2.871, -8.036, -1.810], [1.966, -7.012, -2.073], [1.746, -5.985, -1.181], [2.166, -4.639, -1.429],
    [1.040, -3.625, -1.105], [0.472, -3.774, -0.027], [-0.698, -5.913, -4.745], [-1.582, -4.677, -4.936],
    [2.224, 4.028, -1.930], [0.683, 0.403, -0.514], [1.759, -0.075, -1.703], [3.083, 2.129, -1.388],
    [1.276, 1.807, -3.058], [0.076, 2.365, - --1.890], [-1.171, 0.167, -2.104], [0.263, 0.536, -4.111],
    [-0.745, -0.924, -4.234], [2.196, -0.805, -3.479], [1.515, -1.414, -5.005], [2.173, -3.169, -3.327],
    [0.500, -3.259, -3.924], [-0.388, -1.484, -0.523], [-1.970, -3.100, -1.227], [-2.529, -1.621, -1.905],
    [-2.787, -5.949, -2.751], [-2.424, -4.618, -1.630], [-0.868, -6.663, -1.853], [-0.226, -5.020, -1.624],
    [2.295, -9.968, -4.538], [3.697, -9.897, -2.489], [3.485, -8.004, -0.911], [3.072, -4.414, -0.858],
    [2.462, -4.586, -2.485], [-0.148, -6.054, -5.682], [-1.365, -6.786, -4.616], [-1.028, -3.811, -5.319],
    [-2.385, -4.862, -5.665], [1.284, 4.569, -1.842], [2.468, 3.942, -2.987], [3.001, 4.619, -1.449]
])

initial_molecule = Atoms(symbols=symbols, positions=positions)

# Step 2: DFT Geometry Optimization and Conformer Generation
def optimize_geometry(molecule, method='B3LYP', basis='6-31G*'):
    mol = molecule.copy()
    calc = ORCA(label='opt', orca_command='orca', charge=0, mult=1)
    calc.parameters = {
        'input_keywords': f'! {method} {basis} Opt',
        'input_blocks': {'pal': 'nprocs 4\nend'}
    }
    mol.calc = calc
    opt = BFGS(mol)
    opt.run(fmax=0.05)  # Converge to 0.05 eV/Å
    return mol

def generate_conformers(molecule, n_conformers=5):
    conformers = [optimize_geometry(molecule)]  # Start with optimized initial structure
    for _ in range(n_conformers - 1):
        mol = molecule.copy()
        # Simple perturbation: Rotate a random bond by ±30° (assumes rigidity otherwise)
        pivot = random.randint(0, len(mol) - 1)
        axis = np.random.randn(3)
        angle = random.choice([-30, 30]) * np.pi / 180
        mol.rotate(angle, axis, center=mol.positions[pivot])
        optimized = optimize_geometry(mol)
        conformers.append(optimized)
    return conformers

def compute_single_point_energy(molecule):
    mol = molecule.copy()
    calc = ORCA(label='sp', orca_command='orca', charge=0, mult=1)
    calc.parameters = {
        'input_keywords': '! B3LYP D3BJ 6-31G*',
        'input_blocks': {'pal': 'nprocs 4\nend'}
    }
    mol.calc = calc
    energy = mol.get_potential_energy()  # eV
    return energy * 23.0605  # Convert eV to kcal/mol

# Optimize and filter conformers
print("Optimizing initial geometry...")
base_conformer = optimize_geometry(initial_molecule)
conformers = generate_conformers(base_conformer, n_conformers=5)
energies = [compute_single_point_energy(conf) for conf in conformers]
min_energy = min(energies)
filtered_conformers = [conf for conf, energy in zip(conformers, energies) if energy - min_energy <= 10]
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

# Run CrystalMath for each conformer
all_structures_p21 = []
all_structures_p212121 = []

for conf_idx, conformer in enumerate(filtered_conformers):
    print(f"Processing conformer {conf_idx + 1}/{len(filtered_conformers)}")
    principal_axes = compute_inertial_tensor(conformer)
    initial_structures_p21 = []
    initial_structures_p212121 = []
    for _ in range(10):  # Reduced for speed
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