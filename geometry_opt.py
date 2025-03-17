import psi4
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

# Define the molecule (example: ethanol)
molecule = psi4.geometry("""
0 1
C      -0.687071   -0.100677    0.124159
C      -1.342061    0.319578   -1.013310
C      -1.265771    1.627741   -1.440361
C      -0.513815    2.535180   -0.714689
C       0.157601    2.125055    0.425299
C       0.073179    0.805538    0.842657
O       0.889237    3.073489    1.144264
C       2.207911    3.352203    0.906241
N       2.876685    2.717869   -0.046083
C       4.839193    4.006836    0.556804
C       4.174507    3.020959   -0.241587
C       2.761338    4.317924    1.742252
C       4.092612    4.635204    1.553022
C       6.268356    4.446060    0.399745
N       6.856999    4.259600   -0.810963
C       8.198746    4.468859   -1.221704
C       8.558397    4.206391   -2.537452
C      10.684509    4.831006   -2.125917
N       9.797194    4.377349   -3.005348
C       9.224783    4.935568   -0.404852
N      10.471467    5.122761   -0.846568
O       6.811933    4.992224    1.355846
N       4.776710    2.258898   -1.161786
Cl     -2.286888   -0.822868   -1.925953
H      -0.763095   -1.132602    0.450618
H      -1.790457    1.943182   -2.336144
H      -0.450493    3.568696   -1.038819
H       0.597396    0.484286    1.736597
H       2.170293    4.802547    2.512337
H       4.560781    5.384411    2.182887
H       6.234650    3.965963   -1.561167
H       7.795425    3.843418   -3.218150
H      11.694829    4.979165   -2.492698
H       9.024287    5.155700    0.636416
H       4.224071    1.592913   -1.701608
H       5.790661    2.265578   -1.261459

""")

# Save the initial geometry
initial_xyz = molecule.save_string_xyz()

# Set the computational method and basis set
psi4.set_options({
    'basis': '6-31G(d)',   # Basis set
    'scf_type': 'df',      # Density fitting for SCF
    'e_convergence': 1e-6, # Energy convergence criterion
    'd_convergence': 1e-6, # Density convergence criterion
    'maxiter': 200,        # Maximum number of iterations
    'geom_maxiter': 200,   # Maximum number of geometry optimization iterations
    'opt_coordinates': 'cartesian',  # Use Cartesian coordinates for optimization
    'g_convergence': 'qchem'  # Use a valid convergence criterion
})

# Perform the geometry optimization using DFT (B3LYP functional)
energy, wavefunction = psi4.optimize('b3lyp/6-31G(d)', return_wfn=True)

# Save the optimized geometry
optimized_xyz = molecule.save_string_xyz()

# Print the optimized geometry and final energy
print("Optimized Geometry (in Angstroms):")
print(optimized_xyz)
print(f"Final Energy: {energy} Hartree")

# Function to manually parse XYZ string and create RDKit molecule object
def parse_xyz_to_rdkit_mol(xyz_str):
    lines = xyz_str.strip().split('\n')
    num_atoms = int(lines[0])  # First line is the number of atoms
    atom_lines = lines[2:2+num_atoms]  # Skip the first two lines
    
    mol = Chem.RWMol()
    conf = Chem.Conformer(num_atoms)
    
    for i, line in enumerate(atom_lines):
        parts = line.split()
        atom_symbol = parts[0]
        x, y, z = map(float, parts[1:])
        
        atom = Chem.Atom(atom_symbol)
        idx = mol.AddAtom(atom)
        conf.SetAtomPosition(idx, (x, y, z))
    
    mol.AddConformer(conf)
    AllChem.Compute2DCoords(mol)
    return mol

# Convert initial and optimized geometries to RDKit molecules
try:
    initial_mol = parse_xyz_to_rdkit_mol(initial_xyz)
    optimized_mol = parse_xyz_to_rdkit_mol(optimized_xyz)
except Exception as e:
    print(f"Error: {e}")
    initial_mol = None
    optimized_mol = None

# Draw the initial and optimized molecules side by side if conversion was successful
if initial_mol and optimized_mol:
    img = Draw.MolsToGridImage([initial_mol, optimized_mol], molsPerRow=2, subImgSize=(300,300), legends=["Initial Geometry", "Optimized Geometry"])
    img.save("molecule_visualization.png")
    print("Visualization saved as molecule_visualization.png")
else:
    print("Failed to create RDKit molecules from XYZ blocks.")
