import numpy
import torch
import json
import psy
from tqdm import tqdm
from pymatgen.core.structure import Structure
from torch_geometric.data import Data
from mendeleev.fetch import fetch_table
from mendeleev import element
from sklearn.preprocessing import scale
from sklearn.metrics import pairwise_distances
import pickle


atom_nums = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O':8, 'F': 9, 'Ne': 10,
             'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
             'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
             'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
             'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
             'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
             'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
             'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
             'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
             'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100}

atom_syms = {v: k for k, v in atom_nums.items()}
elem_feat_names = ['atomic_number', 'period', 'en_pauling', 'covalent_radius_bragg',
                   'electron_affinity', 'atomic_volume', 'atomic_weight', 'fusion_heat']
n_elem_feats = len(elem_feat_names) + 1
n_bond_feats = 32


def load_elem_feats(path_elem_embs=None):
    if path_elem_embs is None:
        # Get a table from Mendeleev features
        return get_mendeleev_feats()
    else:
        # Get a table of elemental features from elemental embeddings
        elem_feats = list()

        with open(path_elem_embs) as json_file:
            elem_embs = json.load(json_file)

            for elem in atom_nums.keys():
                elem_feats.append(numpy.array(elem_embs[elem]))

        return scale(numpy.vstack(elem_feats))


def get_mendeleev_feats():
    tb_atom_feats = fetch_table('elements')[:100]
    elem_feats = numpy.nan_to_num(numpy.array(tb_atom_feats[elem_feat_names]))
    ion_engs = numpy.zeros((elem_feats.shape[0], 1))

    for i in range(0, ion_engs.shape[0]):
        ion_eng = element(i + 1).ionenergies
        if 1 in ion_eng:
            ion_engs[i, 0] = element(i + 1).ionenergies[1]
        else:
            ion_engs[i, 0] = 0

    return scale(numpy.hstack([elem_feats, ion_engs]))



from ase import Atom, Atoms


def load_dataset(mp_data, dos_data,original_dataset_mp_id, path_elem_embs="./matscholar-embedding.json"):
    elem_feats = load_elem_feats(path_elem_embs)
    dataset = list()

    error = 0
    original_dataset_mp_id = set(original_dataset_mp_id)
    for i in tqdm(range(0, len(mp_data))):
        mp_id = list(mp_data.keys())[i]
        str_cif = mp_data[mp_id]["cif"]

        try :
            if mp_id in original_dataset_mp_id:
                y = torch.tensor(dos_data[mp_id]["densities_total_1"], dtype=torch.float)
                y_ft = torch.tensor(dos_data[mp_id]["densities_total_1_ft"], dtype=torch.float)
                glob = torch.tensor([mp_data[mp_id]["energy_per_atom"], mp_data[mp_id]["formation_energy_per_atom"]], dtype = torch.float)

                cg = get_crystal_graph(elem_feats, str_cif, n_edge_feats=n_bond_feats, radius=8)
                cg.y = y / y.max()
                cg.y_ft = y_ft / y_ft.max()
                cg.y_max = y_ft.max()
                cg.glob = glob
                cg.mp_id = mp_id
                cg.band_gap = torch.tensor(mp_data[mp_id]["band_gap"], dtype = torch.float)
                cg.efermi = torch.tensor(dos_data[mp_id]["efermi"], dtype = torch.float)
                
                if mp_data[mp_id]['spacegroup']['crystal_system'] == "cubic":
                    cg.system = torch.tensor(0, dtype = torch.long)
                elif mp_data[mp_id]['spacegroup']['crystal_system'] == "hexagonal":
                    cg.system = torch.tensor(1, dtype = torch.long)
                elif mp_data[mp_id]['spacegroup']['crystal_system'] == "tetragonal":
                    cg.system = torch.tensor(2, dtype = torch.long)
                elif mp_data[mp_id]['spacegroup']['crystal_system'] == "trigonal":
                    cg.system = torch.tensor(3, dtype = torch.long)
                elif mp_data[mp_id]['spacegroup']['crystal_system'] == "orthorhombic":
                    cg.system = torch.tensor(4, dtype = torch.long)
                elif mp_data[mp_id]['spacegroup']['crystal_system'] == "monoclinic":
                    cg.system = torch.tensor(5, dtype = torch.long)
                else:
                    cg.system = torch.tensor(6, dtype = torch.long)

                dataset.append(cg)

        except :
            error += 1
            pass
    
    print("Converted data : {} || Total error : {}".format(len(dataset), error))

    return dataset


def get_crystal_graph(elem_feats, str_cif, n_edge_feats, radius):
    crystal = Structure.from_str(str_cif, fmt='cif')
    atom_feats, bonds, bond_feats, atom_coords, atom_attr = get_graph_info(crystal, str_cif, elem_feats, n_edge_feats, radius)

    if bonds is None:
        return None

    atom_feats = torch.tensor(atom_feats, dtype=torch.float)
    bonds = torch.tensor(bonds, dtype=torch.long).T
    bond_feats = torch.tensor(bond_feats, dtype=torch.float)
    atom_coords = torch.tensor(atom_coords, dtype=torch.float)
    atom_attr = torch.tensor(atom_attr,dtype=torch.float)
    
    assert torch.isnan(atom_feats).sum() == 0 , "atom_feats nan"
    assert torch.isnan(bonds).sum() == 0 ,"bonds nan"
    assert torch.isnan(bond_feats).sum() == 0, "bond_feats nan"
    assert torch.isnan(atom_coords).sum() == 0, "atom_coords nan"
    assert torch.isinf(atom_feats).sum() == 0
    assert torch.isinf(bonds).sum() == 0
    assert torch.isinf(bond_feats).sum() == 0
    assert torch.isinf(atom_coords).sum() == 0


    return Data(x=atom_feats, edge_index=bonds, edge_attr=bond_feats, coords = atom_coords, z=atom_attr)


def get_graph_info(crystal, str_cif, elem_feats, n_bond_feats, radius):
    atom_coords, atom_feats, atom_attr, list_nbrs = get_atom_info(crystal, elem_feats, radius)
    pdist = pairwise_distances(atom_coords)
    bonds, bond_feats = get_bond_info(crystal, list_nbrs, str_cif, pdist, n_bond_feats, radius)

    if bonds is None:
        return None, None, None, 


    #For zero prompt 
    zeros_prompt = numpy.zeros(200).astype(float)
    # given = numpy.hstack([atom_feats, fps])
    atom_feats = numpy.vstack([atom_feats, zeros_prompt])
    return atom_feats, bonds, bond_feats, atom_coords, atom_attr


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):

        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = numpy.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):

        return numpy.exp(-(distances[..., numpy.newaxis] - self.filter)**2 /
                      self.var**2)
    
def get_atom_info(crystal, elem_feats, radius):
    atoms = list(crystal.atomic_numbers)
    atom_coords = list()
    atom_feats = list()
    atom_attr = list()
    type_encoding = {}

    #e3nn type encoding
    for Z in range(1, 101):
        specie = Atom(Z)
        type_encoding[specie.symbol] = Z - 1
    type_onehot = torch.eye(len(type_encoding))
    list_nbrs = crystal.get_all_neighbors(radius, include_index = True)

    coords = dict()
    for coord in list(crystal.cart_coords):
        coord_key = ','.join(list(coord.astype(str)))
        coords[coord_key] = True

    for coord in coords.keys():
        atom_coords.append(numpy.array([float(x) for x in coord.split(',')]))
    atom_coords = numpy.vstack(atom_coords)

    atom_feats = numpy.vstack([elem_feats[crystal[i].specie.number - 1]
                              for i in range(len(crystal))])
    
    atom_attr = numpy.vstack([type_onehot[crystal[i].specie.number - 1]
                              for i in range(len(crystal))])
    
    return atom_coords, atom_feats, atom_attr, list_nbrs

def get_bond_info(crystal, list_nbrs, str_cif, pdist, n_bond_feats, radius):
    bonds = []
    bond_feats = []
    gdf = GaussianDistance(dmin = 0.0, dmax = radius, step = 0.2)
    max_num_nbr = 12
    all_nbrs = list_nbrs
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    
    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:

            nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                            [radius + 1.] * (max_num_nbr - len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2],
                                        nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[1],
                                    nbr[:max_num_nbr])))
    
    nbr_fea_idx, nbr_fea = numpy.array(nbr_fea_idx), numpy.array(nbr_fea)
    nbr_fea = gdf.expand(nbr_fea)

    bond_feats = torch.Tensor(nbr_fea).reshape(-1, nbr_fea.shape[-1])
    nbr_fea_idx = torch.Tensor(nbr_fea_idx)
    index1 = torch.LongTensor([i for i in range(len(crystal))]).reshape(-1, 1).expand(nbr_fea_idx.shape).reshape(1, -1)
    index2 = nbr_fea_idx.to(torch.int).reshape(1,-1)
    bonds = torch.cat([index1, index2], dim = 0).T
    bonds = numpy.array(bonds)
    return bonds, bond_feats




if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    with open('dos_ft.pkl', "rb") as f:
        dos_data = pickle.load(f)

    with open('mp.pkl', "rb") as f:
        mp_data = pickle.load(f)

    dataset = load_dataset(mp_data, dos_data)
    torch.save(dataset,"./data/processed/dos_data_random.pt")
