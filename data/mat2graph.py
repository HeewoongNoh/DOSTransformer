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
from psy import cutoff_func
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


def load_orbs(path_orbs):
    orbs = list()

    with open(path_orbs) as json_file:
        orb_data = json.load(json_file)

        for elem in atom_nums.keys():
            if elem in orb_data.keys():
                orbs.append(list(orb_data[elem].values())[1:])
    orbs = numpy.vstack(orbs)

    for i in range(0, orbs.shape[0]):
        for j in range(0, orbs.shape[1]):
            if orbs[i, j] < -100:
                orbs[i, j] = -100

    return scale(orbs)

from ase import Atom, Atoms




def load_dataset(mp_data, dos_data, path_elem_embs=None, path_orbs='/home/users/heewoong/krict_2021/elem_orbs.json'):
    elem_attrs = load_elem_feats(path_elem_embs)
    elem_orbs = numpy.vstack([load_orbs(path_orbs), numpy.zeros((5, 36))])
    elem_feats = numpy.hstack([elem_attrs, elem_orbs])
    dataset = list()

    error = 0

    for i in tqdm(range(0, len(mp_data))):
    # for i in tqdm(range(0, 50)):
        mp_id = list(mp_data.keys())[i]
        str_cif = mp_data[mp_id]["cif"]

        try :
            y = torch.tensor(dos_data[mp_id]["densities_total_1"], dtype=torch.float)
            y_ft = torch.tensor(dos_data[mp_id]["densities_total_1_ft"], dtype=torch.float)
            glob = torch.tensor([mp_data[mp_id]["energy_per_atom"], mp_data[mp_id]["formation_energy_per_atom"]], dtype = torch.float)
            
            cg = get_crystal_graph(elem_feats, str_cif, n_edge_feats=n_bond_feats, radius=4)
            cg.y = y / y.max()
            cg.y_ft = y_ft / y_ft.max()
            cg.y_max = y_ft.max()
            cg.glob = glob
            cg.mp_id = mp_id
            cg.band_gap = torch.tensor(mp_data[mp_id]["band_gap"], dtype = torch.float)
            cg.efermi = torch.tensor(dos_data[mp_id]["efermi"], dtype = torch.float)
            cg.gap5 = torch.tensor((numpy.abs(numpy.diff(dos_data[mp_id]['densities_total_1_ft'] / numpy.asarray(dos_data[mp_id]['densities_total_1_ft']).max())) > 0.5), dtype = torch.float)
            cg.gap4 = torch.tensor((numpy.abs(numpy.diff(dos_data[mp_id]['densities_total_1_ft'] / numpy.asarray(dos_data[mp_id]['densities_total_1_ft']).max())) > 0.4), dtype = torch.float)
            cg.gap3 = torch.tensor((numpy.abs(numpy.diff(dos_data[mp_id]['densities_total_1_ft'] / numpy.asarray(dos_data[mp_id]['densities_total_1_ft']).max())) > 0.3), dtype = torch.float)
            cg.gap2 = torch.tensor((numpy.abs(numpy.diff(dos_data[mp_id]['densities_total_1_ft'] / numpy.asarray(dos_data[mp_id]['densities_total_1_ft']).max())) > 0.2), dtype = torch.float)
            cg.gap1 = torch.tensor((numpy.abs(numpy.diff(dos_data[mp_id]['densities_total_1_ft'] / numpy.asarray(dos_data[mp_id]['densities_total_1_ft']).max())) > 0.1), dtype = torch.float)
            cg.gap0 = torch.tensor((numpy.abs(numpy.diff(dos_data[mp_id]['densities_total_1_ft'] / numpy.asarray(dos_data[mp_id]['densities_total_1_ft']).max())) > 0.01), dtype = torch.float)
            dataset.append(cg)
        except :
            error += 1
            pass
    
    print("Converted data : {} || Total error : {}".format(len(dataset), error))

    return dataset


def get_crystal_graph(elem_feats, str_cif, n_edge_feats, radius):
    crystal = Structure.from_str(str_cif, fmt='cif')
    atom_feats, bonds, bond_feats, atom_coords, atom_attr = get_graph_info(crystal, elem_feats, n_edge_feats, radius)

    if bonds is None:
        return None

    atom_feats = torch.tensor(atom_feats, dtype=torch.float)
    bonds = torch.tensor(bonds, dtype=torch.long).t().contiguous()
    bond_feats = torch.tensor(bond_feats, dtype=torch.float)
    atom_coords = torch.tensor(atom_coords, dtype=torch.float)
    atom_attr = torch.tensor(atom_attr,dtype=torch.float)
    
    assert torch.isnan(atom_feats).sum() == 0
    assert torch.isnan(bonds).sum() == 0
    assert torch.isnan(bond_feats).sum() == 0
    assert torch.isnan(atom_coords).sum() == 0
    assert torch.isinf(atom_feats).sum() == 0
    assert torch.isinf(bonds).sum() == 0
    assert torch.isinf(bond_feats).sum() == 0
    assert torch.isinf(atom_coords).sum() == 0
    
    assert torch.isnan(atom_attr).sum() == 0
    assert torch.isinf(atom_attr).sum() == 0

    return Data(x=atom_feats, edge_index=bonds, edge_attr=bond_feats, coords = atom_coords, z=atom_attr)


def get_graph_info(crystal, elem_feats, n_bond_feats, radius):
    atom_coords, atom_feats, atom_attr = get_atom_info(crystal, elem_feats, radius)
    pdist = pairwise_distances(atom_coords)
    bonds, bond_feats = get_bond_info(pdist, n_bond_feats, radius)

    if bonds is None:
        return None, None, None

    fps = get_atom_fp(atom_coords, pdist, bonds, n_bandwidths=18)

    return numpy.hstack([atom_feats, fps]), bonds, bond_feats, atom_coords, atom_attr



def get_atom_info(crystal, elem_feats, radius):
    atoms = list(crystal.atomic_numbers)
    atom_coords = list()
    atom_feats = list()
    atom_attr = list()
    type_encoding = {}
    for Z in range(1, 101):
        specie = Atom(Z)
        type_encoding[specie.symbol] = Z - 1
    type_onehot = torch.eye(len(type_encoding))
    list_nbrs = crystal.get_all_neighbors(radius)

    coords = dict()
    for coord in list(crystal.cart_coords):
        coord_key = ','.join(list(coord.astype(str)))
        coords[coord_key] = True

    for i in range(0, len(list_nbrs)):
        nbrs = list_nbrs[i]

        for j in range(0, len(nbrs)):
            coord_key = ','.join(list(nbrs[j][0].coords.astype(str)))
            if coord_key not in coords.keys():
                coords[coord_key] = True
                atoms.append(atom_nums[nbrs[j][0].species_string])

    for coord in coords.keys():
        atom_coords.append(numpy.array([float(x) for x in coord.split(',')]))
    atom_coords = numpy.vstack(atom_coords)

    for i in range(0, len(atoms)):
        atom_feats.append(elem_feats[atoms[i] - 1, :])
        attr = type_onehot[[atoms[i]-1]]
        # print('eror here')
        atom_attr.append(attr)
    atom_attr = numpy.vstack(atom_attr).astype(float)
    # print('here')
    atom_feats = numpy.vstack(atom_feats).astype(float)

    return atom_coords, atom_feats, atom_attr


def get_bond_info(pdist, n_bond_feats, radius):
    bonds = list()
    bond_feats = list()
    rbf_means = psy.even_samples(0, 4, n_bond_feats)

    for i in range(0, pdist.shape[0]):
        for j in range(0, pdist.shape[1]):
            if i != j and 0 < pdist[i, j] <= radius:
                bonds.append([i, j])
                bond_feats.append(psy.rbf(numpy.full(n_bond_feats, pdist[i, j]), rbf_means, beta=0.5))

    if len(bonds) == 0:
        return None, None
    else:
        bonds = numpy.vstack(bonds)
        bond_feats = numpy.vstack(bond_feats)
        return bonds, bond_feats


def get_atom_fp(atom_coords, pdist, bonds, n_bandwidths, d_c=7):
    fps = list()
    bandwidths = numpy.log10(psy.even_samples(0.25, n_bandwidths, 18))
    norm_consts = (1 / (numpy.sqrt(2 * numpy.pi) * bandwidths))**3
    nn_dict = dict()

    for i in range(0, bonds.shape[0]):
        if bonds[i][0] in nn_dict.keys():
            nn_dict[bonds[i][0]].append(bonds[i][1])
        else:
            nn_dict[bonds[i][0]] = [bonds[i][1]]

    for i in range(0, atom_coords.shape[0]):
        atom_fp = list()

        if i in nn_dict.keys():
            nn_idcs = numpy.array(nn_dict[i])
            for k in range(0, n_bandwidths):
                r = pdist[i, nn_idcs]
                r_xyz = numpy.abs(atom_coords[i, :] - atom_coords[nn_idcs, :])
                coeffs = (numpy.exp(-r ** 2 / (2 * bandwidths[k] ** 2)) * cutoff_func(r, d_c))
                s_k = norm_consts[k] * numpy.sum(coeffs)
                v_k = norm_consts[k] * numpy.sum((r_xyz / r.reshape(-1, 1)) * coeffs.reshape(-1, 1), axis=0)
                v_k = numpy.sqrt(v_k[0] ** 2 + v_k[1] ** 2 + v_k[2] ** 2)
                t_k_xx = norm_consts[k] * numpy.sum(((r_xyz[:, 0] * r_xyz[:, 0]) / r ** 2) * coeffs)
                t_k_yy = norm_consts[k] * numpy.sum(((r_xyz[:, 1] * r_xyz[:, 1]) / r ** 2) * coeffs)
                t_k_zz = norm_consts[k] * numpy.sum(((r_xyz[:, 2] * r_xyz[:, 2]) / r ** 2) * coeffs)
                t_k_xy = norm_consts[k] * numpy.sum(((r_xyz[:, 0] * r_xyz[:, 1]) / r ** 2) * coeffs)
                t_k_yz = norm_consts[k] * numpy.sum(((r_xyz[:, 1] * r_xyz[:, 2]) / r ** 2) * coeffs)
                t_k_xz = norm_consts[k] * numpy.sum(((r_xyz[:, 0] * r_xyz[:, 2]) / r ** 2) * coeffs)
                t_k = t_k_xx + t_k_yy + t_k_zz
                t_k_p = t_k_xx * t_k_yy + t_k_yy * t_k_zz + t_k_xx * t_k_zz - t_k_xy ** 2 - t_k_yz ** 2 - t_k_xz ** 2
                t_k_m = numpy.array([[t_k_xx, t_k_xy, t_k_xz], [t_k_xy, t_k_yy, t_k_yz], [t_k_xz, t_k_yz, t_k_zz]])
                t_k_pp = numpy.linalg.det(t_k_m)
                atom_fp.append(numpy.array([s_k, v_k, t_k, t_k_p, t_k_pp]).reshape(1, -1))
            fps.append(numpy.hstack(atom_fp))
        else:
            fps.append(numpy.zeros(90))

    return numpy.vstack(fps)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    with open('dos_ft.pkl', "rb") as f:
        dos_data = pickle.load(f)

    with open('mp.pkl', "rb") as f:
        mp_data = pickle.load(f)

    dataset = load_dataset(mp_data, dos_data)
    torch.save(dataset,"./data/processed/dos_data_random.pt")
