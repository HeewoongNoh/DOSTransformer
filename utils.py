import argparse
import torch
import numpy as np
import pandas as pd
from torch_scatter import scatter_sum
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch_geometric as tg

from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase import Atom
from torch_geometric.data import Data
from sklearn.metrics import r2_score

#R square 
def r2(x1, x2):
    x1 = x1.cpu().numpy()
    x2 = x2.cpu().numpy()
    return r2_score(x1.flatten(), x2.flatten(), multioutput='variance_weighted')

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", '-d', type=int, default=6, help="GPU to use")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of Epochs for training")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch Size") 
    parser.add_argument("--layers", "-l", type=int, default=3, help="The number layers of the Processor")
    parser.add_argument("--transformer", "-t", type=int, default=2, help="The number of Transformer layers")
    parser.add_argument("--eval", type=int, default=5, help="Evaluation step")
    parser.add_argument("--es", type=int, default=50, help="Early Stopping Criteria")
    parser.add_argument("--embedder", type=str, default="DOSTransformer", help="Embedder")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden dim")
    parser.add_argument("--random_state", type=int, default=0, help = 'Random state for dataset split')
    parser.add_argument("--dataset", type=str, default='whole', help = 'Dataset: ood_crystal or ood_element or whole')
    parser.add_argument("--attn_drop", type=float, default=0.0, help = 'attention dropout ratio')
    parser.add_argument("--seed", type=int, default=0, help = 'Random seed')
    parser.add_argument("--beta", type=float, default=1.0, help = 'alpha for the spark loss2')
    return parser.parse_args()

def training_config(args):
    train_config = dict()
    for arg in vars(args):
        train_config[arg] = getattr(args,arg)
    return train_config

def exp_get_name(train_config):
    name = ''
    dic = train_config
    config = ["seed","beta", 'attn_drop',"transformer", "layers","embedder", "lr", "batch_size", "hidden","random_state","dataset"]

    for key in config:
        a = f'{key}'+f'({dic[key]})'+'_'
        name += a
    return name

def test(model, data_loader, criterion, r2, device):
    model.eval()

    preds_total, y_total, embeddings_total, mp_id_total = None, None, None, None
    preds_y = list()

    with torch.no_grad():
        loss_rmse_sys, loss_mse_sys, loss_mae_sys, loss_r2_sys = 0, 0, 0, 0
        for bc, batch in enumerate(data_loader):

            batch.to(device)
            preds_global, embeddings, preds_system  = model(batch)

            zero = torch.tensor(0,dtype=torch.float).to(device)
            y_ft = torch.where(batch.y_ft < 0, zero, batch.y_ft)
            preds_system = torch.where(preds_system<0, zero, preds_system)

            y = y_ft.reshape(len(batch.mp_id), -1)
            mse_sys = ((y - preds_system)**2).mean(dim = 1)
            rmse_sys = torch.sqrt(mse_sys)
    
            loss_mse_sys += mse_sys.mean()
            loss_rmse_sys += rmse_sys.mean()
            
            mae_sys = criterion(preds_system, y).cpu()
            loss_mae_sys += mae_sys

            r2_score_sys = r2(y, preds_system)
            loss_r2_sys += r2_score_sys

            embeddings = scatter_sum(embeddings, batch.batch, dim=0) #For dos_system embeddings

            if preds_total == None :
                mp_id_total = batch.mp_id
                preds_total = preds_system
                y_total = y
                embeddings_total = embeddings
            
            else :
                mp_id_total = mp_id_total + batch.mp_id
                preds_total = torch.cat([preds_total, preds_system], dim = 0)
                y_total = torch.cat([y_total, y], dim = 0)
                embeddings_total = torch.cat([embeddings_total, embeddings], dim = 0)

        mp_id = mp_id_total
        preds = preds_total.detach().cpu().numpy()
        y = y_total.detach().cpu().numpy()
        embeddings = embeddings_total.detach().cpu().numpy()
        preds_y.append([mp_id, preds, y, embeddings])
    
    #rmse, mse, mae, r2, predicted y
    return loss_rmse_sys/(bc + 1), loss_mse_sys/(bc+1), loss_mae_sys/(bc+1), loss_r2_sys/(bc+1), preds_y




def test_phonon(model, data_loader, criterion, r2, device):
    model.eval()

    with torch.no_grad():
        loss_rmse_sys, loss_mse_sys, loss_mae_sys, loss_r2_sys = 0, 0, 0, 0
        for bc, batch in enumerate(data_loader):
            batch.to(device)
            
            preds_global, _, preds_system  = model(batch)

            y = batch.phdos.reshape(preds_global.shape[0], -1)

            mse_sys = ((y - preds_system)**2).mean(dim = 1)
            rmse_sys = torch.sqrt(mse_sys)
    
            loss_mse_sys += mse_sys.mean()
            loss_rmse_sys += rmse_sys.mean()
            
            mae_sys = criterion(preds_system, y).cpu()
            loss_mae_sys += mae_sys

            r2_score_sys = r2(y, preds_system)
            loss_r2_sys += r2_score_sys


    
    return loss_rmse_sys/(bc + 1), loss_mse_sys/(bc+1), loss_mae_sys/(bc+1), loss_r2_sys/(bc+1)


# format progress bar
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
tqdm.pandas(bar_format=bar_format)
default_dtype = torch.float64
# torch.set_default_dtype(default_dtype)

def load_data(filename):
    # load data from a csv file and derive formula and species columns from structure
    df = pd.read_csv(filename)
    
    try:
        # structure provided as Atoms object
        df['structure'] = df['structure'].apply(eval).progress_map(lambda x: Atoms.fromdict(x))
    
    except:
        # no structure provided
        species = []

    else:
        df['formula'] = df['structure'].map(lambda x: x.get_chemical_formula())
        df['species'] = df['structure'].map(lambda x: list(set(x.get_chemical_symbols())))
        species = sorted(list(set(df['species'].sum())))

    df['phfreq'] = df['phfreq'].apply(eval).apply(np.array)
    df['phdos'] = df['phdos'].apply(eval).apply(np.array)
    df['pdos'] = df['pdos'].apply(eval)

    return df, species


def train_valid_test_split(df, species, valid_size, test_size, seed=12, plot=False):
    # perform an element-balanced train/valid/test split
    print('split train/dev ...')
    dev_size = valid_size + test_size
    stats = get_element_statistics(df, species)
    idx_train, idx_dev = split_data(stats, dev_size, seed)
    
    print('split valid/test ...')
    stats_dev = get_element_statistics(df.iloc[idx_dev], species)
    idx_valid, idx_test = split_data(stats_dev, test_size/dev_size, seed)
    idx_train += df[~df.index.isin(idx_train + idx_valid + idx_test)].index.tolist()

    print('number of training examples:', len(idx_train))
    print('number of validation examples:', len(idx_valid))
    print('number of testing examples:', len(idx_test))
    print('total number of examples:', len(idx_train + idx_valid + idx_test))
    assert len(set.intersection(*map(set, [idx_train, idx_valid, idx_test]))) == 0

    return idx_train, idx_valid, idx_test


def get_element_statistics(df, species):    
    # create dictionary indexed by element names storing index of samples containing given element
    species_dict = {k: [] for k in species}
    for entry in df.itertuples():
        for specie in entry.species:
            species_dict[specie].append(entry.Index)

    # create dataframe of element statistics
    stats = pd.DataFrame({'symbol': species})
    stats['data'] = stats['symbol'].astype('object')
    for specie in species:
        stats.at[stats.index[stats['symbol'] == specie].values[0], 'data'] = species_dict[specie]
    stats['count'] = stats['data'].apply(len)

    return stats


def split_data(df, test_size, seed):
    # initialize output arrays
    idx_train, idx_test = [], []
    
    # remove empty examples
    df = df[df['data'].str.len()>0]
    
    # sort df in order of fewest to most examples
    df = df.sort_values('count')
    
    for _, entry in tqdm(df.iterrows(), total=len(df), bar_format=bar_format):
        df_specie = entry.to_frame().T.explode('data')

        try:
            idx_train_s, idx_test_s = train_test_split(df_specie['data'].values, test_size=test_size,
                                                       random_state=seed)
        except:
            # too few examples to perform split - these examples will be assigned based on other constituent elements
            # (assuming not elemental examples)
            pass

        else:
            # add new examples that do not exist in previous lists
            idx_train += [k for k in idx_train_s if k not in idx_train + idx_test]
            idx_test += [k for k in idx_test_s if k not in idx_train + idx_test]
    
    return idx_train, idx_test


def element_representation(x, idx):
    # get fraction of samples containing given element in dataset
    return len([k for k in x if k in idx])/len(x)


# build data
def build_data(entry, r_max=5.):
    default_dtype = torch.float64
    # one-hot encoding atom type and mass
    type_encoding = {}
    specie_am = []
    for Z in range(1, 119):
        specie = Atom(Z)
        type_encoding[specie.symbol] = Z - 1
        specie_am.append(specie.mass)

    type_onehot = torch.eye(len(type_encoding))
    am_onehot = torch.diag(torch.tensor(specie_am))
    symbols = list(entry.structure.symbols).copy()
    positions = torch.from_numpy(entry.structure.positions.copy())
    lattice = torch.from_numpy(entry.structure.cell.array.copy()).unsqueeze(0)

    # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
    # edge_shift indicates whether the neighbors are in different images or copies of the unit cell
    edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=entry.structure, cutoff=r_max, self_interaction=True)
    
    # compute the relative distances and unit cell shifts from periodic boundaries
    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[torch.from_numpy(edge_src)]
    edge_vec = (positions[torch.from_numpy(edge_dst)]
                - positions[torch.from_numpy(edge_src)]
                + torch.einsum('ni,nij->nj', torch.tensor(edge_shift, dtype=default_dtype), lattice[edge_batch]))

    # compute edge lengths (rounded only for plotting purposes)
    edge_len = np.around(edge_vec.norm(dim=1).numpy(), decimals=2)
    if entry.crystal_system == "Cubic":
        system = 0
    elif entry.crystal_system == "Hexagonal":
        system = 1
    elif entry.crystal_system == "Tetragonal":
        system = 2
    elif entry.crystal_system == "Trigonal":
        system = 3
    elif entry.crystal_system == "Orthorhombic":
        system = 4
    elif entry.crystal_system == "Monoclinic":
        system = 5
    else:
        system = 6
    data = Data(
        pos=positions, lattice=lattice, symbol=symbols,
        x=am_onehot[[type_encoding[specie] for specie in symbols]],   # atomic mass (node feature)
        z=type_onehot[[type_encoding[specie] for specie in symbols]], # atom type (node attribute)
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
        edge_vec=edge_vec, edge_len=edge_len,
        phdos=torch.from_numpy(entry.phdos).unsqueeze(0),
        system = torch.tensor(system),
        mp_id = entry.mp_id
    )
    
    return data




