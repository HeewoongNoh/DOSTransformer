
import torch
import torch.nn as nn
import numpy as np
import random
from torch_geometric.loader import DataLoader
import utils2
from utils2 import correlation
import os
import sys

from embedder import GNTransformer_phonon, gntransformer_phonon, GNTransformer_phonon_v2
from tensorboardX import SummaryWriter
from timeit import default_timer as timer
import time as local_time

# Seed Setting
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# limit CPU usage
torch.set_num_threads(2)

# for phononDoS
import pandas as pd
import torch.nn.functional as F
from ase import Atom, Atoms
from ase.neighborlist import neighbor_list
from e3nn_latest.utils.utils_data import load_data, train_valid_test_split
from e3nn_latest.utils.utils_model import Network
from torch_geometric.data import Data
import torch_scatter
import e3nn
from e3nn import o3
from typing import Dict, Union




default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

df, species = load_data('e3nn_latest/data/data.csv')
print("Dataset Loaded!")

# one-hot encoding atom type and mass
type_encoding = {}
specie_am = []
for Z in range(1, 119):
    specie = Atom(Z)
    type_encoding[specie.symbol] = Z - 1
    specie_am.append(specie.mass)

type_onehot = torch.eye(len(type_encoding))
am_onehot = torch.diag(torch.tensor(specie_am))



# build data
def build_data(entry, type_encoding, type_onehot, r_max=5.):
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
    am = am_onehot[[type_encoding[specie] for specie in symbols]]
    type = type_onehot[[type_encoding[specie] for specie in symbols]]
    summation = torch.cat((am, type), dim = 1)
    data = Data(
        pos=positions, lattice=lattice, symbol=symbols,
        x=am_onehot[[type_encoding[specie] for specie in symbols]],   # atomic mass (node feature)
        z=type_onehot[[type_encoding[specie] for specie in symbols]], # atom type (node attribute)
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
        edge_vec=edge_vec, edge_len=edge_len,
        phdos=torch.from_numpy(entry.phdos).unsqueeze(0)
    )
    
    return data


r_max = 4. # cutoff radius
df['data'] = df.apply(lambda x: build_data(x, type_encoding, type_onehot, r_max), axis=1)
print("build data")



criterion2 = nn.L1Loss()
def test(model, data_loader, criterion, device):
    model.eval()

    preds_total, y_total = None, None

    with torch.no_grad():
        test_loss = 0
        test_mae = 0

        for bc, batch in enumerate(data_loader):
            batch.to(device)
            preds = model(batch)
            
            y = batch.phdos
            loss = criterion(preds, y).cpu()
            rmse = torch.sqrt(loss).mean()
            mae = criterion2(preds, y).cpu()

            test_loss += rmse
            test_mae += mae
            
            if preds_total == None :
                preds_total = preds
                y_total = y
            
            else :
                preds_total = torch.cat([preds_total, preds], dim = 0)
                y_total = torch.cat([y_total, y], dim = 0)

        corr = correlation(y_total, preds_total)

    return test_loss/(bc + 1), test_mae/(bc+1), corr



def main():
    
    args = utils2.parse_args()
    train_config = utils2.training_config(args)
    configuration = utils2.exp_get_name7(train_config)
    print("{}".format(configuration))

    # GPU setting
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) 
    print(device)

    # Model Checkpoint Path
    CHECKPOINT_PATH = "./model_checkpoints/phonon/"
    check_dir = CHECKPOINT_PATH + configuration + "phonon_0210.pt"

    # Prediction Checkpoint Path
    # PRED_PATH = "./preds_y/"
    # pred_dir = PRED_PATH + configuration + "finetune_phononDoS_gnt.pt"

    now = local_time.localtime()
    mday = now.tm_mday
    hour = now.tm_hour
    minute = now.tm_min
    writer = SummaryWriter(log_dir="runs/finetune_e3nn_embedder({})_config({})_time({}_{}_{})".format(args.embedder, configuration, mday, hour, minute))
    idx_train, idx_valid, idx_test = train_valid_test_split(df, species, valid_size=.1, test_size=.1, seed=args.random_state, plot=False)

    # load train/valid/test indices
    with open('e3nn_latest/data/idx_train.txt', 'r') as f: idx_train = [int(i.split('\n')[0]) for i in f.readlines()]
    with open('e3nn_latest/data/idx_valid.txt', 'r') as f: idx_valid = [int(i.split('\n')[0]) for i in f.readlines()]
    with open('e3nn_latest/data/idx_test.txt', 'r') as f: idx_test = [int(i.split('\n')[0]) for i in f.readlines()]

    # format dataloaders
    batch_size = 1
    train_loader = DataLoader(df.iloc[idx_train]['data'].values, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(df.iloc[idx_valid]['data'].values, batch_size=batch_size)
    test_loader = DataLoader(df.iloc[idx_test]['data'].values, batch_size=batch_size)

    print("Dataset Loaded!")

    def get_neighbors(df, idx):
        n = []
        for entry in df.iloc[idx].itertuples():
            N = entry.data.pos.shape[0]
            for i in range(N):
                n.append(len((entry.data.edge_index[0] == i).nonzero()))
        return np.array(n)

    n_train = get_neighbors(df, idx_train)
    n_valid = get_neighbors(df, idx_valid)
    n_test = get_neighbors(df, idx_test)

    print('average number of neighbors (train/valid/test):', n_train.mean(), '/', n_valid.mean(), '/', n_test.mean())


    # embedder = "gntransformer_phonon_v2"
    embedder = "gntransformer_phonon"
    n_hidden = args.hidden
    n_atom_feat = 118
    n_bond_feat = 3
    out_dim = len(df.iloc[0]['phfreq'])
    em_dim = 64
    r_max = 4
    
    # Model selection               
    if embedder == "gntransformer_phonon":
        model = GNTransformer_phonon(args.layers, args.transformer, n_atom_feat, n_bond_feat, n_hidden, out_dim, device).to(device)
    elif embedder =='gntransformer_phonon_v2':
        model = GNTransformer_phonon_v2(args.layers, args.transformer, n_atom_feat, n_bond_feat, n_hidden, out_dim, device).to(device)
    else :
        print("error occured : Inappropriate model name")
    print(model)

    f = open("./experiments/experiments_phonondos_idx.txt".format(embedder), "a")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay) #0.05
    criterion = nn.MSELoss()
    
    train_loss = 0
    best_loss = 1000
    num_batch = 1192

    best_losses = list()

    for epoch in range(args.epochs):

        train_loss = 0
        train_mae = 0
        corr_ = 0
        start = timer()
        model.train()

        for bc, batch in enumerate(train_loader):        
            batch.to(device)
            preds = model(batch)    # dos
            mse = criterion(preds, batch.phdos).cpu()
            rmse = torch.sqrt(mse).mean()

            mae = criterion2(preds, batch.phdos).cpu()
            loss = rmse
            # loss = mse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += rmse
            # corr = correlation(batch.phdos, preds)
            
            # corr_ += corr
            sys.stdout.write('\r[ epoch {}/{} | batch {}/{} ] rmse : {:.4f} | mae : {:.4f} '.format(epoch + 1, args.epochs, bc + 1, num_batch + 1, rmse, mae))
            sys.stdout.flush()
        # writer.add_scalar("accs/train correlation", corr_ / (bc + 1), epoch + 1)
        writer.add_scalar("accs/train MSE", train_loss / (bc + 1), epoch + 1)

        if (epoch + 1) % args.eval == 0 :
            
            time = (timer() - start)/args.eval
            print("\ntraining time per epoch : {:.4f} sec".format(time))

            #valid
            valid_loss, valid_mae, valid_corr = test(model, valid_loader, criterion, device)
            print("\n[ {} epochs ]valid_rmse:{:.4f}|valid mae:{:.4f}|valid corr:{:.4f}".format(epoch + 1, valid_loss, valid_mae, valid_corr))
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch + 1
                # best_corr = valid_corr
                
                test_loss, test_mae, test_corr = test(model, test_loader, criterion, device)
                print("\n[ {} epochs ]test_rmse:{:.4f}|test mae:{:.4f}|test corr:{:.4f}".format(epoch + 1, test_loss, test_mae, test_corr))
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict()}
                torch.save(checkpoint, check_dir)
                # torch.save(preds_y, pred_dir)

            best_losses.append(best_loss)
            st_best = '** [Best epoch: {}] Best RMSE: {:.4f} | Best mae: {:.4f}| Best corr: {:.4f} **\n'.format(best_epoch, test_loss, test_mae, test_corr)
            print(st_best)

            if len(best_losses) > int(args.es / args.eval):
                if best_losses[-1] == best_losses[-int(args.es / 5)]:
                    
                    print("Early stop!!")
                    print("[Final] {}".format(st_best))
                    
                    f.write("\n")
                    f.write("Early stop!!\n")
                    f.write(configuration)
                    f.write("\nbest epoch : {} \n".format(best_epoch))
                    f.write("best RMSE : {:.4f} \n".format(test_loss))
                    f.write("best mae : {:.4f} \n".format(test_mae))
                    f.write("best corr : {} \n".format(test_corr))
                    sys.exit()
        
    print("\ntraining done!")
    print("[Final] {}".format(st_best))

    # write experimental results
    f.write("\n")
    f.write(configuration)
    f.write("\nbest epoch : {} \n".format(best_epoch))
    f.write("best RMSE : {} \n".format(test_loss))
    f.write("best mae : {} \n".format(test_mae))
    f.write("best corr : {} \n".format(test_corr))
    f.close()


if __name__ == "__main__" :
    main()
