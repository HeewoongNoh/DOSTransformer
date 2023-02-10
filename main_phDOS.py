import sys
import torch
import torch.nn as nn
import numpy as np
import random
from torch_geometric.loader import DataLoader
import utils
from utils import test_phonon, build_data, load_data, train_valid_test_split
from embedder_phDOS import DOSTransformer_phonon, Graphnetwork_phonon, Graphnetwork2_phonon, mlp_phonon, mlp2_phonon
# Seed Setting
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# limit CPU usage
torch.set_num_threads(2)

# Default data type float 64 for phdos
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

df, species = load_data('./data/processed/data.csv')
print("Dataset Loaded!")

r_max = 4. # cutoff radius
df['data'] = df.apply(lambda x: build_data(x, r_max), axis=1)
print("build data")


def main():
    
    args = utils.parse_args()
    train_config = utils.training_config(args)
    configuration = utils.exp_get_name(train_config)
    print("{}".format(configuration))

    # GPU setting
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print(device)

    # Load dataset
    idx_train, idx_valid, idx_test = train_valid_test_split(df, species, valid_size=.1, test_size=.1, seed=args.random_state, plot=False)

    # load train/valid/test indices
    with open('./data/processed/idx_train.txt', 'r') as f: idx_train = [int(i.split('\n')[0]) for i in f.readlines()]
    with open('./data/processed/idx_valid.txt', 'r') as f: idx_valid = [int(i.split('\n')[0]) for i in f.readlines()]
    with open('./data/processed/idx_test.txt', 'r') as f: idx_test = [int(i.split('\n')[0]) for i in f.readlines()]

    # Format dataloaders
    batch_size = 1
    train_loader = DataLoader(df.iloc[idx_train]['data'].values, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(df.iloc[idx_valid]['data'].values, batch_size=batch_size)
    test_loader = DataLoader(df.iloc[idx_test]['data'].values, batch_size=batch_size)

    print("Dataset Loaded!")
    embedder = args.embedder.lower()
    n_hidden = args.hidden
    n_atom_feat = 118
    n_bond_feat = 3
    out_dim = len(df.iloc[0]['phfreq'])

    # Model selection
    if embedder == "gntransformer_phonon":
        model = DOSTransformer_phonon(args.layers, args.transformer, n_atom_feat, n_bond_feat, n_hidden, out_dim, device).to(device)

    elif embedder == "graphnetwork":
        model = Graphnetwork_phonon(args.layers, n_atom_feat, n_bond_feat, n_hidden, out_dim, device).to(device)

    elif embedder == "graphnetwork2":
        model = Graphnetwork2_phonon(args.layers, n_atom_feat, n_bond_feat, n_hidden, out_dim, device).to(device)

    elif embedder == "mlp":
        model = mlp_phonon(args.layers, n_atom_feat, n_bond_feat, n_hidden, out_dim, device).to(device)

    elif embedder == "mlp2":
        model = mlp2_phonon(args.layers, n_atom_feat, n_bond_feat, n_hidden, out_dim, device).to(device)

    else :
        print("error occured : Inappropriate model name")
    print(model)

    f = open(f"./experiments_{embedder}.txt", "a")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.MSELoss()
    criterion2 = nn.L1Loss()

    best_loss = 1000
    num_batch = 1192

    best_losses = list()
    for epoch in range(args.epochs):
        model.train()

        for bc, batch in enumerate(train_loader):        
            batch.to(device)
            preds = model(batch)    # Predicted phdos

            mse = criterion(preds, batch.phdos).cpu()
            rmse = torch.sqrt(mse).mean()

            mae = criterion2(preds, batch.phdos).cpu()

            loss = rmse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
            sys.stdout.write('\r[ epoch {}/{} | batch {}/{} ] RMSE: {:.4f}  MAE: {:.4f} '.format(epoch + 1, args.epochs, bc + 1, num_batch, rmse, mae))
            sys.stdout.flush()


        if (epoch + 1) % args.eval == 0 :
            
            #Test on validation dataset
            valid_loss, valid_mae = test_phonon(model, valid_loader, criterion, criterion2, device)
            print("\n[ {} epochs ] Valid RMSE: {:.4f} |  Valid MAE: {:.4f}".format(epoch + 1, valid_loss, valid_mae))

        
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch + 1
                
                #Test on test dataset
                test_loss, test_mae = test_phonon(model, test_loader, criterion, criterion2, device)
                print("\n[ {} epochs ] Test RMSE : {:.4f} |  Test MAE: {:.4f}".format(epoch + 1, test_loss, test_mae))

            best_losses.append(best_loss)
            st_best = '** [Best epoch: {}] Best RMSE: {:.4f} | Best MAE: {:.4f}**\n'.format(best_epoch, test_loss, test_mae)
            print(st_best)

            if len(best_losses) > int(args.es / args.eval):
                if best_losses[-1] == best_losses[-int(args.es / 5)]:
                    
                    print("Early stop!!")
                    print("[Final] {}".format(st_best))
                    
                    f.write("\n")
                    f.write("Early stop!!\n")
                    f.write(configuration)
                    f.write("\nbest epoch : {} \n".format(best_epoch))
                    f.write("best RMSE : {} \n".format(test_loss))
                    f.write("best MAE : {} \n".format(test_mae))
                    sys.exit()
        
    print("\ntraining done!")
    print("[Final] {}".format(st_best))

    # write experimental results
    f.write("\n")
    f.write(configuration)
    f.write("\nbest epoch : {} \n".format(best_epoch))
    f.write("best RMSE : {} \n".format(test_loss))
    f.write("best MAE : {} \n".format(test_mae))
    f.close()


if __name__ == "__main__" :



    main()
