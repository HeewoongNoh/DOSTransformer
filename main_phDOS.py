import sys
import torch
import torch.nn as nn
import numpy as np
import random
from torch_geometric.loader import DataLoader
import utils
from utils import test_phonon, build_data, load_data, train_valid_test_split, r2


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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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
    n_bond_feat = 4
    out_dim = len(df.iloc[0]['phfreq'])
    attn_drop = args.attn_drop

    # Model selection
    if embedder == "DOSTransformer_phonon":
        from embedder_phDOS.DOSTransformer_phonon import DOSTransformer_phonon
        model = DOSTransformer_phonon(args.layers, args.transformer, n_atom_feat, n_bond_feat, n_hidden, out_dim, device).to(device)

    elif embedder == "graphnetwork":
        from embedder_phDOS.graphnetwork_phonon import Graphnetwork_phonon
        model = Graphnetwork_phonon(args.layers, n_atom_feat, n_bond_feat, n_hidden, out_dim, device).to(device)

    elif embedder == "graphnetwork2":
        from embedder_phDOS.graphnetwork_phonon import Graphnetwork2_phonon
        model = Graphnetwork2_phonon(args.layers, n_atom_feat, n_bond_feat, n_hidden, out_dim, device).to(device)

    elif embedder == "mlp":
        from embedder_phDOS.mlp_phonon import mlp_phonon
        model = mlp_phonon(args.layers, n_atom_feat, n_bond_feat, n_hidden, out_dim, device).to(device)

    elif embedder == "mlp2":
        from embedder_phDOS.mlp_phonon import mlp2_phonon
        model = mlp2_phonon(args.layers, n_atom_feat, n_bond_feat, n_hidden, out_dim, device).to(device)

    else :
        print("error occured : Inappropriate model name")
    print(model)

    f = open(f"./experiments_{embedder}.txt", "a")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.MSELoss()
    criterion_2 = nn.L1Loss()

    best_rmse = 1000
    best_mae = 1000
    num_batch = len(df.iloc[idx_train]['data'].values)//batch_size

    best_losses = list()
    for epoch in range(args.epochs):
        model.train()

        for bc, batch in enumerate(train_loader):        
            batch.to(device)

            preds_global, _, preds_system = model(batch)    #DOSTransformer output

            mse_global = criterion(preds_global, batch.phdos).cpu()
            rmse_global = torch.sqrt(mse_global).mean()

            mse_system = criterion(preds_system, batch.phdos).cpu()
            rmse_system = torch.sqrt(mse_system).mean()
            loss = rmse_global + args.beta*rmse_system
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
 
            sys.stdout.write('\r[ epoch {}/{} | batch {}/{} ] Total Loss: {:.4f}'.format(epoch + 1, args.epochs, bc + 1, num_batch, loss))
            sys.stdout.flush()


        if (epoch + 1) % args.eval == 0 :

            
            #valid
            valid_rmse, valid_mse,valid_mae,valid_r2, preds_y = test_phonon(model, valid_loader, criterion_2, r2, device)
            print("\n[ {} epochs ]valid_rmse:{:.4f}|valid_mse:{:.4f}|valid_mae:{:.4f}|valid_r2:{:.4f}".format(epoch + 1, valid_rmse, valid_mse,valid_mae,valid_r2))
            
            if valid_rmse < best_rmse and valid_mae < best_mae:
                best_rmse = valid_rmse
                best_mae = valid_mae
                best_epoch = epoch + 1 
                test_rmse, test_mse,test_mae,test_r2, preds_y= test_phonon(model, test_loader, criterion_2,r2, device)
                print("\n[ {} epochs ]System:test_rmse:{:.4f}|test_mse:{:.4f}|test_mae:{:.4f}|test_r2:{:.4f}".format(epoch + 1, test_rmse, test_mse,test_mae,test_r2))

            if valid_rmse < best_rmse and valid_mae > best_mae:
                best_rmse = valid_rmse
                best_epoch = epoch + 1
                test_rmse, test_mse,test_mae,test_r2, preds_y = test_phonon(model, test_loader, criterion_2, r2, device)
                print("\n[ {} epochs ]System:test_rmse:{:.4f}|test_mse:{:.4f}|test_mae:{:.4f}|test_r2:{:.4f}".format(epoch + 1, test_rmse, test_mse,test_mae,test_r2))

            if valid_rmse > best_rmse and valid_mae < best_mae:
                best_mae = valid_mae
                best_epoch = epoch + 1
                test_rmse, test_mse,test_mae,test_r2, preds_y = test_phonon(model, test_loader,criterion, criterion_2, r2, device)
                print("\n[ {} epochs ]System:test_rmse:{:.4f}|test_mse:{:.4f}|test_mae:{:.4f}|test_r2:{:.4f}".format(epoch + 1, test_rmse, test_mse,test_mae,test_r2))
          
            best_losses.append(best_rmse)
            st_best_sys = '**System [Best epoch: {}] Best RMSE: {:.4f}|Best MSE: {:.4f} |Best MAE: {:.4f}|Best R2: {:.4f}**\n'.format(best_epoch, test_rmse,test_mse, test_mae,test_r2)
            print(st_best_sys)
            if len(best_losses) > int(args.es / args.eval):
                if best_losses[-1] == best_losses[-int(args.es / 5)]:
                    
                    print("Early stop!!")
                    print("[Final]system {}".format(st_best_sys))
                    f.write("\n")
                    f.write("Early stop!!\n")
                    f.write(configuration)
                    f.write("\nbest epoch : {} \n".format(best_epoch))
                    f.write("best RMSE : {:.4f} \n".format(test_rmse))
                    f.write("best MSE : {:.4f} \n".format(test_mse))
                    f.write("best MAE : {:.4f} \n".format(test_mae))
                    f.write("best R2 : {:.4f} \n".format(test_r2))
                    sys.exit()
        
    print("\ntraining done!")
    print("System [Final] {}".format(st_best_sys))
    # write experimental results
    f.write("\n")
    f.write(configuration)
    f.write("\nbest epoch : {} \n".format(best_epoch))
    f.write("best RMSE : {:.4f} \n".format(test_rmse))
    f.write("best MSE : {:.4f} \n".format(test_mse))
    f.write("best MAE : {:.4f} \n".format(test_mae))
    f.write("best R2 : {:.4f} \n".format(test_r2))

    f.close()

if __name__ == "__main__" :



    main()
