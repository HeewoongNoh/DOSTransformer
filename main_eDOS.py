import sys
import torch
import torch.nn as nn
import numpy as np
import random
from torch_geometric.loader import DataLoader
from utils import parse_args,training_config,exp_get_name
from utils import test
from embedder_eDOS import DOSTransformer, Graphnetwork, Graphnetwork2, mlp, mlp2


# Seed Setting
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# limit CPU usage
torch.set_num_threads(2)

def main():
    
    args = parse_args()
    train_config = training_config(args)
    configuration = exp_get_name(train_config)
    print("{}".format(configuration))

    # GPU setting
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print(device)

    # 8:1:1 = train:valid:test dataset random split
    from sklearn.model_selection import train_test_split

    if args.dataset == "ood_crystal" or args.dataset =="ood_element":

        train_dataset = torch.load(f'./data/processed/train_{args.dataset}.pt')
        test_dataset = torch.load(f'./data/processed/test_{args.dataset}.pt')

        valid_dataset, test_dataset = train_test_split(test_dataset, test_size = 0.5, random_state = args.random_state)

    else:
        dataset = torch.load("./data/processed/dos_dataset_random.pt")
        train_ratio = 0.80
        validation_ratio = 0.10
        test_ratio = 0.10

        train_dataset, test_dataset = train_test_split(dataset, test_size=1 - train_ratio, random_state=args.random_state)
        valid_dataset, test_dataset = train_test_split(test_dataset, test_size=test_ratio/(test_ratio + validation_ratio), random_state= args.random_state) 

    print(f'train_dataset_len:{len(train_dataset)}')
    print(f'valid_dataset_len:{len(valid_dataset)}')
    print(f'test_dataset_len:{len(test_dataset)}')
    
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size = args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size)
    
    print("Dataset Loaded!")

    embedder = args.embedder
    n_hidden = args.hidden
    n_atom_feat = train_dataset[0].x.shape[1]
    n_bond_feat = train_dataset[0].edge_attr.shape[1]
    n_glob_feat = train_dataset[0].glob.shape[0]
    dim_out = train_dataset[0].y.shape[0]

    # Model selection
    if embedder =='DOSTransformer':
        model = DOSTransformer(args.layers, args.transformer, n_atom_feat, n_bond_feat, n_glob_feat, n_hidden, dim_out, device).to(device)

    elif embedder == "graphnetwork":
        model = Graphnetwork(args.layers, n_atom_feat, n_bond_feat,n_glob_feat, n_hidden, dim_out, device).to(device)

    elif embedder == "graphnetwork2":
        model = Graphnetwork2(args.layers, n_atom_feat, n_bond_feat,n_glob_feat, n_hidden, dim_out, device).to(device)

    elif embedder == "mlp":
        model = mlp(args.layers, n_atom_feat, n_bond_feat,n_glob_feat, n_hidden, dim_out, device).to(device)

    elif embedder == "mlp2":
        model = mlp2(args.layers, n_atom_feat, n_bond_feat, n_glob_feat, n_hidden, dim_out, device).to(device)

    else :
        print("error occured : Inappropriate model name")
    print(model)

    f = open(f"./experiments_{args.embedder}.txt", "a")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion2 = nn.L1Loss()

    best_loss = 1000
    num_batch = int(len(train_dataset)/args.batch_size)
    best_losses = list()

    for epoch in range(args.epochs):

        model.train()

        for bc, batch in enumerate(train_loader):        
            batch.to(device)

            preds = model(batch)
            y = batch.y_ft.reshape(len(batch.mp_id), -1)
            
            mse = ((y - preds)**2).mean(dim = 1)
            rmse = torch.sqrt(mse).mean()
            mae = criterion2(preds, y)  

            loss = rmse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            sys.stdout.write('\r[ epoch {}/{} | batch {}/{} ] RMSE: {:.4f}  MAE: {:.4f} '.format(epoch + 1, args.epochs, bc + 1, num_batch + 1, rmse, mae))
            sys.stdout.flush()


        if (epoch + 1) % args.eval == 0 :
            
            
            #Test on validation dataset
            valid_loss,valid_mae = test(model, valid_loader, criterion2, device)
            print("\n[ {} epochs ] valid RMSE: {:.4f} |  valid MAE: {:.4f}".format(epoch + 1, valid_loss, valid_mae))

        
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch + 1
                
                ##Test on test dataset
                test_loss, test_mae = test(model, test_loader, criterion2, device)
                print("\n[ {} epochs ] test RMSE : {:.4f} |  test MAE: {:.4f}".format(epoch + 1, test_loss, test_mae))

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
