import sys
import torch
import torch.nn as nn
import numpy as np
import random
from torch_geometric.loader import DataLoader
from utils import parse_args,training_config,exp_get_name
from utils import test, r2


# limit CPU usage
torch.set_num_threads(2)

def main():
    
    args = parse_args()
    train_config = training_config(args)
    configuration = exp_get_name(train_config)
    print("{}".format(configuration))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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
    valid_loader = DataLoader(valid_dataset, batch_size = 1)
    test_loader = DataLoader(test_dataset, batch_size = 1)
    
    print("Dataset Loaded!")

    embedder = args.embedder
    n_hidden = args.hidden
    n_atom_feat = train_dataset[0].x.shape[1]
    n_bond_feat = train_dataset[0].edge_attr.shape[1]
    n_glob_feat = train_dataset[0].glob.shape[0]
    attn_drop = args.attn_drop
    # Model selection
    if embedder =='DOSTransformer':
        from embedder_eDOS.DOSTransformer import DOSTransformer
        model = DOSTransformer(args.layers, args.transformer, n_atom_feat, n_bond_feat, n_glob_feat, n_hidden, device, attn_drop).to(device)

    elif embedder == "graphnetwork":
        from embedder_eDOS.graphnetwork import Graphnetwork
        model = Graphnetwork(args.layers, n_atom_feat, n_bond_feat,n_glob_feat, n_hidden, device).to(device)

    elif embedder == "graphnetwork2":
        from embedder_eDOS.graphnetwork import Graphnetwork2
        model = Graphnetwork2(args.layers, n_atom_feat, n_bond_feat,n_glob_feat, n_hidden, device).to(device)

    elif embedder == "mlp":
        from embedder_eDOS.mlp import mlp
        model = mlp(args.layers, n_atom_feat, n_bond_feat,n_glob_feat, n_hidden, device).to(device)

    elif embedder == "mlp2":
        from embedder_eDOS.mlp import mlp2
        model = mlp2(args.layers, n_atom_feat, n_bond_feat, n_glob_feat, n_hidden, device).to(device)

    else :
        print("error occured : Inappropriate model name")
    print(model)

    f = open(f"./experiments_{args.embedder}.txt", "a")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion_2 = nn.L1Loss()

    best_rmse = 1000
    best_mae = 1000
    num_batch = int(len(train_dataset)/args.batch_size)
    best_losses = list()

    for epoch in range(args.epochs):

        model.train()

        for bc, batch in enumerate(train_loader):        
            batch.to(device)

            batch.to(device)
            preds_global, _, preds_system = model(batch)   #DOSTransformer output

            zero = torch.tensor(0,dtype=torch.float).to(device)
            y_ft = torch.where(batch.y_ft < 0, zero, batch.y_ft)
            y = y_ft.reshape(len(batch.mp_id), -1)

            #For dos global 
            global_mse = ((y - preds_global)**2).mean(dim = 1)
            global_rmse = torch.sqrt(global_mse).mean()

            #For dos system 
            system_mse = ((y - preds_system)**2).mean(dim = 1)
            system_rmse = torch.sqrt(system_mse).mean()

            loss = global_rmse + args.beta*system_rmse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            sys.stdout.write('\r[ epoch {}/{} | batch {}/{} ]  Total Loss: {:.4f} '.format(epoch + 1, args.epochs, bc + 1, num_batch + 1, loss))
            sys.stdout.flush()


        if (epoch + 1) % args.eval == 0 :

            
            #valid
            valid_rmse, valid_mse,valid_mae,valid_r2, preds_y = test(model, valid_loader,criterion_2, r2, device)
            print("\n[ {} epochs ]valid_rmse:{:.4f}|valid_mse:{:.4f}|valid_mae:{:.4f}|valid_r2:{:.4f}".format(epoch + 1, valid_rmse, valid_mse,valid_mae,valid_r2))
            
            if valid_rmse < best_rmse and valid_mae < best_mae:
                best_rmse = valid_rmse
                best_mae = valid_mae
                best_epoch = epoch + 1 
                test_rmse, test_mse,test_mae,test_r2, preds_y= test(model, test_loader, criterion_2,r2, device)
                print("\n[ {} epochs ]System:test_rmse:{:.4f}|test_mse:{:.4f}|test_mae:{:.4f}|test_r2:{:.4f}".format(epoch + 1, test_rmse, test_mse,test_mae,test_r2))

            if valid_rmse < best_rmse and valid_mae > best_mae:
                best_rmse = valid_rmse
                best_epoch = epoch + 1
                test_rmse, test_mse,test_mae,test_r2, preds_y = test(model, test_loader, criterion_2, r2, device)
                print("\n[ {} epochs ]System:test_rmse:{:.4f}|test_mse:{:.4f}|test_mae:{:.4f}|test_r2:{:.4f}".format(epoch + 1, test_rmse, test_mse,test_mae,test_r2))

            if valid_rmse > best_rmse and valid_mae < best_mae:
                best_mae = valid_mae
                best_epoch = epoch + 1
                test_rmse, test_mse,test_mae,test_r2, preds_y = test(model, test_loader, criterion_2, r2, device)
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
