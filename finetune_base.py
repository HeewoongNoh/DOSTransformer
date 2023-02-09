
import torch
import torch.nn as nn
import numpy as np
import random
from torch_geometric.loader import DataLoader
import utils2
from utils2 import correlation
import os
import sys

from embedder import GNTransformer_joint
from tensorboardX import SummaryWriter
from timeit import default_timer as timer
import time as local_time

from torch_scatter import scatter_mean, scatter_sum

# Seed Setting
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# limit CPU usage
torch.set_num_threads(2)


def test(model, data_loader, criterion_2, device):
    model.eval()

    preds_total, y_total, embeddings_total, mp_id_total = None, None, None, None
    preds_y = list()

    with torch.no_grad():
        loss = 0
        loss_mae = 0
        for bc, batch in enumerate(data_loader):
            batch.to(device)
            preds, embeddings, _= model(batch)
            y = batch.y_ft.reshape(len(batch.mp_id), -1)
            mse = ((y - preds)**2).mean(dim = 1)
            rmse = torch.sqrt(mse)
            loss += rmse.mean()
            
            mae = criterion_2(preds, y)
            loss_mae += mae
            
            embeddings = scatter_sum(embeddings, batch.batch, dim=0)

            if preds_total == None :
                mp_id_total = batch.mp_id
                preds_total = preds
                y_total = y
                embeddings_total = embeddings
            
            else :
                mp_id_total = mp_id_total + batch.mp_id
                preds_total = torch.cat([preds_total, preds], dim = 0)
                y_total = torch.cat([y_total, y], dim = 0)
                embeddings_total = torch.cat([embeddings_total, embeddings], dim = 0)

        corr = correlation(y_total, preds_total)

        mp_id = mp_id_total
        preds = preds_total.detach().cpu().numpy()
        y = y_total.detach().cpu().numpy()
        embeddings = embeddings_total.detach().cpu().numpy()
        preds_y.append([mp_id, preds, y, embeddings])
    
    return loss/(bc + 1), corr, loss_mae/(bc+1), preds_y




def main():
    
    args = utils2.parse_args()
    train_config = utils2.training_config(args)
    configuration = utils2.exp_get_name6(train_config)
    if args.OOD:
        configuration = "OOD_species_" + configuration
    print("{}".format(configuration))

    # GPU setting
    args.device = 6
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print(device)

    # Model Checkpoint Path
    CHECKPOINT_PATH = "./model_checkpoints/renew/"
    check_dir = CHECKPOINT_PATH + configuration + "rs_1_only_v2.pt"

    # Prediction Checkpoint Path
    PRED_PATH = "./preds_y/renew/"
    pred_dir = PRED_PATH + configuration + "rs_1_only_v2.pt"

    now = local_time.localtime()
    mday = now.tm_mday
    hour = now.tm_hour
    minute = now.tm_min
    writer = SummaryWriter(log_dir="runs/finetune_embedder({})_config({})_time({}_{}_{})".format(args.embedder, configuration, mday, hour, minute))

    # 8:1:1 = train:valid:test dataset random split
    from sklearn.model_selection import train_test_split

    # Load dataset
    if args.OOD == False:
        if args.dataset == 'dos2':
            dataset = torch.load('./data/processed/data_v6_normalized_dos2.pt')
        elif args.dataset == 'dos3':
            dataset = torch.load('./data/processed/data_v6_normalized_dos3.pt')
        elif args.dataset == 'dos4':
            dataset = torch.load('./data/processed/data_v6_normalized_dos4.pt')  
        else:
            dataset = torch.load("./data/processed/data_pred_bandgap.pt")
            
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

    else:
        dataset_train = torch.load("./data/processed/data_v6_normalized_dos2_3.pt")
        dataset_test = torch.load("./data/processed/data_v6_normalized_dos4.pt")
        
        train_dataset = dataset_train
        valid_dataset, test_dataset = train_test_split(dataset_test, test_size = 0.5, random_state = args.random_state)
        print(f'train_dataset_len:{len(train_dataset)}')
        print(f'valid_dataset_len:{len(valid_dataset)}')
        print(f'test_dataset_len:{len(test_dataset)}')
        
        train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size = args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size = args.batch_size)
        
        print("Dataset Loaded!")

    

    embedder = args.embedder.lower()
    n_hidden = args.hidden
    n_atom_feat = train_dataset[0].x.shape[1]
    n_bond_feat = train_dataset[0].edge_attr.shape[1]
    n_glob_feat = train_dataset[0].glob.shape[0]
    dim_out = train_dataset[0].y.shape[0]
    
    # Model selection
    if embedder == "gntransformer":
        model = GNTransformer_joint(args.layers, args.transformer, n_atom_feat, n_bond_feat, n_glob_feat, n_hidden, dim_out, device).to(device)
    else :
        print("error occured : Inappropriate model name")
    print(model)

    # ## Load pretrained model
    # if args.preepoch == 45:
        
    #     pretrained_dict = torch.load("./model_checkpoints/pretrain/layers(3)_lr(0.0001)_batch_size(8)_hidden(64)_efermi_after_masking{}epoch.pt".format(args.preepoch), map_location=device)
    # elif args.preepoch == 100:
    # # pretrained_dict = torch.load("./model_checkpoints/pretrain/layers(3)_lr(0.001)_batch_size(8)_hidden(64)_bandgap_only{}epoch.pt".format(args.preepoch), map_location = device)
    #     pretrained_dict = torch.load("./model_checkpoints/pretrain/layers(3)_lr(0.0001)_batch_size(8)_hidden(64)_efermi_after_coord_pred{}epoch.pt".format(args.preepoch), map_location=device)


    # f = open("./experiments_{}.txt".format(embedder), "a")
    f = open("./experiments/experiments_finetune_OOD_species.txt".format(embedder), "a")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.MSELoss()
    criterion_2 = nn.L1Loss()
    spark_criterion = nn.BCELoss()

    train_loss = 0
    best_loss = 1000
    num_batch = int(len(train_dataset)/args.batch_size)

    best_losses = list()
    corrs = dict()
    for epoch in range(args.epochs):

        train_loss = 0
        corr_ = 0
        start = timer()
        model.train()

        for bc, batch in enumerate(train_loader):        
            batch.to(device)
            preds, _, x = model(batch)    # dos, x, spark
            y = batch.y_ft.reshape(len(batch.mp_id), -1)
            
            mse = ((y - preds)**2).mean(dim = 1)
            rmse = torch.sqrt(mse).mean()
            mae = criterion_2(preds, y)

            loss = rmse 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss

            corr = correlation(y, preds)
            corr_ += corr
            sys.stdout.write('\r[ epoch {}/{} | batch {}/{} ] rmse : {:.4f} | correlation : {:.4f} | bg_mae : {:.4f}'.format(epoch + 1, args.epochs, bc + 1, num_batch + 1, rmse, corr,mae))
            # sys.stdout.write('\r[ epoch {}/{} | batch {}/{} ] train_loss : {:.4f} | correlation : {:.4f} '.format(epoch + 1, args.epochs, bc + 1, num_batch + 1, rmse, corr))
            sys.stdout.flush()

        writer.add_scalar("accs/train correlation", corr_ / (bc + 1), epoch + 1)
        writer.add_scalar("accs/train MSE", train_loss / (bc + 1), epoch + 1)

        if (epoch + 1) % args.eval == 0 :
            
            time = (timer() - start)
            print("\ntraining time per epoch : {:.4f} sec".format(time))
            
            #valid
            valid_loss, valid_corr, valid_mae,preds_y = test(model, valid_loader, criterion_2, device)
            # print("\n[ {} epochs ] valid_loss : {:.4f} | valid correlation : {:.4f}".format(epoch + 1, valid_loss, valid_corr))
            print("\n[ {} epochs ]valid_rmse:{:.4f}|valid correlation:{:.4f}|valid_mae:{:.4f}".format(epoch + 1, valid_loss, valid_corr,valid_mae))

            
            if valid_loss < best_loss:
                best_loss = valid_loss
                
                best_epoch = epoch + 1
                best_corr = valid_corr
                
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict()}
                test_loss, test_corr, test_mae, preds_y = test(model, test_loader, criterion_2, device)
                # print("\n[ {} epochs ] test_loss : {:.4f} | test correlation : {:.4f}".format(epoch + 1, test_loss, test_corr))
                print("\n[ {} epochs ]test_rmse:{:.4f}|test correlation:{:.4f}|test_mae:{:.4f}".format(epoch + 1, test_loss, test_corr, test_mae))

                corrs[best_epoch] = test_corr
                torch.save(checkpoint, check_dir)
                torch.save(preds_y, pred_dir)
                

 
            writer.add_scalar("accs/MSE", valid_loss, epoch + 1)
            writer.add_scalar("accs/correlation", valid_corr, epoch + 1)
            writer.add_scalar("accs/best MSE", best_loss, epoch + 1)
            writer.add_scalar("time/training time", time, epoch + 1)

            best_losses.append(best_loss)
            st_best = '** [Best epoch: {}] Best RMSE: {:.4f} | Best Corr: {:.4f} | Best MAE: {:.4f} **\n'.format(best_epoch, test_loss, test_corr, test_mae)
            print(st_best)

            if len(best_losses) > int(args.es / args.eval):
                if best_losses[-1] == best_losses[-int(args.es / 5)]:
                    
                    print("Early stop!!")
                    print("[Final] {}".format(st_best))
                    corrs_sorted = sorted(corrs.items(),key=lambda x:x[1], reverse=True )
                    lowest_corr = corrs_sorted[-1]
                    f.write("\n")
                    f.write("Early stop!!\n")
                    f.write(configuration)
                    f.write("\nbest epoch : {} \n".format(best_epoch))
                    f.write("best RMSE : {:.4f} \n".format(test_loss))
                    f.write("best Corr : {:.4f} \n".format(test_corr))
                    f.write("best MAE : {:.4f} \n".format(test_mae))
                    f.write("best Corr on the test : {} \n".format(lowest_corr))
                    sys.exit()
        
    print("\ntraining done!")
    print("[Final] {}".format(st_best))
    corrs_sorted = sorted(corrs.items(),key=lambda x:x[1], reverse=True )
    lowest_corr = corrs_sorted[-1]
    # write experimental results
    f.write("\n")
    f.write(configuration)
    f.write("\nbest epoch : {} \n".format(best_epoch))
    f.write("best RMSE : {} \n".format(test_loss))
    f.write("best Corr : {} \n".format(test_corr))
    f.write("best MAE : {} \n".format(test_mae))
    f.write("best Corr on the test : {} \n".format(lowest_corr))
    f.close()


if __name__ == "__main__" :
    main()
