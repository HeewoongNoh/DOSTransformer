
import torch
import torch.nn as nn
import numpy as np
import random
from torch_geometric.loader import DataLoader
import utils_mlp
from utils_mlp import correlation
import os
import sys

from embedder import GNTransformer_joint2, GNTransformer_joint
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


def test(model, data_loader, criterion, device):
    model.eval()

    preds_total, y_total, embeddings_total, mp_id_total = None, None, None, None
    preds_y = list()

    with torch.no_grad():
        loss = 0
        loss_mae = 0
        for bc, batch in enumerate(data_loader):
            batch.to(device)
            preds, embeddings, _ = model(batch)
            y = batch.y_ft.reshape(len(batch.mp_id), -1)
            mse = ((y - preds)**2).mean(dim = 1)
            rmse = torch.sqrt(mse)
            loss += rmse.mean()
            
            mae = criterion(preds,y)
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
    
    return loss/(bc + 1), corr,loss_mae/(bc+1), preds_y




def main():
    
    args = utils_mlp.parse_args()
    # args.train_joint = True
    train_config = utils_mlp.training_config(args)
    if args.train_joint:
        configuration = utils_mlp.exp_get_name3(train_config)
    else:
        configuration = utils_mlp.exp_get_name2(train_config)
    if args.OOD:
        configuration = "OOD_" + configuration
    print("{}".format(configuration))

    # GPU setting
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print(device)

    # Model Checkpoint Path
    CHECKPOINT_PATH = "./model_checkpoints/finetune_final/"
    check_dir = CHECKPOINT_PATH + configuration + "finetune_final.pt"

    # Prediction Checkpoint Path
    PRED_PATH = "./preds_y/finetune_final/"
    pred_dir = PRED_PATH + configuration + "finetune_final.pt"

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
            dataset = torch.load("./data/processed/data_v6_normalized.pt")
            
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
    # if embedder == "gntransformer2":
    #     model = GNTransformer_joint2(args.layers, args.transformer, n_atom_feat, n_bond_feat, n_glob_feat, n_hidden, dim_out,window_size, device).to(device)
    if embedder =='gntransformer':
        model = model = GNTransformer_joint(args.layers, args.transformer, n_atom_feat, n_bond_feat, n_glob_feat, n_hidden, dim_out, device).to(device)
    else :
        print("error occured : Inappropriate model name")
    print(model)



    #model evaluating
    criterion_2 = nn.L1Loss()
    checkpoint = torch.load('/home/users/heewoong/krict_2021/model_checkpoints/finetune_exp2/transformer(2)_layers(3)_lr(0.0001)_batch_size(8)_hidden(256)_pretrain(efermi_masking)_spark(0.1)_preepoch(40)_alpha(0.25)_sampling_ratio(1)_finetune_exp2_final.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    test_loss, test_corr, test_mae,_ = test(model, test_loader, criterion_2, device)
    print("\n[ {} epochs ] test_loss : {:.4f} | test correlation : {:.4f} | test mae : {:.4f}".format(epoch + 1, test_loss, test_corr, test_mae))

    










    # ## Load pretrained model
    # if args.pretrain == 'efermi':
    #     pretrained_dict = torch.load("./model_checkpoints/pretrain2/layers(3)_lr(0.001)_batch_size(256)_hidden(256)_efermi_only{}epoch.pt".format(args.preepoch), map_location = device)
    # elif args.pretrain == 'bandgap':
    #     pretrained_dict = torch.load("./model_checkpoints/pretrain2/layers(3)_lr(0.001)_batch_size(256)_hidden(256)_bandgap_only{}epoch.pt".format(args.preepoch), map_location = device)
    # elif args.pretrain == 'density':
    #     pretrained_dict = torch.load("./model_checkpoints/pretrain2/layers(3)_lr(0.001)_batch_size(256)_hidden(256)_density_only{}epoch.pt".format(args.preepoch), map_location = device)
    # elif args.pretrain == 'masking':
    #     pretrained_dict = torch.load("./model_checkpoints/pretrain2/layers(3)_lr(0.001)_batch_size(256)_hidden(256)_masking_ratio(0.1)_masking_2.pt{}epoch.pt".format(args.preepoch), map_location = device)
    # elif args.pretrain == 'coord_pred':
    #     pretrained_dict = torch.load("./model_checkpoints/pretrain2/layers(3)_lr(0.001)_batch_size(256)_hidden(256)_coord_pred.pt{}epoch.pt".format(args.preepoch), map_location = device)
        
    # elif args.pretrain == 'efermi_masking':
    #     pretrained_dict = torch.load("./model_checkpoints/pretrain2/layers(3)_lr(0.001)_batch_size(256)_hidden(256)_efermi_after_masking{}epoch_masking60.pt".format(args.preepoch), map_location=device)
    # # elif args.pretrain == 'efermi_coord_pred':
    # #     pretrained_dict = torch.load("./model_checkpoints/pretrain/layers(3)_lr(0.0001)_batch_size(256)_hidden(256)_efermi_after_coord_pred{}epoch.pt".format(args.preepoch), map_location=device)

    # # elif args.pretrain == 'bandgap_masking':
    # #     pretrained_dict = torch.load("./model_checkpoints/pretrain/layers(3)_lr(0.0001)_batch_size(8)_hidden(64)_efermi_after_coord_pred{}epoch.pt".format(args.preepoch), map_location=device)
    # # elif args.pretrain == 'bandgap_masking':
    # #     pretrained_dict = torch.load("./model_checkpoints/pretrain/layers(3)_lr(0.0001)_batch_size(8)_hidden(64)_efermi_after_coord_pred{}epoch.pt".format(args.preepoch), map_location=device)
    # else:
    #     print('not available pretraining task')
        
    # pretrained_dict = pretrained_dict["model_state_dict"]
    # model_dict = model.state_dict()

    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # matching = [s for s in list(pretrained_dict.keys()) if "GN_decoder" in s]
    
    # for i in range(len(matching)):
    #     del pretrained_dict[matching[i]]
    
    # assert len([s for s in list(pretrained_dict.keys()) if "GN_decoder" in s]) == 0

    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    # print(f'pretrained_{args.pretrain}_weights_loaded')
    
    # f = open("./experiments_{}.txt".format(embedder), "a")
    f = open("./experiments/experiments_finetune_exp_final.txt".format(embedder), "a")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.MSELoss()
    criterion_2 = nn.L1Loss()
    spark_criterion = nn.BCELoss()

    train_loss = 0
    best_loss = 1000
    num_batch = int(len(train_dataset)/args.batch_size)

    best_losses = list()
    for epoch in range(args.epochs):

        train_loss = 0
        corr_ = 0
        start = timer()
        model.train()

        for bc, batch in enumerate(train_loader):        
            batch.to(device)

            preds, _, spark1 = model(batch)
            # y_spark = batch.y_ft.reshape(len(batch.mp_id),-1)
            # y_spark = torch.diff(y_spark).reshape(-1)

        
            y = batch.y_ft.reshape(len(batch.mp_id), -1)
            
            # y_spark = batch.y_ft
            # zero_pd = torch.zeros(1).to(device)
            # y_spark_1 = torch.cat([y_spark, zero_pd],dim=0)
            # y_spark_2 = torch.cat([zero_pd, y_spark],dim=0)
            # minus_spark = torch.abs(torch.subtract(y_spark_2, y_spark_1))[1:-1]
            
            
            mse = ((y - preds)**2).mean(dim = 1)
            rmse = torch.sqrt(mse).mean()
            mae = criterion_2(preds, y)
            
            if args.train_joint:
                if args.spark == 0.1 :
                    gap = batch.gap1
                elif args.spark == 0.2 :
                    gap = batch.gap2
                elif args.spark == 0.3 :
                    gap = batch.gap3
                elif args.spark == 0.4 :
                    gap = batch.gap4
                elif args.spark == 0.5 :
                    gap = batch.gap5
                elif args.spark == 0.01 :
                    gap = batch.gap0
                else :
                    print("not available spark criterion")

                sampling_ratio = args.sampling_ratio
                
                sampled = torch.randint((gap == 0).nonzero().reshape(-1).size()[0], (1, (gap == 1).nonzero().reshape(-1).size()[0]*sampling_ratio)).reshape(-1)
                sampled = (gap == 0).nonzero().reshape(-1)[sampled]
                sampled = torch.cat([sampled, (gap == 1).nonzero().reshape(-1)])

                
                if len(sampled) == 0: # set the 1 
                    sampled = torch.randint((gap == 0).nonzero().reshape(-1).size()[0], (1, 1*sampling_ratio)).reshape(-1)
     
                    
                spark1 = spark1.reshape(-1)[sampled]
                spark_loss1 = spark_criterion(spark1, gap[sampled])
                loss = rmse + args.alpha * spark_loss1
                
            else:
                loss = rmse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss
            corr = correlation(y, preds)
            
            if corr.isnan():
                print("corr_nan")
                continue
            corr_ += corr

            sys.stdout.write('\r[ epoch {}/{} | batch {}/{} ] RMSE: {:.4f} | Corr: {:.4f} | MAE: {:.4f} | spark1: {:.4f}'.format(epoch + 1, args.epochs, bc + 1, num_batch + 1, rmse, corr, mae, spark_loss1))
            sys.stdout.flush()

        writer.add_scalar("accs/train correlation", corr_ / (bc + 1), epoch + 1)
        writer.add_scalar("accs/train MSE", train_loss / (bc + 1), epoch + 1)

        if (epoch + 1) % args.eval == 0 :
            
            time = (timer() - start)
            print("\ntraining time per epoch : {:.4f} sec".format(time))
            
            #valid
            valid_loss, valid_corr, valid_mae, preds_y = test(model, valid_loader, criterion_2, device)
            print("\n[ {} epochs ] valid RMSE: {:.4f} | valid Corr: {:.4f} | valid MAE: {:.4f}".format(epoch + 1, valid_loss, valid_corr, valid_mae))

            # best_loss, best_corr on the valid set
        
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch + 1
                best_corr = valid_corr
                
                test_loss, test_corr, test_mae, preds_y = test(model, test_loader, criterion_2, device)
                print("\n[ {} epochs ] test RMSE : {:.4f} | test Corr : {:.4f}| test MAE: {:.4f}".format(epoch + 1, test_loss, test_corr, test_mae))
                
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict()}
                torch.save(checkpoint, check_dir)
                torch.save(preds_y, pred_dir)
                
            writer.add_scalar("accs/RMSE", valid_loss, epoch + 1)
            writer.add_scalar("accs/correlation", valid_corr, epoch + 1)
            writer.add_scalar("accs/spark_loss1", spark_loss1, epoch + 1)
            writer.add_scalar("accs/best RMSE", test_loss, epoch + 1)
            writer.add_scalar("time/training time", time, epoch + 1)

            best_losses.append(best_loss)
            st_best = '** [Best epoch: {}] Best RMSE: {:.4f} | Best Corr: {:.4f} | Best MAE: {:.4f}**\n'.format(best_epoch, test_loss, test_corr, test_mae)
            print(st_best)

            if len(best_losses) > int(args.es / args.eval):
                if best_losses[-1] == best_losses[-int(args.es / 5)]:
                    
                    print("Early stop!!")
                    print("[Final] {}".format(st_best))
                    
                    f.write("\n")
                    f.write("Early stop!!\n")
                    f.write(configuration)
                    f.write("\nbest epoch : {} \n".format(best_epoch))
                    f.write("best MSE : {} \n".format(test_loss))
                    f.write("best Corr : {} \n".format(test_corr))
                    f.write("best MAE : {} \n".format(test_mae))
                    sys.exit()
        
    print("\ntraining done!")
    print("[Final] {}".format(st_best))

    # write experimental results
    f.write("\n")
    f.write(configuration)
    f.write("\nbest epoch : {} \n".format(best_epoch))
    f.write("best MSE : {} \n".format(test_loss))
    f.write("best Corr : {} \n".format(test_corr))
    f.write("best MAE : {} \n".format(test_mae))
    f.close()


if __name__ == "__main__" :
    main()
