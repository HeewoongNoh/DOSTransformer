# CUDA_LAUNCH_BLOCKING = "1"

import torch
import torch.nn as nn
import numpy as np
import random
from torch_geometric.loader import DataLoader
import utils
from utils import correlation
import os
import sys

from embedder import GNTransformer_joint, graph_encoder
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
        for bc, batch in enumerate(data_loader):
            batch.to(device)
            preds, embeddings, _ = model(batch)
            y = batch.y_ft.reshape(len(batch.mp_id), -1)
            mse = ((y - preds)**2).mean(dim = 1)
            rmse = torch.sqrt(mse)
            loss += rmse.mean()

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
    
    return loss/(bc + 1), corr, preds_y


# def get_pred(model, data_loader, device):
#     model.eval()

#     temp = list()

#     with torch.no_grad():
#         for batch in data_loader:
#             batch.to(device)
#             preds, embeddings = model(batch)
#             preds = preds.reshape(-1).detach().cpu().numpy()
#             y = batch.y_ft.detach().cpu().numpy()
#             embeddings = embeddings.detach().cpu()
#             temp.append([batch.mp_id, preds, y, embeddings])
                
#     return temp


def main():
    
    args = utils.parse_args()
    train_config = utils.training_config(args)
    configuration = utils.finetune_joint_get_name(train_config)
    print("{}".format(configuration))

    # GPU setting
    GPU_NUM = 6 # Number of GPU that you want to use
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print(device)

    # code from krict_2021 github
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    # device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
    # print(device)
    # torch.cuda.set_device(device)

    # Model Checkpoint Path
    CHECKPOINT_PATH = "./model_checkpoints/"
    check_dir = CHECKPOINT_PATH + configuration + "prac_finetune.pt"

    # Prediction Checkpoint Path
    PRED_PATH = "./preds_y/"
    pred_dir = PRED_PATH + configuration + "prac_finetune.pt"

    now = local_time.localtime()
    mday = now.tm_mday
    hour = now.tm_hour
    minute = now.tm_min
    writer = SummaryWriter(log_dir="runs/finetune_joint_embedder({})_config({})_time({}_{}_{})".format(args.embedder, configuration, mday, hour, minute))

    # Load dataset
    # os.chdir("../")
    dataset = torch.load("./data/processed/data_v6_normalized.pt")
    # os.chdir("crossmodal")
    print("Dataset Loaded!")
    '''
    os.chdir, crossmodal
    '''
    n_train = int(0.8 * len(dataset))
    train_dataset = dataset[:n_train]
    test_dataset = dataset[n_train:]

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size)

    embedder = args.embedder.lower()
    n_hidden = args.hidden
    n_atom_feat = dataset[0].x.shape[1]
    n_bond_feat = dataset[0].edge_attr.shape[1]
    n_glob_feat = dataset[0].glob.shape[0]
    dim_out = dataset[0].y.shape[0]
    
    # Model selection
    if embedder == "gntransformer":
        model = GNTransformer_joint(args.layers, args.transformer, n_atom_feat, n_bond_feat, n_glob_feat, n_hidden, dim_out, device).to(device)
    elif embedder == "graph_encoder":
        model = graph_encoder(args.layers, n_atom_feat, n_bond_feat,n_glob_feat, n_hidden,dim_out, device).to(device)
    else :
        print("error occured : Inappropriate model name")
    print(model)

    # ## Load pretrained model
    # pretrained_dict = torch.load("./model_checkpoints/pretrain/layers(3)_lr(0.001)_batch_size(8)_hidden(64)_bandgap_only{}epoch.pt".format(args.preepoch), map_location = device)
    # # pretrained_dict = torch.load("./model_checkpoints/pretrain/layers(3)_lr(0.001)_batch_size(64)_hidden(64)_efermi_after_masking{}epoch.pt".format(args.preepoch), map_location=device)
    # pretrained_dict = pretrained_dict["model_state_dict"]
    # model_dict = model.state_dict()

    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # matching = [s for s in list(pretrained_dict.keys()) if "GN_decoder" in s]
    
    # for i in range(len(matching)):
    #     del pretrained_dict[matching[i]]
    
    # assert len([s for s in list(pretrained_dict.keys()) if "GN_decoder" in s]) == 0

    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    # f = open("./experiments_{}.txt".format(embedder), "a")
    f = open("./experiments/experiments_practice_finetune.txt".format(embedder), "a")
    # f.write("finetune_on_pretraining_bandgap_only")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.MSELoss()
    spark_criterion = nn.BCELoss()

    train_loss = 0
    best_loss = 1000
    count = 0
    num_batch = int(n_train/args.batch_size)

    best_losses = list()

    for epoch in range(args.epochs):
        # if epoch ==1:
        #     break
        train_loss = 0
        corr_ = 0
        start = timer()
        model.train()
        cnt = 0
        for bc, batch in enumerate(train_loader):        
            batch.to(device)
            preds, _, spark = model(batch)    # dos, x, spark
            y = batch.y_ft.reshape(len(batch.mp_id), -1)
            mse = ((y - preds)**2).mean(dim = 1)
            rmse = torch.sqrt(mse).mean()

            # gap = batch.gap1

            # sampled = torch.randint((gap == 0).nonzero().reshape(-1).size()[0], (1, (gap == 1).nonzero().reshape(-1).size()[0])).reshape(-1)
            # sampled = (gap == 0).nonzero().reshape(-1)[sampled]
            # sampled = torch.cat([sampled, (gap == 1).nonzero().reshape(-1)])
            # # if len(sampled)==0:
            # #     continue
            # spark = spark.reshape(-1)[sampled]
            # spark_loss = spark_criterion(spark, gap[sampled])
            # loss = spark_loss + rmse
            loss = rmse
            # if len(sampled) ==0:
            #     loss = rmse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if bc == 2000 :
            #     print("here")
            train_loss += loss
            # print(f'train_loss: {train_loss}/{bc}')

            corr = correlation(y, preds)
            corr_ += corr

            # sys.stdout.write('\r[ epoch {}/{} | batch {}/{} ] correlation : {:.4f} '.format(epoch + 1, args.epochs, bc + 1, num_batch + 1, corr))
            # print('\r[ epoch {}/{} | batch {}/{} ] train_loss : {:.4f} | correlation : {:.4f} '.format(epoch + 1, args.epochs, bc + 1, num_batch + 1, rmse, corr))
            sys.stdout.write('\r[ epoch {}/{} | batch {}/{} ] train_loss : {:.4f} | correlation : {:.4f} '.format(epoch + 1, args.epochs, bc + 1, num_batch + 1, rmse, corr))
            sys.stdout.flush()
        # print(f'empty_sample_cnt:{cnt}/{3889}')
        writer.add_scalar("accs/train correlation", corr_ / (bc + 1), epoch + 1)
        writer.add_scalar("accs/train MSE", train_loss / (bc + 1), epoch + 1)

        if (epoch + 1) % args.eval == 0 :
            
            time = (timer() - start)/args.eval
            print("\ntraining time per epoch : {:.4f} sec".format(time))
            
            test_loss, test_corr, preds_y = test(model, test_loader, criterion, device)
            print("\n[ {} epochs ] test_loss : {:.4f} | test correlation : {:.4f}".format(epoch + 1, test_loss, test_corr))
            
            if test_loss < best_loss:
                best_loss = test_loss
                best_epoch = epoch + 1
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict()}
                torch.save(checkpoint, check_dir)
                torch.save(preds_y, pred_dir)
            
            writer.add_scalar("accs/MSE", test_loss, epoch + 1)
            writer.add_scalar("accs/correlation", test_corr, epoch + 1)
            writer.add_scalar("accs/best MSE", best_loss, epoch + 1)
            writer.add_scalar("time/training time", time, epoch + 1)

            best_losses.append(best_loss)
            st_best = '** [Best epoch: {}] Best test: {:.4f} **\n'.format(best_epoch, best_loss)
            print(st_best)

            if len(best_losses) > int(args.es / args.eval):
                if best_losses[-1] == best_losses[-int(args.es / 5)]:
                    
                    print("Early stop!!")
                    print("[Final] {}".format(st_best))
                    
                    f.write("\n")
                    f.write("Early stop!!\n")
                    # f.wite("W/O Pretrain and Multi-task")
                    f.write(configuration)
                    f.write("\nbest epoch : {} \n".format(best_epoch))
                    f.write("best MSE : {} \n".format(best_loss))
                    
                    sys.exit()
        
    print("\ntraining done!")
    print("[Final] {}".format(st_best))

    # write experimental results
    f.write("\n")
    f.write(configuration)
    f.write("\nbest epoch : {} \n".format(best_epoch))
    f.write("best MSE : {} \n".format(best_loss))
    f.close()


if __name__ == "__main__" :
    main()
