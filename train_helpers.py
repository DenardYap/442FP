
import itertools
import os
import sys
import torch
from torch.nn.functional import softmax
from sklearn import metrics
import warnings
import numpy as np
import csv 
warnings.filterwarnings("ignore")


# parser = argparse.ArgumentParser()
# # Adding optional argument
# parser.add_argument("-b", "--batch_size")
# parser.add_argument("-lr", "--learning_rate")
# parser.add_argument("-p", "--momentum")
# parser.add_argument("-wd", "--weight_decay")
# parser.add_argument("-g", "--gpu")
# parser.add_argument("-s", "--save_folder")
# # parser.add_argument("-a", "--all_layers")

# args = parser.parse_args()


# batch_size = int(args.batch_size)
# learning_rate = float(args.learning_rate)
# momentum = float(args.momentum)
# weight_decay = float(args.weight_decay)
# gpu = str(args.gpu)
# save_folder = args.save_folder



# print("starting training")
# start_time = time.time()
# trial(batch_size, learning_rate, momentum, weight_decay, gpu)
# end_time = time.time()
# name = f"b{batch_size}_lr{learning_rate}_p{momentum}_wd{weight_decay}"
# time_dir = os.path.join(save_folder, name)
# time_dir = os.path.join(time_dir,"time.txt")
# with open(time_dir,"w") as file:
#     file.write(str(end_time - start_time))

# print("done!")

def log_aurocs(epoch,aurocs,info,log_dir,train_val):
    name = "b{b}_lr{lr}_p{p}_wd{wd}".format(b = info["batch"],
                                                lr = info["lr"],
                                                p = info["p"],
                                                wd = info["wd"])
    model_path = os.path.join(log_dir,name)
    model_path = os.path.join(model_path,f"{train_val}_aurocs.csv")

    if epoch == 0:
        if os.path.exists(model_path):
            os.remove(model_path)
        with open(model_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Epoch','ABOUT','BECAUSE','CALLED', "DAVID", "EASTERN"])
        
    with open(model_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([epoch, *aurocs])

def log_acc(epoch,accs,info,log_dir,train_val):
    name = "b{b}_lr{lr}_p{p}_wd{wd}".format(b = info["batch"],
                                                lr = info["lr"],
                                                p = info["p"],
                                                wd = info["wd"])
    model_path = os.path.join(log_dir,name)
    model_path = os.path.join(model_path,f"{train_val}_acuracy.csv")

    if epoch == 0:
        if os.path.exists(model_path):
            os.remove(model_path)
        with open(model_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Epoch','ABOUT','BECAUSE','CALLED', "DAVID", "EASTERN"])
        
    with open(model_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([epoch, accs])


def log_training(epoch, stats_at_epoch, info, log_dir):
    name = "b{b}_lr{lr}_p{p}_wd{wd}".format(b = info["batch"],
                                                lr = info["lr"],
                                                p = info["p"],
                                                wd = info["wd"])
    model_path = os.path.join(log_dir,name)
    model_path = os.path.join(model_path,"loss_log.csv")

    if epoch == 0:
        if os.path.exists(model_path):
            os.remove(model_path)
        with open(model_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Epoch","Train Loss","Val Loss","Test Loss"])
        
    with open(model_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([epoch, *stats_at_epoch])
        import numpy as np

def early_stopping(stats, curr_count_to_patience, global_min_loss):
    if stats[-1][-2] >= global_min_loss:
        curr_count_to_patience += 1
    else:
        global_min_loss = stats[-1][-2]
        curr_count_to_patience = 0
    return curr_count_to_patience, global_min_loss


def evaluate_epoch(tr_loader,val_loader,te_loader,model,criterion,epoch,
                   stats,device,info,save_folder,reg,include_test=True,
                   update_plot=True,multiclass=False):
    def _get_metrics(loader,is_train):
        y_true, y_score = [], []
        correct, total = 0, 0
        running_loss = []
        model.eval()
        max_iterations = 50
        iteration = 0
        for X, y in loader:
            if is_train and iteration > max_iterations:
                break
            with torch.no_grad():
                X = X.to(device)
                y = y.to(device)
                output = model(X)
                # y = y.float()
                y_true.append(y)
                y_score.append(output.data)
                running_loss.append(criterion(output, y).cpu())
            iteration += 1
        y_true = torch.cat(y_true)
        y_score = torch.cat(y_score)
        loss = np.mean(running_loss)
        aurocs = metrics.roc_auc_score(y_true.cpu(), y_score.cpu(),average=None)
        accs = metrics.accuracy_score( np.argmax(y_true.cpu(), axis=1),np.argmax(y_score.cpu(), axis=1))
        return np.round(aurocs, decimals=3), np.round(accs, decimals=3), round(loss,3)

    train_aurocs, train_accs, train_loss = _get_metrics(tr_loader,True)
    val_aurocs, val_accs, val_loss  = _get_metrics(val_loader,False)
    te_aurocs, test_accs, te_loss  = _get_metrics(te_loader,False)

    print(f'Epoch:{epoch}')
    print(f'train loss {train_loss}')
    print(f'val loss {val_loss}')
    print(f'test loss {te_loss}\n')

    log_aurocs(epoch,train_aurocs,info,save_folder,'train')
    log_aurocs(epoch,val_aurocs,info,save_folder,'val')
    log_aurocs(epoch,te_aurocs,info,save_folder,'test')
    log_acc(epoch,train_accs,info,save_folder,'train')
    log_acc(epoch,val_accs,info,save_folder,'val')
    log_acc(epoch,test_accs,info,save_folder,'test')

    stats_at_epoch = [train_loss, val_loss, te_loss]
    stats.append(stats_at_epoch)
    
    log_training(epoch, stats_at_epoch, info, save_folder)
    

def train_epoch(data_loader, model, criterion, optimizer, device):
    model.train()              
    for i, (X, y) in enumerate(data_loader):
        X,y = X.to(device), y.to(device) # todo: change it up 
        y = y.float() # todo: change it up 
        optimizer.zero_grad()
        y_pred = model(X) 
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

