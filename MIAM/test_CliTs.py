
from models import Multi_Duration_Pipeline_Residual
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import os
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
import pandas as pd
from torch.optim import RAdam
from torch.nn import MSELoss
from helper import *



class CliTsDataset(Dataset):
    def __init__(self, df, feature_cols, label_col, time_col, test=False):
        super(CliTsDataset, self).__init__()
        
        self.df = df
        ids = self.df['id'].unique().tolist()
        X_ = []
        y_ = []
        times_ = []
        for i in range(len(ids)):
            pid = ids[i]
            
            time = self.df[df['id']==pid][time_col].to_numpy()
            x = self.df[df['id']==pid][feature_cols]
            time = time[(~x.isna().all(axis=1)).tolist()]
            x = x.dropna(how='all').to_numpy()
                   
            if len(x)==0:
                continue
            
            time = np.expand_dims(time, 1)
            
            y = np.nan_to_num(self.df[df['id']==pid][label_col].to_numpy())
            y = 1 if np.max(y) != 0 else 0
            

            X_.append(x)
            times_.append(time)
            y_.append(y)
                    
        if not test:
            self.X = X_
            self.y = y_
            self.times = times_
        else:
            num_vte = int(np.sum(y_))
            X_v = [] # vte data
            X_nv = [] # non vte data
            times_v = []
            times_nv = []
            for i in range(len(y_)):
                if y_[i]:
                    X_v.append(X_[i])
                    times_v.append(times_[i])
                else:
                    X_nv.append(X_[i])
                    times_nv.append(times_[i])     
            
            zipped = list(zip(X_nv, times_nv))
            np.random.shuffle(zipped)
            X_nv, times_nv = zip(*zipped)
            X_nv = list(X_nv[:num_vte])
            times_nv = list(times_nv[:num_vte])
                        
            self.X = X_v+X_nv
            self.y = np.hstack([np.ones(num_vte), np.zeros(len(X_nv))])
            self.times = times_v+times_nv
                    
            
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.times[idx]



### ==============================================================================================

cancer_types = [
    'Breast', 
    'Cervical', 
    'Colon', 
    'Esophageal', 
    'Gastric', 
    'Liver', 
    'Lung', 
    'Pancreatic', 
    'Rectal'
]

type_dict = dict(zip(cancer_types, range(1, len(cancer_types)+1)))


feature_cols = ["APTT", "PT", "TT", "FIB", "D-dimer", "FDP", 
                "NE", "LYM", "MONO", "WBC", "PLT", "Hb", 
                "ALB", "GLB", "GGT", 
                "Beta2", "CR", 
                "CD4", "CD8", "CD4/CD8", 
                "LDH",
                "age", "height", "weight", "BMI",
                "sex", "bloodType_A", "bloodType_B", "RH", 
                "Radiotherapy", "Chemotherapy", "Surgery", "Targeted", "HPT", "DM"]

feature_cols += cancer_types

norm_cols = ["APTT", "PT", "TT", "FIB", "D-dimer", "FDP", 
                "NE", "LYM", "MONO", "WBC", "PLT", "Hb", 
                "ALB", "GLB", "GGT", 
                "Beta2", "CR", 
                "CD4", "CD8", "CD4/CD8", 
                "LDH",
                "age", "height", "weight", "BMI"]


def read_data_file(path, type):
    df = pd.read_csv(path)
    for t in cancer_types:
        if t == type:
            df[t] = [1]*len(df)
        else:
            df[t] = [0]*len(df)
            
    return df

def load_data(path, data_types=[]):
    # load data
    data = load_data_from_dir(path)
    dfs = []
    for data_type in data_types:
        if data_type in data:
            dfs.append(data[data_type])
        else:
            print('data type %s not found in data folder, skip' % data_type)
    data = pd.concat(dfs, axis=0) if len(dfs) > 0 else pd.DataFrame([])
    return data


def load_data_from_dir(path):
    data = {}
    for file in os.listdir(path):
        if file.endswith(".csv"):
            try:
                key = file.split("_")[1]
            except:
                key = file.split('.')[0]
            if not key in cancer_types:
                key = 'other'
            df = read_data_file(os.path.join(path, file), key) 
            if key in cancer_types:   
                df['id'] = reform_id(df['id'].to_numpy(), key)      
            data[key] = df
    return data    


def split_data(data, train_size=0.8, val_size=0, mode='seq'):
    if mode == 'seq':
        pids = data['id'].unique()
        np.random.shuffle(pids)
        
        train_idx = int(len(pids)*train_size)
        val_idx = int(len(pids)*(train_size+val_size))
        
        train_ids = pids[:train_idx]
        val_ids = pids[train_idx:val_idx]
        
        train_df = data[data['id'].isin(train_ids)]
        val_df = data[data['id'].isin(val_ids)]
        test_df = data[~data['id'].isin(np.concatenate([train_ids, val_ids]))]
    
    elif mode == 'tp':
        train_df = data.sample(frac=train_size)
        test_df = data.drop(train_df.index)
        val_df = None
    
    return train_df, val_df, test_df


def norm_df_ignorena(df: pd.DataFrame, cols, means=None, stds=None):
    
    convert_dict = dict(zip(cols, [float]*len(cols)))
    df = df.astype(convert_dict)
    
    if means is None or stds is None:
        means = np.zeros(len(cols))
        stds = np.zeros(len(cols))
        for i, col in enumerate(cols):
            mean = df.loc[pd.notnull(df.loc[:, col]), col].mean()
            std = df.loc[pd.notnull(df.loc[:, col]), col].std()
            df.loc[pd.notnull(df.loc[:, col]), col] = (df.loc[pd.notnull(df.loc[:, col]), col] - mean) / std
            means[i] = mean
            stds[i] = std
    else:
        assert len(cols) == len(means), '[Error] given num means doesnot match num features'
        assert len(cols) == len(stds), '[Error] given num stds doesnot match num features'

        for i, col in enumerate(cols):
            df.loc[pd.notnull(df.loc[:, col]), col] = (df.loc[pd.notnull(df.loc[:, col]), col] - means[i]) / stds[i]

    return df, means, stds



def collate_fn(batch, fillna=0, max_len=None, device='cuda'):
    y = torch.from_numpy(np.array([item[1] for item in batch]).astype(np.float32))
    
    if max_len is None:
        lengths = [len(item[0]) for item in batch]
        max_len = np.max(lengths)
    else:
        lengths = [min(len(item[0]), max_len) for item in batch]
    
    X = []
    times = []
    deltas = []
    for item in batch:
        x = np.array(item[0])
        time = np.array(item[2])
        if len(x) < max_len:   
            pad = np.zeros((max_len-x.shape[0], x.shape[1]))
            time_pad = np.zeros((max_len-time.shape[0], time.shape[1]))
            x = np.vstack([x, pad])
            time = np.vstack([time, time_pad])
        else:
            x = x[:max_len]
            time = time[:max_len]
        
        t = np.zeros_like(x)  
        for l in range(x.shape[0]):
            t[l] = np.where(~np.isnan(x[l]), time[l], t[l])
        
        X.append(x)
        times.append(time)
        deltas.append(t)
    
    times = torch.squeeze(torch.from_numpy(np.array(times).astype(np.float32)))
    deltas = torch.squeeze(torch.from_numpy(np.array(deltas).astype(np.float32)))
    X = torch.from_numpy(np.array(X).astype(np.float32))    
    mask_nan = torch.where(torch.isnan(X), True, False)
    
    X = torch.masked_fill(X, mask_nan, fillna)
    mask_nan = torch.where(mask_nan, 0., 1.)
    
    batch = {
        "values" : X,
        "masks" : mask_nan,
        "times" : times,
        "deltas" : deltas,
        "labels" : y
    }
    
    return batch



def build_dataloader(train_data, val_data, test_data, feature_cols, norm_cols, label_col='VTE', time_col='date', device='cuda:0', max_len=200, train_batch_size=256, test_batch_size=256):
    
    fillna = 0
    # normalize        
    train_data, means, stds = norm_df_ignorena(train_data, norm_cols)
    val_data, means, stds = norm_df_ignorena(val_data, norm_cols, means, stds)
    test_data, means, stds = norm_df_ignorena(test_data, norm_cols, means, stds)
    
    train_dataset = CliTsDataset(train_data, feature_cols, label_col, time_col)
    val_dataset = CliTsDataset(val_data, feature_cols, label_col, time_col, test=True)
    test_dataset = CliTsDataset(test_data, feature_cols, label_col, time_col, test=True)

    train_vte = np.sum(train_dataset.y)
    train_weights = np.where(np.array(train_dataset.y) == 1, (len(train_dataset.y)-train_vte)/train_vte, 1)
    train_sampler = WeightedRandomSampler(train_weights, len(train_dataset.y), replacement=True)

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=lambda x: collate_fn(x, fillna=fillna, max_len=max_len, device=device), sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=test_batch_size, collate_fn=lambda x: collate_fn(x, fillna=fillna, max_len=max_len, device=device), shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, collate_fn=lambda x: collate_fn(x, fillna=fillna, max_len=max_len, device=device), shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader

def reform_id(ids, key):
    ids = ids.astype(int)
    if ids[0] > 1e10:
        return ids
    return ids + type_dict[key] * 1e10

def split_build_dataloader(data, feature_cols, device):
    train_data, val_data, test_data = split_data(data, train_size=0.7, val_size=0.15)
    return build_dataloader(train_data, val_data, test_data, feature_cols, norm_cols, device=device)


def init_resdict():
    return {
        'auc': [],
        'f1': [],
        'rec': [],
        'prec': [],
        'acc': [],
        'loss': []
    }



def test_model(model, loss1, loss2, beta, delta, dataloader):
    model.eval()
    
    losses = []
    preds = []
    targets = []
    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():
            x = batch['values'].to(device)  # Batch x Time x Variable
            m = batch['masks'].to(device)  # Batch x Time x Variable
            deltas = batch['deltas'].to(device)  # Batch x Time x Variable
            times = batch['times'].to(device)  # Batch x Time x Variable
            y = batch['labels'].to(device)

            attn_mask = deltas.data.eq(0)[:, :, 0]
            attn_mask[:, 0] = 0

            # Zero Grad
            optimizer.zero_grad()

            # model
            output, out = model(x, m, times, deltas, attn_mask)

            # Calculate and store the loss
            loss_a = loss1(model, output, y)
            loss_b = loss2(out, x)
            loss = beta*loss_a + delta*loss_b

            preds.append(output.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
            losses.append(loss.item())
            
    loss = np.mean(losses)
            
    pred = np.hstack(preds)
    targets = np.hstack(targets)

    auc = roc_auc_score(targets, pred)
    pred = np.where(pred<0.5, 0, 1)
    f1, rec, prec, acc = f1_score(targets, pred), recall_score(targets, pred), precision_score(targets, pred), accuracy_score(targets, pred)
    
    return auc, f1, rec, prec, acc, loss


# 

### ==========================================================================================================



device = 'cuda:0'
input_dim = len(feature_cols)
d_model = 32
d_ff = 64
num_stacks = 2
num_heads = 4
max_length = 200

l1 = 5e-4
lr = 5e-4
w_decay = 1e-3
alpha = 9
gamma = 0.1
beta = 0.1
delta = 11

criterion_focal = FocalLoss(l1, device, gamma=gamma, alpha=alpha, logits=False).to(device)
criterion_mse = nn.MSELoss()

data = load_data('/home/projects/CliTsRNN/data/data', cancer_types)

res = init_resdict()
val_res = init_resdict()

for i in range(10):
    print('step %d' % i, '='*50)
    
    train_dataloader, val_dataloader, test_dataloader = split_build_dataloader(data, feature_cols, device)
    model = Multi_Duration_Pipeline_Residual(input_dim, d_model, d_ff, num_stacks, num_heads, max_length, n_iter=1).to(device)
    optimizer = RAdam(list(model.parameters()), lr=lr, weight_decay=w_decay)
    
    best_score = 0
    best_epoch = 0
    for epoch in range(50):
        model.train()
        
        train_loss = []
        for batch_idx, batch in enumerate(train_dataloader):
            x = batch['values'].to(device)  # Batch x Time x Variable
            m = batch['masks'].to(device)  # Batch x Time x Variable
            deltas = batch['deltas'].to(device)  # Batch x Time x Variable
            times = batch['times'].to(device)  # Batch x Time x Variable
            y = batch['labels'].to(device)

            attn_mask = deltas.data.eq(0)[:, :, 0]
            attn_mask[:, 0] = 0

            # Zero Grad
            optimizer.zero_grad()

            # model
            output, out = model(x, m, times, deltas, attn_mask)

            # Calculate and store the loss
            loss_a = criterion_focal(model, output, y)
            loss_b = criterion_mse(out, x)
            loss = beta*loss_a + delta*loss_b

            train_loss.append(loss.item())

            # Backward Propagation
            loss.backward()

            # Update the weights
            optimizer.step()
            
        train_loss = np.mean(train_loss)
        
        auc, f1, rec, prec, acc, val_loss = test_model(model, criterion_focal, criterion_mse, beta, delta, val_dataloader)
        score = np.mean([auc, f1, acc])
        
        print('[Epoch %d] train loss: %.4f || val loss: %.4f' % (epoch, train_loss, val_loss))
        print('auc: %.4f | f1: %.4f | recall: %.4f | precision: %.4f | accuracy: %.4f' % (auc, f1, rec, prec, acc))
        
        if score > best_score:
                torch.save(model.state_dict(), './ckpt/best.pt')
                best_score = score
                best_epoch = epoch
                print('update best score %.4f, model saved' % best_score)
        
    print('training finished', '='*50)
    print('best score %.4f at epoch %d' % (best_score, best_epoch))    
    
    model.load_state_dict(torch.load('./ckpt/best.pt'))
    
    auc, f1, rec, prec, acc, loss = test_model(model, criterion_focal, criterion_mse, beta, delta, test_dataloader)
    res['auc'].append(float(auc))
    res['f1'].append(float(f1))
    res['rec'].append(float(rec))
    res['prec'].append(float(prec))
    res['acc'].append(float(acc))
    res['loss'].append(float(loss))
    
    auc, f1, rec, prec, acc, loss = test_model(model, criterion_focal, criterion_mse, beta, delta, val_dataloader)
    val_res['auc'].append(float(auc))
    val_res['f1'].append(float(f1))
    val_res['rec'].append(float(rec))
    val_res['prec'].append(float(prec))
    val_res['acc'].append(float(acc))
    val_res['loss'].append(float(loss))
    
    print('MIAM: ')
    print('val auc: %.4f (%.4f)' % (np.mean(val_res['auc']), 1.96/np.sqrt(100) * np.std(val_res['auc'])))
    print('val f1 score: %.4f (%.4f)' % (np.mean(val_res['f1']), 1.96/np.sqrt(100) * np.std(val_res['f1'])))
    print('val recall: %.4f (%.4f)' % (np.mean(val_res['rec']), 1.96/np.sqrt(100) * np.std(val_res['rec'])))
    print('val precision: %.4f (%.4f)' % (np.mean(val_res['prec']), 1.96/np.sqrt(100) * np.std(val_res['prec'])))
    print('val accuracy: %.4f (%.4f)' % (np.mean(val_res['acc']), 1.96/np.sqrt(100) * np.std(val_res['acc'])))
    print('='*10)
    print('auc: %.4f (%.4f)' % (np.mean(res['auc']), 1.96/np.sqrt(100) * np.std(res['auc'])))
    print('f1 score: %.4f (%.4f)' % (np.mean(res['f1']), 1.96/np.sqrt(100) * np.std(res['f1'])))
    print('recall: %.4f (%.4f)' % (np.mean(res['rec']), 1.96/np.sqrt(100) * np.std(res['rec'])))
    print('precision: %.4f (%.4f)' % (np.mean(res['prec']), 1.96/np.sqrt(100) * np.std(res['prec'])))
    print('accuracy: %.4f (%.4f)' % (np.mean(res['acc']), 1.96/np.sqrt(100) * np.std(res['acc'])))
    print('='*30)
        

    