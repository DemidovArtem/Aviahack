import torch
from torch import nn
from math import sin, cos, atan, pi, radians
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from sklearn.externals import joblib 
from time import time
from matplotlib import pyplot as plt

def get_seq_len(data):
    seq_len = 0
    for id, data_id in data.groupby('id'):
        seq_len = max(seq_len, data_id.shape[0])
    return seq_len


class net(nn.Module):
    def __init__(self, input_size=7, hidden_size=10):
        super(net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size
        )
        self.out = nn.Sequential(
            nn.Linear(
                in_features=hidden_size * 3,
                out_features=hidden_size
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=hidden_size,
                out_features=2
            )
        )

    def forward(self, batch, sizes):
        print('batch', batch)
        output, (h_n, c_n) = self.lstm(batch)
        print(h_n)
        h = output[sizes - 1, range(len(sizes)), :]
        h_max = torch.empty_like(h)
        h_mean = torch.empty_like(h)
        for i in range(len(sizes)):
            elem = output[:sizes[i] - 1, i, :]
            try:
                h_max[i, :] = torch.max(elem, dim=0)[0]
                h_mean[i, :] += torch.mean(elem, dim=0)[0]
            except:
                print('size', sizes[i])
                print('elem', elem)
    
        h_max = h_max.squeeze()
        h_mean = h_mean.squeeze()
        h = h.squeeze()
        h_new = torch.cat([h, h_max, h_mean], dim=1)
        res = self.out(h_new)
        
        return res
    

def clean(data, cols):
    for col in cols:
        data.loc[data['time_diff'] == 0, col] = None
        data.loc[data[col].isnull(), col] = data[col].mean(axis=0)
        yield data[col]    
        
        
def data_converter(batch_id, data, device='cuda', labels=None, input_size=7, verbose=False):
    batch_size = len(batch_id)
    y = np.zeros(shape=batch_size)
    if isinstance(data, dict):
        seq_len = max(df.shape[0] for df in data.values())
    else:
        seq_len = get_seq_len(data)
    batch = np.empty(shape=(seq_len, batch_size, input_size))
    sizes = []
    curr_tqdm = lambda x: x
    if verbose:
        curr_tqdm = tqdm
    for i, id in enumerate(curr_tqdm(batch_id)):
        if labels is not None:
            y[i] = labels[id]
        if not isinstance(data, dict):
            val = data[data['id'] == id].drop(columns='id').values
        else:
            val = data[id]
        curr_size = val.shape[0]
        sizes.append(curr_size)
        batch[:curr_size, i, :] = val

    y = torch.from_numpy(y).to(device).long()
    X = torch.from_numpy(batch).to(device).float()
    return X, y, np.array(sizes)
    

def hist_maker(data, alpha=0.05):
    cols = [col for col in data.columns if col != 'id']
    bounds = dict()
    for col in cols:
        beta = alpha
        bounds[col] = {
            'lower': data[col].quantile(beta/2),
            'upper': data[col].quantile(1 - beta/2)
        }
    
    for array in data[cols].hist(figsize=(20, 15), bins=20):
        for i, subplot in enumerate(array):
            subplot.set_xlim((bounds[cols[i]]['lower'], bounds[cols[i]]['upper']))    
    plt.savefig('./Data/hists')
        
        
def predict(path, device='cuda'):
    start = time()
    res = pd.read_csv('Data/' + path, sep=' ', encoding='cp1250', header=None)
    res.columns = ['time', 'id', 'lat', 'lon', 'height', 'code', 'name']
    res['id'] = path + '_' + res['id'].astype(str)
    res['time'] = pd.to_timedelta(res['time'])
    res['time'] = res['time'].dt.total_seconds()
    data = res.copy()
    data = data.sort_values(by=['id', 'time']).reset_index(drop=True)
#     генерация признаков
    lon2 = data['lon'].apply(radians)
    lon1 = data.groupby('id')['lon'].shift(1).apply(radians)
    dlon = (lon2 - lon1)

    lat2 = data['lat'].apply(radians)
    lat1 = data.groupby('id')['lat'].shift(1).apply(radians)

    dlat = lat2 - lat1

    y = dlon.apply(sin) * lat2.apply(cos)
    x = lat1.apply(cos) * lat2.apply(sin) \
    - lat1.apply(sin) * lat2.apply(cos) * dlon.apply(cos)

    data['angle'] = (y /x).apply(atan)
    data['angle'] = (data['angle'] + pi) % pi


    a = (dlat / 2).apply(sin)**2 + lat1.apply(cos) * lat2.apply(cos) * (dlon / 2).apply(sin)**2
    data['dist'] = 2 * (a**(1/2) / (1 - a)**(1/2)).apply(atan)    

    shift_cols = ['height', 'time']
    for col in shift_cols:
        data[col + '_diff'] = data[col] - data.groupby('id')[col].shift(1)
    data = data.drop(columns=['time', 'lat', 'lon', 'code', 'name']).dropna()

    data['speed'] = data['dist'] / data['time_diff']
    data['accel'] = (data['speed'] - data['speed'].shift(1)) / data['time_diff']
    cols = ['accel', 'speed']
    for i, val in enumerate(clean(data, cols)):
        data[cols[i]] = val
#     удаление inf в ускорении
    min_value = data['accel'].min()
    max_value = data['accel'].max()
    data.loc[data['accel'] == min_value, 'accel'] = data.loc[data['accel'] != min_value, 'accel'].min()
    data.loc[data['accel'] == max_value, 'accel'] = data.loc[data['accel'] != max_value, 'accel'].max()
        
#     нормировка данных
    x = data.drop(columns=['id']).values
    min_max_scaler = joblib.load('./Models/scaler.pkl')
    x_scaled = min_max_scaler.transform(x)
    cols = [col for col in data.columns if col != 'id']
    data = data.reset_index(drop=True)
    data.loc[:, cols] = pd.DataFrame(x_scaled, columns=cols)
    data_dict = {id: data[data['id'] == id].drop(columns='id').values for id in data['id'].unique()}
    #     графики
    hist_maker(data)
    end = time()
    preprocess_time = int(end - start)
    
#     преобразование к нужному виду
    start = time()
    X, _, sizes = data_converter(data['id'].unique(), data_dict, device)
    end = time()
    convertion_time = int(end - start)
#     выгрузка и предсказание модели
    pre_trained_model = net()
    pre_trained_model.load_state_dict(torch.load('./Models/classifier'))
    pre_trained_model.to(device)
    pre_trained_model.eval()
    
    start = time()
    pred = pre_trained_model.forward(X, sizes)
    end = time()
    prediction_time = int(end - start)
    labels = pred.cpu().detach().numpy().argmax(axis=1)
#     запись результатов
    labels = {
        id: val for (id, val) in zip(data['id'].unique(), labels)
    }
    bad_ids = [id for id in data['id'].unique() if labels[id]]
    good_ids = [id for id in data['id'].unique() if not labels[id]]
    res[res['id'].isin(good_ids)].to_csv('./Data/GoodTracks' + path, index=False)
    res[res['id'].isin(bad_ids)].to_csv('./Data/BadTracks' + path, index=False) 
    return preprocess_time, convertion_time, prediction_time