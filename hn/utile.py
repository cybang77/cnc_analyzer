
import numpy as np
from tensorflow.python.client import device_lib
import plotly.graph_objects as go
import plotly.io as pio
import tensorflow as tf
import os
import re

def GPU_INFO():
    print(device_lib.list_local_devices())


def create_dataset(signal_data, look_back=1):
    '''
    룩백사이즈로 데이터 생성
     dataX 루백사이즈 데이터 array
     dataY 루백+1번째 데이터 array(1건)
    '''
    dataX, dataY = [], []
    for i in range(len(signal_data)-look_back):
        dataX.append(signal_data[i:(i+look_back), 0])
        dataY.append(signal_data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def CreateContinuTrainDatas(data, look_back=1, train_ratio=0.9):
    train_size = int(len(data) * train_ratio)
 
    x_train, y_train = create_dataset(data[:train_size])
    x_val, y_val = create_dataset(data[train_size:])

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))

    return x_train, y_train, x_val, y_val

def CreateTrainDatas(datas, look_back=1):
    '''
    '''
    x_trains = []
    y_trains = []
    x_vals = []
    y_vals = []

    for i in range(90):
        x_train, y_train = create_dataset(datas[i], look_back)
        x_trains.append(x_train)
        y_trains.append(y_train)
    
    for i in range(10):
        x_val, y_val = create_dataset(datas[i+90], look_back)
        x_vals.append(x_val)
        y_vals.append(y_val)

    # 입력형 변환
    for i in range(len(x_trains)):
        x_trains[i] = np.reshape(x_trains[i], (x_trains[i].shape[0], x_trains[i].shape[1], 1))

    for i in range(len(x_vals)):
        x_vals[i] = np.reshape(x_vals[i], (x_vals[i].shape[0], x_vals[i].shape[1], 1))

    return x_trains, y_trains, x_vals, y_vals

def ShowContinueGraph(data, yaxis='Load'):
    pio.templates.default = "plotly_dark"
    fig = go.Figure(layout=go.Layout(height=600, width=1200))
    fig = fig.add_trace(go.Scatter(y=data[yaxis], mode='lines', name = yaxis, line=dict(width=1)))

    fig.update_layout(title=yaxis,
                    xaxis_title='x',
                    yaxis_title='y')
    fig.show()

def ShowGraph(datas, yaxis='Load'):
    pio.templates.default = "plotly_dark"
   
    fig = go.Figure(layout=go.Layout(height=600, width=1200))
    cnt = 0
    for df in datas:
        fig = fig.add_trace(go.Scatter(y=df[yaxis], mode='lines', name = yaxis + str(cnt), line=dict(width=1)))
        cnt += 1

    fig.update_layout(title=yaxis,
                    xaxis_title='x',
                    yaxis_title='y')
    fig.show()


def FindBestModel(path_dir, file_pre_fix):
    regex = re.compile(r'^%s(\d{3})-(\d{3})-(\d.\d{6})-(\d.\d{6}).h5'%(file_pre_fix))
    file_list = os.listdir(path_dir)
    val_loss_list = []
    for i in range(len(file_list)):
        name = file_list[i]
        matchobj = regex.search(name)
        if matchobj != None:
            val_loss_list.append(float(matchobj.group(4)))

    # print(val_loss_list)
    if len(val_loss_list) > 0:
        return file_list[val_loss_list.index(min(val_loss_list))]
    return None

def MakeDir(path):
    '''
    '''
    if not os.path.exists(path):
        os.makedirs(path)



class CustomEarlyStopCallback(tf.keras.callbacks.Callback):
    '''
    '''
    def init(self, model_dir, file_count, patience=10, file_pre_fix=None, best_model_save=True):
        self.loss = []
        self.val_loss = []
        self.model_dir = model_dir
        self.file_count = file_count
        self.patience = patience
        self.wait = 0
        self.best_val = np.Inf
        self.file_pre_fix = file_pre_fix
        self.best_model_save = best_model_save

    def early_stop(self):
        if self.wait >= self.patience:
            print('Fit %05d: early stopping'%(self.file_count)) 
            return True
        return False

    def set_fit_count(self, fit_count):
        self.fit_count = fit_count

    def on_epoch_end(self, batch, logs={}):
   
        if len(self.val_loss) > 0:
            min_val_loss = min(self.val_loss)
            
            filename = os.path.join(
                    self.model_dir,
                    '%03d'%(self.file_count),
                    '%s%03d-%03d-%.6f-%.6f.h5'%(self.file_pre_fix, self.file_count, self.fit_count, logs.get('loss'), logs.get('val_loss'))
                    )
            if min_val_loss > logs.get('val_loss'): # 성능개선이 된 경우
                # save
                self.model.save(filename)
                self.best_val = logs.get('val_loss')
                self.wait = 0
                print('Fit %05d: val_loss improved from %.6f to %.6f, saving model to %s'%(self.fit_count, min_val_loss, logs.get('val_loss'), filename))
            else: # 성능개선이 없는경우
                self.wait +=1
                if self.best_model_save == False:
                    self.model.save(filename)
                    print('Fit %05d: val_loss did not improve from %.6f, wait %d/%d, saving model to %s'%(self.fit_count, min_val_loss, self.wait, self.patience, filename))
                else:
                    print('Fit %05d: val_loss did not improve from %.6f, wait %d/%d'%(self.fit_count, min_val_loss, self.wait, self.patience))
        else: # 학습 처음시작인경우
            filename = os.path.join(
                    self.model_dir,
                    '%03d'%(self.file_count),
                    '%s%03d-%03d-%.6f-%.6f.h5'%(self.file_pre_fix, self.file_count, self.fit_count, logs.get('loss'), logs.get('val_loss'))
                    )
            self.model.save(filename)
            self.best_val = logs.get('val_loss')
            self.wait = 0
            print('Fit %05d: val_loss improved from inf to %.6f, saving model to %s'%(self.fit_count, logs.get('val_loss'), filename))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))