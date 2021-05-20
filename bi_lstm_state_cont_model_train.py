import numpy as np
import pandas as pd
import datetime
import os
import tensorflow as tf
import hn.utile as hnutile
import plotly.io as pio
import plotly.graph_objects as go
# from keras_self_attention import SeqSelfAttention
# from keras_self_attention import SeqSelfAttention

# 루트 디렉토리
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# 모델 저장 루트 디렉토리명
MODEL_ROOT_DIR_NAME = 'model'
# 모델 저장 서브 디렉토리명
MODEL_SUB_DIR_NAME = 'bi-lstm-state-cont'
# 모델파일 접두사
MODEL_FILE_PREFIX = 'bi-lstm-state-cont-model-'
# 학습데이터 루트 디렉토리명
TRAIN_ROOT_DIR_NAME = 'data'
# 학습데이터 서브 디렉토리명
TRAIN_SUB_DIR_NAME = 'cont-roll'
# 학습데이터 파일 접두사
TRAIN_FILR_PREFIX = 'train-01-'
# 학습데이터 경로
CSV_FILE_PATH = os.path.join(ROOT_DIR, TRAIN_ROOT_DIR_NAME, TRAIN_SUB_DIR_NAME)

# 학습, 예측 데이터 루프백 사이즈
LOOK_BACK = 50

FIT_PATIENCE = 80
LEARNING_RATE = 0.00001
def run_trainning(model, fits, x_trains, y_trains, x_vals, y_vals):
    '''
    학습 자동화
     모델 학습 및 저장, 조기학습종료, 베스트 모델선정 등
    '''

    custom_hist = hnutile.CustomEarlyStopCallback()

    for i in range(len(x_trains)):
        custom_hist.init(os.path.join(ROOT_DIR, MODEL_ROOT_DIR_NAME, MODEL_SUB_DIR_NAME), i, FIT_PATIENCE, MODEL_FILE_PREFIX)
        # 모델 저장 폴더 생성
        hnutile.MakeDir(os.path.join(ROOT_DIR, MODEL_ROOT_DIR_NAME, MODEL_SUB_DIR_NAME,'%03d'%(i)))
    
        print('File %03dth: Start trainning.'%(i))
        for j in range(fits):
            custom_hist.set_fit_count(j)
            print('Fit %d/%d.'%(j+1, fits))
            hist = model.fit(x_trains[i], y_trains[i], epochs=1, batch_size=1, shuffle=False, callbacks=[custom_hist], validation_data=(x_vals[i], y_vals[i]))
            model.reset_states()
            # 조기 종료 체크
            if custom_hist.early_stop():
                break

        # Bset model 선정
        base_model_name = hnutile.FindBestModel(os.path.join(ROOT_DIR, MODEL_ROOT_DIR_NAME, MODEL_SUB_DIR_NAME, '%03d'%(i)), MODEL_FILE_PREFIX)
        if base_model_name != None:
            print('File %03dth: The best model to be use is model >> %s.'%(i ,base_model_name))
            model = tf.keras.models.load_model(os.path.join(ROOT_DIR, MODEL_ROOT_DIR_NAME, MODEL_SUB_DIR_NAME, '%03d'%(i), base_model_name), compile=True)
            # model.compile(loss=root_mean_squared_error, optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
        else:
            print('File %03dth: Best model does not exist.'%(i))

def root_mean_squared_error(y_true, y_pred):
    return tf.math.sqrt(tf.keras.losses.mean_squared_error(y_true, y_pred))

def create_bi_state_lstm_model(look_back=1):
    '''
    양방향 상태 전이 LSTM모델 생성
    '''
    model = tf.keras.Sequential()

    for i in range(3):
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, batch_input_shape=(1, look_back, 1), stateful=True, return_sequences=True)))
        model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, stateful=True, return_sequences=False)))
    model.add(tf.keras.layers.Dropout(0.3))
    # model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    # model.add(tf.keras.layers.Dense(1, activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
    model.add(tf.keras.layers.Dense(1))

    # model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001))
    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
    # model.compile(loss=root_mean_squared_error, optimizer=tf.keras.optimizers.Adam())
    return model

def main():
    '''
    메인 함수
    '''

    df = pd.read_csv(os.path.join(CSV_FILE_PATH ,'op10-3-train-01.csv'))

    print(df.info())
    # 그래프 출력
    hnutile.ShowContinueGraph(df, yaxis='RollLoad')
    
    load_datas = df['RollLoad'].values[:, None]

    # 모델 생성
    model = create_bi_state_lstm_model(LOOK_BACK)

    
    # x_train, y_train, x_val, y_val = hnutile.CreateContinuTrainDatas(data=load_datas[:10000], look_back=LOOK_BACK)

    data_fit_size= 12000
    data_fit_count = int(len(load_datas) / data_fit_size)
    x_trains = []
    y_trains = []
    x_vals = []
    y_vals = []
    for i in range(data_fit_count):
        x_train, y_train, x_val, y_val = hnutile.CreateContinuTrainDatas(data=load_datas[i*data_fit_size:(i+1)*data_fit_size], look_back=LOOK_BACK)
        x_trains.append(x_train)
        y_trains.append(y_train)
        x_vals.append(x_val)
        y_vals.append(y_val)

    run_trainning(model, 200, x_trains, y_trains, x_vals, y_vals)

    print('Exit process.')


if __name__ == '__main__':
    main()
    