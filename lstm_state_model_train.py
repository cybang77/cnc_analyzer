import numpy as np
import pandas as pd
import datetime
import os
import tensorflow as tf
import hn.utile as hnutile
# from keras_self_attention import SeqSelfAttention
# from keras_self_attention import SeqSelfAttention

# 루트 디렉토리
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# 모델 저장 루트 디렉토리명
MODEL_ROOT_DIR_NAME = 'model'
# 모델 저장 서브 디렉토리명
MODEL_SUB_DIR_NAME = 'lstm-state'
# 모델파일 접두사
MODEL_FILE_PREFIX = 'lstm-state-model-'
# 학습데이터 루트 디렉토리명
TRAIN_ROOT_DIR_NAME = 'data'
# 학습데이터 서브 디렉토리명
TRAIN_SUB_DIR_NAME = 'roll'
# 학습데이터 파일 접두사
TRAIN_FILR_PREFIX = 'train-01-'
# 학습데이터 경로
CSV_FILE_PATH = os.path.join(ROOT_DIR, TRAIN_ROOT_DIR_NAME, TRAIN_SUB_DIR_NAME, TRAIN_FILR_PREFIX)

# 학습, 예측 데이터 루프백 사이즈
LOOK_BACK = 30

FIT_PATIENCE = 80

def run_trainning(model, fits, x_trains, y_trains, x_vals, y_vals):
    '''
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
            hist = model.fit(x_trains[i], y_trains[i], epochs=1, batch_size=1, shuffle=False, callbacks=[custom_hist], validation_data=(x_vals[i%10], y_vals[i%10]))
            model.reset_states()
            # 조기 종료 체크
            if custom_hist.early_stop():
                break

        # Bset model 선정
        base_model_name = hnutile.FindBestModel(os.path.join(ROOT_DIR, MODEL_ROOT_DIR_NAME, MODEL_SUB_DIR_NAME, '%03d'%(i)), MODEL_FILE_PREFIX)
        if base_model_name != None:
            print('File %03dth: The best model to be use is model >> %s.'%(i ,base_model_name))
            model = tf.keras.models.load_model(os.path.join(ROOT_DIR, MODEL_ROOT_DIR_NAME, MODEL_SUB_DIR_NAME, '%03d'%(i), base_model_name))
        else:
            print('File %03dth: Best model does not exist.'%(i))


def create_state_lstm_model(look_back=1):
    '''
    상태 전이 LSTM모델 생성
    '''
    model = tf.keras.Sequential()

    for i in range(3):
        model.add(tf.keras.layers.LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True, return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(128, batch_input_shape=(1, look_back, 1), stateful=True))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def main():
    '''
    메인 함수
    '''

    datas = []
    # train-01-001.csv ~ train-01-100.csv 까지 100개의 데이터 사용
    for i in range(100):
        file_path = "%s%.3d.csv"%(CSV_FILE_PATH, i+1)
        datas.append(pd.read_csv(file_path))
        print('%s 파일 데이터 추가.'%(file_path))

    # 그래프 출력
    hnutile.ShowGraph(datas, yaxis='RollLoad')

    load_datas = []
    for df in datas:
        load_datas.append(df['RollLoad'].values[:, None])

    # 학습, 검증 데이터 생성
    x_trains, y_trains, x_vals, y_vals = hnutile.CreateTrainDatas(datas=load_datas, look_back=LOOK_BACK)

    # 모델 생성
    model = create_state_lstm_model(LOOK_BACK)

    run_trainning(model, 200,x_trains, y_trains, x_vals, y_vals)
    print('Exit process.')


if __name__ == '__main__':
    main()