# cnc_analyzer
* 데이터 원본
  - input 폴더
* 데이터 전처리
  - data
    + roll : preprocess_smoothing_op10_3.ipynb 롤링(평균이동) 기법 적용
* State-LSTM
  - lstm_state_model_train_op10_3_001.py
* Bi-State-LSTM
  - bi_lstm_state_model_train_op10_3_001.py
* Bi-State-GRU
  - bi_gru_state_model_train_op10_3.py
* 모델 평가
  - model_predict.ipynb
* 모델 저장
  - model
    + lstm-state
    + bi-lstm-state
    + bi-gru-state
