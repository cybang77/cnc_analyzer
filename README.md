# cnc_analyzer

## 의존성
* python 3.7.10
* tensorflow-gpu 2.4.1
* matplotlib 3.4.2
* pandas 1.2.4
* ipykernel 5.3.4

## 디렉토리 및 파일 구성
* 데이터 원본
  - input 폴더
* 데이터 전처리
  - preprocess_smoothing.ipynb : 롤링(평균이동) 기법 적용
  - preprocess_smoothing_cont.ipynb : 롤링(평균이동) 기법과 연속데이터 형식
  - data
    + roll : preprocess_smoothing.ipynb 롤링(평균이동) 기법 적용 전처리 데이터
    + cont-roll : 롤링(평균이동) 기법 적용 연속된 전처리 데이터
* State-LSTM
  - lstm_state_model_train.py
* Bi-State-LSTM
  - bi_lstm_state_model_train.py
* Bi-Saate_LSTM-Cont
  - bi_lstm_state_cont_model_train.py
* Bi-State-GRU
  - bi_gru_state_model_train.py
* 모델 평가
  - model_predict.ipynb
* 모델 저장(대용량으로 git에서는 제외)
  - model
    + lstm-state
    + bi-lstm-state
    + bi-lstm-state-cont
    + bi-gru-state
