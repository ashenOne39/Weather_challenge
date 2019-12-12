import os
from os import listdir

import numpy as np
import pandas as pd

import cv2

import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Activation, Lambda,\
                         Conv2D, RepeatVector, TimeDistributed, Flatten
from tensorflow.keras.models import Sequential, load_model


os.makedirs('predictions_by_each_model/', exist_ok=True)

def make_predictions(model_name, resize):

    X_test = np.load(f'X_test_folder/X_test_resize{resize}_float32.npy')

    model = load_model('trained_models/' + model_name + '.h5')

    for i in range(50):
        
        predictions = model.predict(X_test[i:(i+1)])
        
        # 24時間分の予測から、対象時刻のみを抜き出す
        for j in [5, 11, 17, 23]:
            
            img = predictions[0, j, :, :, 0]

            resized_img = cv2.resize(img, (512, 672), interpolation=cv2.INTER_LINEAR)
            trimmed_img = resized_img[40:40+420, 130:130+340]
            one_predict = np.round(trimmed_img * 255)
            
            if i == 0 and j == 5:
                
                predict_matrix = one_predict
                
            else:
                
                predict_matrix = np.vstack([predict_matrix, one_predict])

    predict_matrix = np.array(predict_matrix, dtype='int64')
            
    np.save(f'predictions_by_each_model/{model_name}_prediction.npy', predict_matrix)

# make_predictions関数の実行
make_predictions(model_name='model_resize4', resize=4)
make_predictions(model_name='model_resize6', resize=6)
make_predictions(model_name='model_resize8_a', resize=8)
make_predictions(model_name='model_resize8_b', resize=8)
make_predictions(model_name='model_resize10', resize=10)
make_predictions(model_name='model_resize12', resize=12)
make_predictions(model_name='model_resize15', resize=15)



'''
次に、それぞれのモデルの予測結果の平均を取るアンサンブルを行う。
保存されたNumPyデータを読み込み、その合計をモデルの数で割る。

その後、sample_submitを読み込み、予測結果と連結させることで、
提出用フォーマットに従うCSVファイルを作成。
'''

def make_ensemble_prediction():

    os.makedirs('ensemble_prediction/', exist_ok=True)

    predictions_file_list = listdir('predictions_by_each_model/')
    length = len(predictions_file_list)

    for i, file_name in enumerate(predictions_file_list):

        if i == 0:
            sum_preds = np.load('predictions_by_each_model/' + file_name)
        else:
            new_preds = np.load('predictions_by_each_model/' + file_name)
            sum_preds = sum_preds + new_preds
    
    mean_preds = np.round(sum_preds / length).astype('int64')

    # sample_submit.csvの読み込み
    sample_submission = pd.read_csv('sample_submit.csv', header=None)
    ensemble_submission = pd.concat([sample_submission[[0]], 
                               pd.DataFrame(mean_preds)], axis=1)

    ensemble_submission.to_csv('ensemble_prediction/ensemble_prediction.csv',
                                index=False, header=False)

# make_ensemble_prediction関数を実行
make_ensemble_prediction()

