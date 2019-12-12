import os
from os import listdir

import numpy as np
import pandas as pd

import cv2

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import ConvLSTM2D, Activation, Lambda,\
                                    Conv2D, RepeatVector, TimeDistributed, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence

os.makedirs('trained_models/', exist_ok=True)
os.makedirs('checkpoint/', exist_ok=True)

# TrainGeneratorの設定
class TrainGenerator(Sequence):
    
    def __init__(self, sat_path, met_HPRT_path, met_Wind_path,
                 resize, batch_size):
        
        self.sat_path = sat_path
        self.met_HPRT_path = met_HPRT_path
        self.met_Wind_path = met_Wind_path
        
        self.resize = resize
        self.img_height = int(672 / resize)
        self.img_width = int(512 / resize)
        
        self.sample_size = len(sat_path) - 1
        self.batch_size = batch_size
        self.steps_per_epoch = int((self.sample_size - 1) / batch_size)+ 1
        
           
    def __getitem__(self, idx):
        
        X_start_pos = self.batch_size * idx
        X_end_pos = X_start_pos + self.batch_size
        if X_end_pos > self.sample_size:
            X_end_pos = self.sample_size
        
        X_batch = []
        Y_batch = []
        
        for i in range(X_start_pos, X_end_pos):
        
            # Xの読み込み
            X_sat = np.load(f'resize{self.resize}/train/sat/' + self.sat_path[i])
            X_sat = X_sat.astype('float32') / 255
                    
            X_met_HPRT = np.load(f'resize{self.resize}/train/met_HPRT/' + self.met_HPRT_path[i])
            X_met_Wind = np.load(f'resize{self.resize}/train/met_Wind/' + self.met_Wind_path[i])

            X_data = np.concatenate([X_sat, X_met_HPRT, X_met_Wind], axis=4)
            
            # Yの読み込み
            Y_sat = np.load(f'resize{self.resize}/train/sat/' + self.sat_path[i + 1])
            Y_sat = Y_sat.astype('float32') / 255
            
            X_batch.append(X_data)
            Y_batch.append(Y_sat)
            
        X_batch = np.asarray(X_batch, dtype='float32').reshape((X_end_pos - X_start_pos), 24, self.img_height, self.img_width, 52)
        Y_batch = np.asarray(Y_batch, dtype='float32').reshape((X_end_pos - X_start_pos), 24, self.img_height, self.img_width, 1)
            
        return X_batch, Y_batch
        
    def __len__(self, ):
        return self.steps_per_epoch
    
    def __on_epoch_end__(self, ):
        pass



# ValidGeneratorの設定
class ValidGenerator(Sequence):
    
    def __init__(self, X_sat_path, X_met_HPRT_path, X_met_Wind_path, Y_path, 
                 resize, batch_size):
        
        self.X_sat_path = X_sat_path
        self.X_met_HPRT_path = X_met_HPRT_path
        self.X_met_Wind_path = X_met_Wind_path
        self.Y_path = Y_path
        
        self.resize = resize
        self.img_height = int(672 / resize)
        self.img_width = int(512 / resize)
        
        self.sample_size = 150
        self.batch_size = batch_size
        self.steps_per_epoch = int((self.sample_size - 1) / batch_size)+ 1
        
        
        
    def __getitem__(self, idx):
        
        X_start_pos = self.batch_size * idx
        X_end_pos = X_start_pos + self.batch_size
        if X_end_pos > self.sample_size:
            X_end_pos = self.sample_size
        
        X_batch = []
        Y_batch = []
        
        for i in range(X_start_pos, X_end_pos):
            # Xの読み込み
            X_sat = np.load(f'resize{self.resize}/valid/X_sat/' + self.X_sat_path[i])
            X_sat = X_sat.astype('float32') / 255
            
            X_met_HPRT = np.load(f'resize{self.resize}/valid/X_met_HPRT/' + self.X_met_HPRT_path[i])
            X_met_Wind = np.load(f'resize{self.resize}/valid/X_met_Wind/' + self.X_met_Wind_path[i])
            
            X_data = np.concatenate([X_sat, X_met_HPRT, X_met_Wind], axis=4)
            
            # Yの読み込み
            Y_sat = np.load(f'resize{self.resize}/valid/Y/' + self.Y_path[i])
            Y_sat = Y_sat.astype('float32') / 255
            
            X_batch.append(X_data)
            Y_batch.append(Y_sat)
            
        X_batch = np.asarray(X_batch).reshape((X_end_pos - X_start_pos), 24, self.img_height, self.img_width, 52)
        Y_batch = np.asarray(Y_batch).reshape((X_end_pos - X_start_pos), 24, self.img_height, self.img_width, 1)
            
        return X_batch, Y_batch
        
    def __len__(self, ):
        return self.steps_per_epoch
    
    def __on_epoch_end__(self, ):
        pass


# model_resize4の学習
def model_resize4(resize=4):
    os.makedirs('checkpoint/model_resize4/')

    # Trainデータのpathを取得
    train_sat_dir = f'resize{resize}/train/sat/'
    train_met_HPRT_dir = f'resize{resize}/train/met_HPRT/'
    train_met_Wind_dir = f'resize{resize}/train/met_Wind/'

    # ファイル名のリストを取得
    train_sat_file_list = listdir(train_sat_dir)
    train_met_HPRT_file_list = listdir(train_met_HPRT_dir)
    train_met_Wind_file_list = listdir(train_met_Wind_dir)

    # 順番を整える
    train_sat_file_list.sort()
    train_met_HPRT_file_list.sort()
    train_met_Wind_file_list.sort()

    # Validデータのpathを取得
    X_valid_sat_dir = f'resize{resize}/valid/X_sat/'
    X_valid_met_HPRT_dir = f'resize{resize}/valid/X_met_HPRT/'
    X_valid_met_Wind_dir = f'resize{resize}/valid/X_met_Wind/'
    Y_valid_dir = f'resize{resize}/valid/Y/'

    # ファイル名のリストを取得
    X_valid_sat_file_list = listdir(X_valid_sat_dir)
    X_valid_met_HPRT_file_list = listdir(X_valid_met_HPRT_dir)
    X_valid_met_Wind_file_list = listdir(X_valid_met_Wind_dir)
    Y_valid_file_list = listdir(Y_valid_dir)

    # 順番を整える
    X_valid_sat_file_list.sort()
    X_valid_met_HPRT_file_list.sort()
    X_valid_met_Wind_file_list.sort()
    Y_valid_file_list.sort()

    # train_generator生成
    train_generator = TrainGenerator(train_sat_file_list,
                                    train_met_HPRT_file_list,
                                    train_met_Wind_file_list,
                                    resize=4,
                                    batch_size=10)

    # valid_generator生成
    valid_generator = ValidGenerator(X_valid_sat_file_list,
                                    X_valid_met_HPRT_file_list,
                                    X_valid_met_Wind_file_list,
                                    Y_valid_file_list,
                                    resize=4,
                                    batch_size=10)

    # modelの定義
    model = Sequential()

    # encoder 1
    model.add(ConvLSTM2D(filters=48, kernel_size=(5, 5),
                        padding='same', return_sequences=False,
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='glorot_uniform',
                        activation='tanh',
                        recurrent_activation='sigmoid',
                        input_shape=(24, 168, 128, 52)))

    # Repeat last status
    model.add(Flatten())
    model.add(RepeatVector(24))
    model.add(Lambda(lambda x: K.reshape(x, (-1, 24, 168, 128, 48)),
                    output_shape=(24, 168, 128, 48)))

    # decoder 1
    model.add(ConvLSTM2D(filters=48, kernel_size=(5, 5),
                        padding='same', return_sequences=True,
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='glorot_uniform',
                        activation='tanh',
                        recurrent_activation='sigmoid'))

    # output layer
    model.add(TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1),
                    kernel_initializer='glorot_uniform')))
    model.add(Activation('sigmoid'))

    model.compile(loss='mae',
                optimizer='adam')


    # fit model
    epochs = 150

    e_stopping = EarlyStopping(verbose=1, patience=40, restore_best_weights=True)
    checkpoint = ModelCheckpoint('checkpoint/model_resize4/model_resize4_Loss_{loss:.4f}_vLoss_{val_loss:.4f}.h5')

    model.fit_generator(train_generator, epochs=epochs,
                        validation_data=valid_generator, callbacks=[e_stopping, checkpoint],
                        max_queue_size=2)

    model.save('trained_models/model_resize4.h5')



# model_resize6の学習
def model_resize6(resize=6):
    os.makedirs('checkpoint/model_resize6/')

    # Trainデータのpathを取得
    train_sat_dir = f'resize{resize}/train/sat/'
    train_met_HPRT_dir = f'resize{resize}/train/met_HPRT/'
    train_met_Wind_dir = f'resize{resize}/train/met_Wind/'

    # ファイル名のリストを取得
    train_sat_file_list = listdir(train_sat_dir)
    train_met_HPRT_file_list = listdir(train_met_HPRT_dir)
    train_met_Wind_file_list = listdir(train_met_Wind_dir)

    # 順番を整える
    train_sat_file_list.sort()
    train_met_HPRT_file_list.sort()
    train_met_Wind_file_list.sort()

    # Validデータのpathを取得
    X_valid_sat_dir = f'resize{resize}/valid/X_sat/'
    X_valid_met_HPRT_dir = f'resize{resize}/valid/X_met_HPRT/'
    X_valid_met_Wind_dir = f'resize{resize}/valid/X_met_Wind/'
    Y_valid_dir = f'resize{resize}/valid/Y/'

    # ファイル名のリストを取得
    X_valid_sat_file_list = listdir(X_valid_sat_dir)
    X_valid_met_HPRT_file_list = listdir(X_valid_met_HPRT_dir)
    X_valid_met_Wind_file_list = listdir(X_valid_met_Wind_dir)
    Y_valid_file_list = listdir(Y_valid_dir)

    # 順番を整える
    X_valid_sat_file_list.sort()
    X_valid_met_HPRT_file_list.sort()
    X_valid_met_Wind_file_list.sort()
    Y_valid_file_list.sort()

    # train_generator生成
    train_generator = TrainGenerator(train_sat_file_list,
                                    train_met_HPRT_file_list,
                                    train_met_Wind_file_list,
                                    resize=6,
                                    batch_size=16)

    # valid_generator生成
    valid_generator = ValidGenerator(X_valid_sat_file_list,
                                    X_valid_met_HPRT_file_list,
                                    X_valid_met_Wind_file_list,
                                    Y_valid_file_list,
                                    resize=6,
                                    batch_size=16)
    

    model = Sequential()

    # encoder
    model.add(ConvLSTM2D(filters=48, kernel_size=(5, 5),
                        padding='same', return_sequences=False,
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='glorot_uniform',
                        activation='tanh',
                        recurrent_activation='sigmoid',
                        input_shape=(24, 112, 85, 52)))

    # repeat last status
    model.add(Flatten())
    model.add(RepeatVector(24))
    model.add(Lambda(lambda x: K.reshape(x, (-1, 24, 112, 85, 48)),
                    output_shape=(24, 112, 85, 48)))

    ### Decoder(Forecasting)

    # decoder
    model.add(ConvLSTM2D(filters=48, kernel_size=(5, 5),
                        padding='same', return_sequences=True,
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='glorot_uniform',
                        activation='tanh',
                        recurrent_activation='sigmoid'))

    # output layer
    model.add(TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1),
                    kernel_initializer='glorot_uniform')))
    model.add(Activation('sigmoid'))

    model.compile(loss='mae',
                optimizer='adam')

    epochs = 200

    e_stopping = EarlyStopping(verbose=1, patience=30, restore_best_weights=True)
    checkpoint = ModelCheckpoint('checkpoint/model_resize6/model_resize6_Loss_{loss:.4f}_vLoss_{val_loss:.4f}.h5')

    model.fit_generator(train_generator, epochs=epochs,
                        validation_data=valid_generator, callbacks=[e_stopping, checkpoint],
                        max_queue_size=3)

    model.save('trained_models/model_resize6.h5')



# model_resize8_aの定義
def model_resize8_a(resize=8):
    os.makedirs('checkpoint/model_resize8_a/')

    # Trainデータのpathを取得
    train_sat_dir = f'resize{resize}/train/sat/'
    train_met_HPRT_dir = f'resize{resize}/train/met_HPRT/'
    train_met_Wind_dir = f'resize{resize}/train/met_Wind/'

    # ファイル名のリストを取得
    train_sat_file_list = listdir(train_sat_dir)
    train_met_HPRT_file_list = listdir(train_met_HPRT_dir)
    train_met_Wind_file_list = listdir(train_met_Wind_dir)

    # 順番を整える
    train_sat_file_list.sort()
    train_met_HPRT_file_list.sort()
    train_met_Wind_file_list.sort()

    # Validデータのpathを取得
    X_valid_sat_dir = f'resize{resize}/valid/X_sat/'
    X_valid_met_HPRT_dir = f'resize{resize}/valid/X_met_HPRT/'
    X_valid_met_Wind_dir = f'resize{resize}/valid/X_met_Wind/'
    Y_valid_dir = f'resize{resize}/valid/Y/'

    # ファイル名のリストを取得
    X_valid_sat_file_list = listdir(X_valid_sat_dir)
    X_valid_met_HPRT_file_list = listdir(X_valid_met_HPRT_dir)
    X_valid_met_Wind_file_list = listdir(X_valid_met_Wind_dir)
    Y_valid_file_list = listdir(Y_valid_dir)

    # 順番を整える
    X_valid_sat_file_list.sort()
    X_valid_met_HPRT_file_list.sort()
    X_valid_met_Wind_file_list.sort()
    Y_valid_file_list.sort()

    # train_generator生成
    train_generator = TrainGenerator(train_sat_file_list,
                                    train_met_HPRT_file_list,
                                    train_met_Wind_file_list,
                                    resize=8,
                                    batch_size=16)

    # valid_generator生成
    valid_generator = ValidGenerator(X_valid_sat_file_list,
                                    X_valid_met_HPRT_file_list,
                                    X_valid_met_Wind_file_list,
                                    Y_valid_file_list,
                                    resize=8,
                                    batch_size=16)


    model = Sequential()

    # encoder
    model.add(ConvLSTM2D(filters=48, kernel_size=(6, 6),
                        padding='same', return_sequences=False,
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='glorot_uniform',
                        activation='sigmoid',
                        recurrent_activation='sigmoid',
                        input_shape=(24, 84, 64, 52)))

    # repeat last status
    model.add(Flatten())
    model.add(RepeatVector(24))
    model.add(Lambda(lambda x: K.reshape(x, (-1, 24, 84, 64, 48)),
                    output_shape=(24, 84, 64, 48)))

    # decoder
    model.add(ConvLSTM2D(filters=48, kernel_size=(6, 6),
                        padding='same', return_sequences=True,
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='glorot_uniform',
                        activation='sigmoid',
                        recurrent_activation='sigmoid'))

    # output layer
    model.add(TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1),
                                    kernel_initializer='glorot_uniform')))
    model.add(Activation('sigmoid'))

    model.compile(loss='mae',
                optimizer='adam')

    epochs = 230

    e_stopping = EarlyStopping(verbose=1, patience=50, restore_best_weights=True)
    checkpoint = ModelCheckpoint('checkpoint/model_resize8_a/model_resize8_a_Loss_{loss:.4f}_vLoss_{val_loss:.4f}.h5')

    model.fit_generator(train_generator, epochs=epochs,
                        validation_data=valid_generator, callbacks=[e_stopping, checkpoint],
                        max_queue_size=3)

    model.save('trained_models/model_resize8_a.h5')



# model_resize8_bの定義
def model_resize8_b(resize=8):
    os.makedirs('checkpoint/model_resize8_b/')

    # Trainデータのpathを取得
    train_sat_dir = f'resize{resize}/train/sat/'
    train_met_HPRT_dir = f'resize{resize}/train/met_HPRT/'
    train_met_Wind_dir = f'resize{resize}/train/met_Wind/'

    # ファイル名のリストを取得
    train_sat_file_list = listdir(train_sat_dir)
    train_met_HPRT_file_list = listdir(train_met_HPRT_dir)
    train_met_Wind_file_list = listdir(train_met_Wind_dir)

    # 順番を整える
    train_sat_file_list.sort()
    train_met_HPRT_file_list.sort()
    train_met_Wind_file_list.sort()

    # Validデータのpathを取得
    X_valid_sat_dir = f'resize{resize}/valid/X_sat/'
    X_valid_met_HPRT_dir = f'resize{resize}/valid/X_met_HPRT/'
    X_valid_met_Wind_dir = f'resize{resize}/valid/X_met_Wind/'
    Y_valid_dir = f'resize{resize}/valid/Y/'

    # ファイル名のリストを取得
    X_valid_sat_file_list = listdir(X_valid_sat_dir)
    X_valid_met_HPRT_file_list = listdir(X_valid_met_HPRT_dir)
    X_valid_met_Wind_file_list = listdir(X_valid_met_Wind_dir)
    Y_valid_file_list = listdir(Y_valid_dir)

    # 順番を整える
    X_valid_sat_file_list.sort()
    X_valid_met_HPRT_file_list.sort()
    X_valid_met_Wind_file_list.sort()
    Y_valid_file_list.sort()

    # train_generator生成
    train_generator = TrainGenerator(train_sat_file_list,
                                    train_met_HPRT_file_list,
                                    train_met_Wind_file_list,
                                    resize=8,
                                    batch_size=20)

    # valid_generator生成
    valid_generator = ValidGenerator(X_valid_sat_file_list,
                                    X_valid_met_HPRT_file_list,
                                    X_valid_met_Wind_file_list,
                                    Y_valid_file_list,
                                    resize=8,
                                    batch_size=20)

    model = Sequential()

    # encoder 1
    model.add(ConvLSTM2D(filters=80, kernel_size=(5, 5),
                        padding='same', return_sequences=False,
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='glorot_uniform',
                        activation='tanh',
                        recurrent_activation='sigmoid',
                        input_shape=(24, 84, 64, 52)))

    # repeat last status
    model.add(Flatten())
    model.add(RepeatVector(24))
    model.add(Lambda(lambda x: K.reshape(x, (-1, 24, 84, 64, 80)),
                    output_shape=(24, 84, 64, 80)))

    # decoder
    model.add(ConvLSTM2D(filters=80, kernel_size=(5, 5),
                        padding='same', return_sequences=True,
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='glorot_uniform',
                        activation='tanh',
                        recurrent_activation='sigmoid'))

    # output layer
    model.add(TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1),
                                    kernel_initializer='glorot_uniform')))
    model.add(Activation('sigmoid'))

    model.compile(loss='mae',
                optimizer='adam')

    epochs = 160

    e_stopping = EarlyStopping(verbose=1, patience=30, restore_best_weights=True)
    checkpoint = ModelCheckpoint('checkpoint/model_resize8_b/model_resize8_b_Loss_{loss:.4f}_vLoss_{val_loss:.4f}.h5')

    model.fit_generator(train_generator, epochs=epochs,
                        validation_data=valid_generator, callbacks=[e_stopping, checkpoint],
                        max_queue_size=5)

    model.save('trained_models/model_resize8_b.h5')



# model_resize10の定義
def model_resize10(resize=10):
    os.makedirs('checkpoint/model_resize10/')

    # Trainデータのpathを取得
    train_sat_dir = f'resize{resize}/train/sat/'
    train_met_HPRT_dir = f'resize{resize}/train/met_HPRT/'
    train_met_Wind_dir = f'resize{resize}/train/met_Wind/'

    # ファイル名のリストを取得
    train_sat_file_list = listdir(train_sat_dir)
    train_met_HPRT_file_list = listdir(train_met_HPRT_dir)
    train_met_Wind_file_list = listdir(train_met_Wind_dir)

    # 順番を整える
    train_sat_file_list.sort()
    train_met_HPRT_file_list.sort()
    train_met_Wind_file_list.sort()

    # Validデータのpathを取得
    X_valid_sat_dir = f'resize{resize}/valid/X_sat/'
    X_valid_met_HPRT_dir = f'resize{resize}/valid/X_met_HPRT/'
    X_valid_met_Wind_dir = f'resize{resize}/valid/X_met_Wind/'
    Y_valid_dir = f'resize{resize}/valid/Y/'

    # ファイル名のリストを取得
    X_valid_sat_file_list = listdir(X_valid_sat_dir)
    X_valid_met_HPRT_file_list = listdir(X_valid_met_HPRT_dir)
    X_valid_met_Wind_file_list = listdir(X_valid_met_Wind_dir)
    Y_valid_file_list = listdir(Y_valid_dir)

    # 順番を整える
    X_valid_sat_file_list.sort()
    X_valid_met_HPRT_file_list.sort()
    X_valid_met_Wind_file_list.sort()
    Y_valid_file_list.sort()

    # train_generator生成
    train_generator = TrainGenerator(train_sat_file_list,
                                    train_met_HPRT_file_list,
                                    train_met_Wind_file_list,
                                    resize=10,
                                    batch_size=24)

    # valid_generator生成
    valid_generator = ValidGenerator(X_valid_sat_file_list,
                                    X_valid_met_HPRT_file_list,
                                    X_valid_met_Wind_file_list,
                                    Y_valid_file_list,
                                    resize=10,
                                    batch_size=24)

    model = Sequential()

    # encoder
    model.add(ConvLSTM2D(filters=64, kernel_size=(5, 5),
                        padding='same', return_sequences=False,
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='glorot_uniform',
                        activation='tanh',
                        recurrent_activation='sigmoid',
                        input_shape=(24, 67, 51, 52)))

    # repeat last status
    model.add(Flatten())
    model.add(RepeatVector(24))
    model.add(Lambda(lambda x: K.reshape(x, (-1, 24, 67, 51, 64)),
                    output_shape=(24, 67, 51, 64)))

    # decoder
    model.add(ConvLSTM2D(filters=64, kernel_size=(5, 5),
                        padding='same', return_sequences=True,
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='glorot_uniform',
                        activation='tanh',
                        recurrent_activation='sigmoid'))

    # output layer
    model.add(TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1),
                    kernel_initializer='glorot_uniform')))
    model.add(Activation('sigmoid'))

    model.compile(loss='mae',
                optimizer='adam')

    epochs = 180

    e_stopping = EarlyStopping(verbose=1, patience=30, restore_best_weights=True)
    checkpoint = ModelCheckpoint('checkpoint/model_resize10/model_resize10_Loss_{loss:.4f}_vLoss_{val_loss:.4f}.h5')

    model.fit_generator(train_generator, epochs=epochs,
                        validation_data=valid_generator, callbacks=[e_stopping, checkpoint],
                        max_queue_size=3)

    model.save('trained_models/model_resize10.h5')



# model_resize12の定義
def model_resize12(resize=12):
    os.makedirs('checkpoint/model_resize12/')

    # Trainデータのpathを取得
    train_sat_dir = f'resize{resize}/train/sat/'
    train_met_HPRT_dir = f'resize{resize}/train/met_HPRT/'
    train_met_Wind_dir = f'resize{resize}/train/met_Wind/'

    # ファイル名のリストを取得
    train_sat_file_list = listdir(train_sat_dir)
    train_met_HPRT_file_list = listdir(train_met_HPRT_dir)
    train_met_Wind_file_list = listdir(train_met_Wind_dir)

    # 順番を整える
    train_sat_file_list.sort()
    train_met_HPRT_file_list.sort()
    train_met_Wind_file_list.sort()

    # Validデータのpathを取得
    X_valid_sat_dir = f'resize{resize}/valid/X_sat/'
    X_valid_met_HPRT_dir = f'resize{resize}/valid/X_met_HPRT/'
    X_valid_met_Wind_dir = f'resize{resize}/valid/X_met_Wind/'
    Y_valid_dir = f'resize{resize}/valid/Y/'

    # ファイル名のリストを取得
    X_valid_sat_file_list = listdir(X_valid_sat_dir)
    X_valid_met_HPRT_file_list = listdir(X_valid_met_HPRT_dir)
    X_valid_met_Wind_file_list = listdir(X_valid_met_Wind_dir)
    Y_valid_file_list = listdir(Y_valid_dir)

    # 順番を整える
    X_valid_sat_file_list.sort()
    X_valid_met_HPRT_file_list.sort()
    X_valid_met_Wind_file_list.sort()
    Y_valid_file_list.sort()

    # train_generator生成
    train_generator = TrainGenerator(train_sat_file_list,
                                    train_met_HPRT_file_list,
                                    train_met_Wind_file_list,
                                    resize=12,
                                    batch_size=36)

    # valid_generator生成
    valid_generator = ValidGenerator(X_valid_sat_file_list,
                                    X_valid_met_HPRT_file_list,
                                    X_valid_met_Wind_file_list,
                                    Y_valid_file_list,
                                    resize=12,
                                    batch_size=36)


    model = Sequential()

    # encoder 1
    model.add(ConvLSTM2D(filters=80, kernel_size=(4, 4),
                        padding='same', return_sequences=False,
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='glorot_uniform',
                        activation='sigmoid',
                        recurrent_activation='sigmoid',
                        input_shape=(24, 56, 42, 52)))

    # repeat last status
    model.add(Flatten())
    model.add(RepeatVector(24))
    model.add(Lambda(lambda x: K.reshape(x, (-1, 24, 56, 42, 80)),
                    output_shape=(24, 56, 42, 80)))

    # decoder
    model.add(ConvLSTM2D(filters=80, kernel_size=(4, 4),
                        padding='same', return_sequences=True,
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='glorot_uniform',
                        activation='sigmoid',
                        recurrent_activation='sigmoid'))

    # output layer
    model.add(TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1),
                    kernel_initializer='glorot_uniform')))
    model.add(Activation('sigmoid'))

    model.compile(loss='mae',
                optimizer='adam')


    epochs = 350

    e_stopping = EarlyStopping(verbose=1, patience=50, restore_best_weights=True)
    checkpoint = ModelCheckpoint('checkpoint/model_resize12/model_resize12_Loss_{loss:.4f}_vLoss_{val_loss:.4f}.h5')

    model.fit_generator(train_generator, epochs=epochs,
                        validation_data=valid_generator, callbacks=[e_stopping, checkpoint],
                        max_queue_size=3)

    model.save('trained_models/model_resize12.h5')



# model_resize15の定義
def model_resize15(resize=15):
    os.makedirs('checkpoint/model_resize15/')

    # Trainデータのpathを取得
    train_sat_dir = f'resize{resize}/train/sat/'
    train_met_HPRT_dir = f'resize{resize}/train/met_HPRT/'
    train_met_Wind_dir = f'resize{resize}/train/met_Wind/'

    # ファイル名のリストを取得
    train_sat_file_list = listdir(train_sat_dir)
    train_met_HPRT_file_list = listdir(train_met_HPRT_dir)
    train_met_Wind_file_list = listdir(train_met_Wind_dir)

    # 順番を整える
    train_sat_file_list.sort()
    train_met_HPRT_file_list.sort()
    train_met_Wind_file_list.sort()

    # Validデータのpathを取得
    X_valid_sat_dir = f'resize{resize}/valid/X_sat/'
    X_valid_met_HPRT_dir = f'resize{resize}/valid/X_met_HPRT/'
    X_valid_met_Wind_dir = f'resize{resize}/valid/X_met_Wind/'
    Y_valid_dir = f'resize{resize}/valid/Y/'

    # ファイル名のリストを取得
    X_valid_sat_file_list = listdir(X_valid_sat_dir)
    X_valid_met_HPRT_file_list = listdir(X_valid_met_HPRT_dir)
    X_valid_met_Wind_file_list = listdir(X_valid_met_Wind_dir)
    Y_valid_file_list = listdir(Y_valid_dir)

    # 順番を整える
    X_valid_sat_file_list.sort()
    X_valid_met_HPRT_file_list.sort()
    X_valid_met_Wind_file_list.sort()
    Y_valid_file_list.sort()

    # train_generator生成
    train_generator = TrainGenerator(train_sat_file_list,
                                    train_met_HPRT_file_list,
                                    train_met_Wind_file_list,
                                    resize=15,
                                    batch_size=48)

    # valid_generator生成
    valid_generator = ValidGenerator(X_valid_sat_file_list,
                                    X_valid_met_HPRT_file_list,
                                    X_valid_met_Wind_file_list,
                                    Y_valid_file_list,
                                    resize=15,
                                    batch_size=48)

    model = Sequential()

    # encoder
    model.add(ConvLSTM2D(filters=80, kernel_size=(3, 3),
                        padding='same', return_sequences=False,
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='glorot_uniform',
                        activation='tanh',
                        recurrent_activation='sigmoid',
                        input_shape=(24, 44, 34, 52)))

    # repeat last status
    model.add(Flatten())
    model.add(RepeatVector(24))
    model.add(Lambda(lambda x: K.reshape(x, (-1, 24, 44, 34, 80)),
                    output_shape=(24, 44, 34, 80)))

    # decoder
    model.add(ConvLSTM2D(filters=80, kernel_size=(3, 3),
                        padding='same', return_sequences=True,
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='glorot_uniform',
                        activation='tanh',
                        recurrent_activation='sigmoid'))

    # output layer
    model.add(TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1),
                    kernel_initializer='glorot_uniform')))
    model.add(Activation('sigmoid'))

    model.compile(loss='mae',
                optimizer='adam')

    epochs = 160

    e_stopping = EarlyStopping(verbose=1, patience=40, restore_best_weights=True)
    checkpoint = ModelCheckpoint('checkpoint/model_resize15/model_resize15_Loss_{loss:.4f}_vLoss_{val_loss:.4f}.h5')

    model.fit_generator(train_generator, epochs=epochs,
                        validation_data=valid_generator, callbacks=[e_stopping, checkpoint])

    model.save('trained_models/model_resize15.h5')


# 各モデルの実行
model_resize4()
model_resize6()
model_resize8_a()
model_resize8_b()
model_resize10()
model_resize12()
model_resize15()





