# ファイル内で使用するモジュールのインポート
import os
from os import listdir

import numpy as np
import pandas as pd

from datetime import datetime as dt
from datetime import timedelta

import cv2

import gzip
import shutil

# 2016-01-01-01以降のtrainデータから欠損ファイルの日付を探し出す関数
def find_missing_train_files():

    start_date = dt(2016, 1, 1, 1, 0, 0)
    missing_date = []
    
    for i in range(365*24*2):
        date = start_date + timedelta(hours=i)
        file_name = "train/sat/{year}-{month:02}-{day:02}/{year}-{month:02}-{day:02}-{hour:02}-00.fv.png".\
                    format(year=date.year, month=date.month, day=date.day, hour=date.hour)
        if not os.path.exists(file_name):
            missing_date.append(date)
        
    return missing_date

# testデータから欠損ファイルの日付を探し出す関数
def find_missing_test_files():

    terms = pd.read_csv('inference_terms.csv')
    terms = pd.to_datetime(terms.iloc[:, 0])
    missing_date = []

    for i in range(50):

        start_date = terms[i]

        for j in range(96):

            date = start_date + timedelta(hours=j)
            file_name = "test/sat/{year}-{month:02}-{day:02}/{year}-{month:02}-{day:02}-{hour:02}-00.fv.png".\
                        format(year=date.year, month=date.month, day=date.day, hour=date.hour)

            if not os.path.exists(file_name):
                missing_date.append(date)
    return missing_date

'''
欠損データを前後のデータの平均値で埋める

注意点：
- ２０１６年２月１５日 １５時～１７時のファイルが３つ連続で欠けている。
- ２０１７年２月１３日 １５時～１６時のファイルが２つ連続で欠けている。

そのため、１時間前と１時間後のファイルの平均値による穴埋めはできない。
→先に一部のファイルを2時間前と２時間後の平均値で埋める

'''

# 2時間以上連続で欠損が続いているファイルを穴埋めするための関数
# 2016年2月15日16時のファイル
# 2017年2月13日15時のファイル
def fill_by_mean_2hours(date):
    
    file_name = "train/sat/{year}-{month:02}-{day:02}/{year}-{month:02}-{day:02}-{hour:02}-00.fv.png".\
                    format(year=date.year, month=date.month, day=date.day, hour=date.hour)
    
    prev_date = date - timedelta(hours=2)
    next_date = date + timedelta(hours=2)
    
    prev_file = "train/sat/{year}-{month:02}-{day:02}/{year}-{month:02}-{day:02}-{hour:02}-00.fv.png".\
                    format(year=prev_date.year, month=prev_date.month, day=prev_date.day, hour=prev_date.hour)
    next_file = "train/sat/{year}-{month:02}-{day:02}/{year}-{month:02}-{day:02}-{hour:02}-00.fv.png".\
                    format(year=next_date.year, month=next_date.month, day=next_date.day, hour=next_date.hour)
    
    prev_img = cv2.imread(prev_file, 0)
    next_img = cv2.imread(next_file, 0)
    
    mean_img = (prev_img.astype('float32') + next_img.astype('float32')) / 2
    mean_img = mean_img.astype('uint8')
    
    cv2.imwrite(file_name, mean_img)

# fill_by_mean_2hours関数を実行
fill_by_mean_2hours(dt(2016, 2, 15, 16, 0, 0))
fill_by_mean_2hours(dt(2017, 2, 13, 15, 0, 0))

# 欠損ファイルを前後1時間の平均で埋める関数
def fill_by_mean(date, train_or_test=None):
    
    file_name = "{train_or_test}/sat/{year}-{month:02}-{day:02}/{year}-{month:02}-{day:02}-{hour:02}-00.fv.png".\
                    format(train_or_test=train_or_test,
                           year=date.year, month=date.month, day=date.day, hour=date.hour)
    
    prev_date = date - timedelta(hours=1)
    next_date = date + timedelta(hours=1)
    
    prev_file_name = "{train_or_test}/sat/{year}-{month:02}-{day:02}/{year}-{month:02}-{day:02}-{hour:02}-00.fv.png".\
                    format(train_or_test=train_or_test,
                           year=prev_date.year, month=prev_date.month, day=prev_date.day, hour=prev_date.hour)
    next_file_name = "{train_or_test}/sat/{year}-{month:02}-{day:02}/{year}-{month:02}-{day:02}-{hour:02}-00.fv.png".\
                    format(train_or_test=train_or_test,
                           year=next_date.year, month=next_date.month, day=next_date.day, hour=next_date.hour)
    
    prev_img = cv2.imread(prev_file_name, 0)
    next_img = cv2.imread(next_file_name, 0)
    
    mean_img = (prev_img.astype('float32') + next_img.astype('float32')) / 2
    mean_img = mean_img.astype('uint8')
    
    cv2.imwrite(file_name, mean_img)

# find_missing_train_files, find_missing_test_files関数を実行し、欠損ファイルの日付をリスト化
train_missing_date_list = find_missing_train_files()
test_missing_date_list = find_missing_test_files()

# 欠損ファイルのそれぞれの日付に対して、fill_by_mean関数を実行
for train_missing_date in train_missing_date_list:
    fill_by_mean(train_missing_date, train_or_test='train')

for test_missing_date in test_missing_date_list:
    fill_by_mean(test_missing_date, train_or_test='test')



# (2016, 1, 1, 0, 0)の画像の破損部分を前後1時間の平均で埋める関数
def fill_broken_part(): 
    broken_file = 'train/sat/2017-01-01/2017-01-01-00-00.fv.png'
    prev_file = 'train/sat/2016-12-31/2016-12-31-23-00.fv.png'
    next_file = 'train/sat/2017-01-01/2017-01-01-01-00.fv.png'

    broken_img = cv2.imread(broken_file, 0)
    prev_img = cv2.imread(prev_file, 0)
    next_img = cv2.imread(next_file, 0)

    mean_img = (prev_img.astype('float32') + next_img.astype('float32')) / 2
    mean_img = mean_img.astype('uint8')

    # 壊れている部分にのみ平均値を埋め込む
    new_img = np.concatenate([mean_img[:110], broken_img[110:]], axis=0)

    cv2.imwrite(broken_file, new_img)

# fill_broken_part関数を実行
fill_broken_part()



'''
ここからは,

衛星データをsat
気象データをmet_HPRT, met_Windに分け、
モデルの学習に使用するデータセットの作成を行う。
(HPRTは高度、海面気圧、気温、湿度をまとめたもの、
 Windは東西風、南北風、鉛直流をまとめたもの)

また、画像のリサイズに際して、
例えば、「resize12」と言う表現は、元の画像を縦横それぞれ12分の1サイズに
縮小したことを意味する。

これらの表現はtrain.py, predict.pyにおいても同様

'''


'''
まずは衛星データの処理を行う
'''

# train_satデータ(衛星データ)の前処理を行う関数
def make_train_sat_dataset(resize, interpolation):
    
    dir_name = f'resize{resize}/train/sat/'
    os.makedirs(dir_name, exist_ok=True)
    
    start_date = dt(2016, 1, 1, 16, 0, 0)
    dataset = []
    
    for i in range(730):
        
        init_date = start_date + timedelta(days=i)
        
        data_in_1day = []
    
        for j in range(24):
            
            date = init_date + timedelta(hours=j)
            image_file_name = "train/sat/{year}-{month:02}-{day:02}/{year}-{month:02}-{day:02}-{hour:02}-00.fv.png".\
                               format(year=date.year, month=date.month, day=date.day, hour=date.hour)
            
            img = cv2.imread(image_file_name, 0)
            resized_img = cv2.resize(img, (int(512 / resize), int(672 / resize)),
                                     interpolation=interpolation)
            
            data_in_1day.append(resized_img)
            
        data_in_1day = np.array(data_in_1day, dtype='uint8').reshape(1, 24, int(672 / resize), int(512 / resize), 1)
        
        zfilled_i = str(i).zfill(3)
        numpy_file_name = f'resize{resize}/train/sat/train_sat_rs{resize}_{zfilled_i}.npy'
        np.save(numpy_file_name, data_in_1day)

# make_train_sat_dataset関数をそれぞれのサイズに対して実行
make_train_sat_dataset(resize=4, interpolation=cv2.INTER_AREA)
make_train_sat_dataset(resize=6, interpolation=cv2.INTER_AREA)
make_train_sat_dataset(resize=8, interpolation=cv2.INTER_AREA)
make_train_sat_dataset(resize=10, interpolation=cv2.INTER_AREA)
make_train_sat_dataset(resize=12, interpolation=cv2.INTER_AREA)
make_train_sat_dataset(resize=15, interpolation=cv2.INTER_AREA)

# test期間のopen_dateの日付を取得する関数を定義
def get_test_open_date_list():

    terms = pd.read_csv('inference_terms.csv')
    terms = pd.to_datetime(terms.iloc[:, 0])

    open_date_list = []

    for i in range(50):
        
        open_date = terms[i]
        open_date_list.append(open_date)

    return open_date_list



# test_satデータを処理する関数
def make_test_sat_dataset(resize, interpolation):

    open_date_list = get_test_open_date_list()
    
    dir_name = f'resize{resize}/test/sat/'
    os.makedirs(dir_name, exist_ok=True)
    
    count = 0
    
    for open_date in open_date_list:
        
        for i in range(4):
            
            init_date = open_date + timedelta(days=i)
            data_in_1day = []
            
            for j in range(24):
                
                date = init_date + timedelta(hours=j)
                image_file_name = "test/sat/{year}-{month:02}-{day:02}/{year}-{month:02}-{day:02}-{hour:02}-00.fv.png".\
                            format(year=date.year, month=date.month, day=date.day, hour=date.hour)

                img = cv2.imread(image_file_name, 0)
                resized_img = cv2.resize(img, (int(512 / resize), int(672 / resize)),
                                         interpolation=interpolation)

                data_in_1day.append(resized_img)
            
            data_in_1day = np.array(data_in_1day, dtype='uint8').reshape(1, 24, int(672 / resize), int(512 / resize), 1)
            
            zfilled_count = str(count).zfill(3)
            
            numpy_file_name = f'resize{resize}/test/sat/test_sat_rs{resize}_{zfilled_count}.npy'
            np.save(numpy_file_name, data_in_1day)
            
            count += 1

# make_test_sat_dataset関数をそれぞれのサイズに対して実行
make_test_sat_dataset(resize=4, interpolation=cv2.INTER_AREA)
make_test_sat_dataset(resize=6, interpolation=cv2.INTER_AREA)
make_test_sat_dataset(resize=8, interpolation=cv2.INTER_AREA)
make_test_sat_dataset(resize=10, interpolation=cv2.INTER_AREA)
make_test_sat_dataset(resize=12, interpolation=cv2.INTER_AREA)
make_test_sat_dataset(resize=15, interpolation=cv2.INTER_AREA)



'''
ここからは気象データに対する処理を行う
'''

# train期間の日付を取得する関数を定義
def get_train_days_list():
    
    start_date = dt(2016, 1, 1, 16, 0, 0)
    train_days_list = []
    
    for i in range(730):
        
        day = start_date + timedelta(days=i)
        
        train_days_list.append(day)
        
    return train_days_list

# test期間の日付を取得する関数を定義
def get_test_days_list():
    
    open_date_list = get_test_open_date_list()

    test_days_list = []
    
    for open_start_day in open_date_list:
        
        for i in range(4):
            
            date = open_start_day + timedelta(days=i)
            test_days_list.append(date)
            
    return test_days_list

'''
バイナリデータを扱う関数を定義する。

こちらのコードはチュートリアルからそのままの形で利用させてもらいました。
'''

# バイナリデータを読み込む関数
def Read_gz_Binary(file):
    
    file_tmp = file + '_tmp'
    
    with gzip.open(file, 'rb') as f_in:
        
        with open(file_tmp, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            
    bin_data = np.fromfile(file_tmp, np.float32)
    os.remove(file_tmp)
    
    return bin_data.reshape([168, 128])

# 欠損部分を埋める関数
def fill_lack_data(data):
    
    data[0:2] = data[2]
    data[154:] = data[153]
    
    data[:, :8] = data[:, 8].reshape(-1, 1)
    
    return data


# HPRTデータのtrain期間における最大値、最小値を取得する関数
def get_max_min(data_type):
    
    start_date = dt(2016, 1, 1, 0, 0, 0)
    data_list = []
    
    # train期間のバイナリデータをロードする
    for i in range(731*8):
        
        date = start_date + timedelta(hours=(3*i))
        file_name = 'train/met/{year}/{month:02}/{day:02}/{data_type}.3.{year}{month:02}{day:02}{hour:02}.gz'\
        .format(year=date.year, month=date.month, day=date.day, hour=date.hour, data_type=data_type)
        
        met_data = Read_gz_Binary(file_name)
        met_data = fill_lack_data(met_data)
        
        data_list.append(met_data)
        
    np_list = np.array(data_list, dtype='float32')
    
    max_data = np_list.max()
    min_data = np_list.min()
    
    return max_data, min_data

# get_max_min関数を実行し、HPRTデータの最大値、最小値を取得
# 処理に10分ほど時間がかかります
HPRT_data_type_list = ['HGT.200', 'HGT.300', 'HGT.500', 'HGT.700', 'HGT.850', 
                       'PRMSL.msl', 
                       'RH.1p5m', 'RH.300', 'RH.500', 'RH.700', 'RH.850', 
                       'TMP.1p5m', 'TMP.200', 'TMP.300', 'TMP.500', 'TMP.700', 'TMP.850']

HPRT_max_min = {}

for data_type in HPRT_data_type_list:
    HPRT_max_min[data_type] = get_max_min(data_type)

# HPRTデータを読み込みnumpyデータとして保存する関数
def make_met_HPRT_dataset(target, resize, interpolation):
    
    dir_name = f'resize{resize}/{target}/met_HPRT/'
    os.makedirs(dir_name, exist_ok=True)
    
    if target == 'train':
        target_days = get_train_days_list()
        
    elif target == 'test':
        target_days = get_test_days_list()
        
    count = 0
    
    for target_day in target_days:
        
        
        all_type_in_1day = []
    
        for data_type in HPRT_data_type_list:

            one_type_in_1day = []

            for i in range(24):

                date = target_day + timedelta(hours=i)
                
                file_name = '{target}/met/{year}/{month:02}/{day:02}/{data_type}.3.{year}{month:02}{day:02}{hour:02}.gz'\
                .format(target=target, year=date.year, month=date.month, day=date.day, hour=date.hour, data_type=data_type)

                # 気象データは３時間毎のデータしか存在しないので、欠損部分を埋める必要がある。
                # ３時間を一定の変化量で推移したと仮定して穴埋めを行う。
                # 時刻が[0, 3, 6, 9, 12, 15, 18, 21]時の場合
                if date.hour % 3 == 0:

                    bin_data = Read_gz_Binary(file_name)
                    bin_data = fill_lack_data(bin_data)

                # 時刻が[1, 4, 7, 10, 13, 16, 19, 22]時の場合
                elif date.hour % 3 == 1:

                    before_1h = date - timedelta(hours=1)
                    file_before_1h = '{target}/met/{year}/{month:02}/{day:02}/{data_type}.3.{year}{month:02}{day:02}{hour:02}.gz'\
                    .format(target=target, year=before_1h.year, month=before_1h.month, day=before_1h.day, hour=before_1h.hour, data_type=data_type)

                    after_2h = date + timedelta(hours=2)
                    file_after_2h = '{target}/met/{year}/{month:02}/{day:02}/{data_type}.3.{year}{month:02}{day:02}{hour:02}.gz'\
                    .format(target=target, year=after_2h.year, month=after_2h.month, day=after_2h.day, hour=after_2h.hour, data_type=data_type)


                    data_before_1h = fill_lack_data(Read_gz_Binary(file_before_1h))
                    data_after_2h = fill_lack_data(Read_gz_Binary(file_after_2h))

                    bin_data = (2/3) * data_before_1h + (1/3) * data_after_2h

                # 時刻が[2, 5, 8, 11, 14, 17, 20, 23]時の場合
                else:

                    before_2h = date - timedelta(hours=2)
                    file_before_2h = '{target}/met/{year}/{month:02}/{day:02}/{data_type}.3.{year}{month:02}{day:02}{hour:02}.gz'\
                    .format(target=target, year=before_2h.year, month=before_2h.month, day=before_2h.day, hour=before_2h.hour, data_type=data_type)

                    after_1h = date + timedelta(hours=1)
                    file_after_1h = '{target}/met/{year}/{month:02}/{day:02}/{data_type}.3.{year}{month:02}{day:02}{hour:02}.gz'\
                    .format(target=target, year=after_1h.year, month=after_1h.month, day=after_1h.day, hour=after_1h.hour, data_type=data_type)


                    data_before_2h = fill_lack_data(Read_gz_Binary(file_before_2h))
                    data_after_1h = fill_lack_data(Read_gz_Binary(file_after_1h))

                    bin_data = (1/3) * data_before_2h + (2/3) * data_after_1h
                    
               
                    
                # resize=4の場合、リサイズは必要ない
                if resize == 4:
                    
                    resized_data = bin_data
                    
                else:
                    
                    resized_data = cv2.resize(bin_data, (int(512 / resize), int(672 / resize)),
                                              interpolation=interpolation)
                    
                one_type_in_1day.append(resized_data)
                     
            one_type_in_1day = np.array(one_type_in_1day).reshape(1, 24, int(672 / resize), int(512 / resize), 1)
            
            # データを最小値0, 最大値0の間に正規化する
            max_value, min_value = HPRT_max_min[data_type]
            one_type_in_1day = (one_type_in_1day - min_value) / (max_value - min_value).astype('float32')
            
            if data_type == 'HGT.200':
                
                all_type_in_1day = one_type_in_1day
            
            else:
                
                all_type_in_1day = np.concatenate([all_type_in_1day, one_type_in_1day], axis=4)
        
        zfilled_count = str(count).zfill(3)
        
        numpy_file_name = f'resize{resize}/{target}/met_HPRT/{target}_met_HPRT_rs{resize}_f32_{zfilled_count}.npy'
        np.save(numpy_file_name, all_type_in_1day)
        
        count += 1

# make_met_HPRT_dataset関数をtrainデータに対して実行
make_met_HPRT_dataset(target='train', resize=4, interpolation=cv2.INTER_AREA)
make_met_HPRT_dataset(target='train', resize=6, interpolation=cv2.INTER_AREA)
make_met_HPRT_dataset(target='train', resize=8, interpolation=cv2.INTER_AREA)
make_met_HPRT_dataset(target='train', resize=10, interpolation=cv2.INTER_AREA)
make_met_HPRT_dataset(target='train', resize=12, interpolation=cv2.INTER_AREA)
make_met_HPRT_dataset(target='train', resize=15, interpolation=cv2.INTER_AREA)

# make_met_HPRT_dataset関数をtestデータに対して実行
make_met_HPRT_dataset(target='test', resize=4, interpolation=cv2.INTER_AREA)
make_met_HPRT_dataset(target='test', resize=6, interpolation=cv2.INTER_AREA)
make_met_HPRT_dataset(target='test', resize=8, interpolation=cv2.INTER_AREA)
make_met_HPRT_dataset(target='test', resize=10, interpolation=cv2.INTER_AREA)
make_met_HPRT_dataset(target='test', resize=12, interpolation=cv2.INTER_AREA)
make_met_HPRT_dataset(target='test', resize=15, interpolation=cv2.INTER_AREA)


'''
met_Windデータに対する処理
'''

# 正規化のため、最大値を取得する。最小値は0になるため省略
def get_pos_neg_max(data_type):
    
    start_date = dt(2016, 1, 1, 0, 0, 0)
    data_list = []
    
    # 学習データ期間のバイナリデータをロード
    for i in range(731*8):
        
        date = start_date + timedelta(hours=(3*i))
        file_name = 'train/met/{year}/{month:02}/{day:02}/{data_type}.3.{year}{month:02}{day:02}{hour:02}.gz'\
        .format(year=date.year, month=date.month, day=date.day, hour=date.hour, data_type=data_type)
        
        met_data = Read_gz_Binary(file_name)
        met_data = fill_lack_data(met_data)
        
        data_list.append(met_data)
        
    np_list = np.array(data_list, dtype='float32')
    
    pos_max = np_list.max()
    neg_max = (- np_list).max()
    
    return pos_max, neg_max

# 負の数を0にする関数
def negative_to_zero(x):
    
    if x < 0:
        x = 0
        
    return x

# get_pos_neg_max関数を実行し、それぞれのデータの方向成分ごとの最大値を取得
# 処理に10分ほど時間がかかります
Wind_data_type_list = ['UGRD.10m','UGRD.200', 'UGRD.300', 'UGRD.500', 'UGRD.700', 'UGRD.850', 
                       'VGRD.10m', 'VGRD.200', 'VGRD.300', 'VGRD.500', 'VGRD.700', 'VGRD.850', 
                       'VVEL.200', 'VVEL.300', 'VVEL.500', 'VVEL.700', 'VVEL.850']

pos_neg_max = {}

for data_type in Wind_data_type_list:
    pos_neg_max[data_type] = get_pos_neg_max(data_type)

# met_Windデータを読み込み、numpyファイルとして保存する関数
def make_met_Wind_dataset(target, resize, interpolation=None):
    
    dir_name = f'resize{resize}/{target}/met_Wind/'
    os.makedirs(dir_name, exist_ok=True)
    
    if target == 'train':
        target_days = get_train_days_list()
        
    elif target == 'test':
        target_days = get_test_days_list()
        
    count = 0
    
    for target_day in target_days:
        
        all_type_in_1day = []
    
        for data_type in Wind_data_type_list:

            one_type_in_1day = []

            for i in range(24):

                date = target_day + timedelta(hours=i)
                
                file_name = '{target}/met/{year}/{month:02}/{day:02}/{data_type}.3.{year}{month:02}{day:02}{hour:02}.gz'\
                .format(target=target, year=date.year, month=date.month, day=date.day, hour=date.hour, data_type=data_type)

                # 気象データは３時間毎のデータしか存在しないので、欠損部分を埋める必要がある。
                # ３時間を一定の変化量で推移したと仮定して穴埋めを行う。
                # 時刻が[0, 3, 6, 9, 12, 15, 18, 21]時の場合
                if date.hour % 3 == 0:

                    bin_data = Read_gz_Binary(file_name)
                    bin_data = fill_lack_data(bin_data)

                # 時刻が[1, 4, 7, 10, 13, 16, 19, 22]時の場合
                elif date.hour % 3 == 1:

                    before_1h = date - timedelta(hours=1)
                    file_before_1h = '{target}/met/{year}/{month:02}/{day:02}/{data_type}.3.{year}{month:02}{day:02}{hour:02}.gz'\
                    .format(target=target, year=before_1h.year, month=before_1h.month, day=before_1h.day, hour=before_1h.hour, data_type=data_type)

                    after_2h = date + timedelta(hours=2)
                    file_after_2h = '{target}/met/{year}/{month:02}/{day:02}/{data_type}.3.{year}{month:02}{day:02}{hour:02}.gz'\
                    .format(target=target, year=after_2h.year, month=after_2h.month, day=after_2h.day, hour=after_2h.hour, data_type=data_type)


                    data_before_1h = fill_lack_data(Read_gz_Binary(file_before_1h))
                    data_after_2h = fill_lack_data(Read_gz_Binary(file_after_2h))

                    bin_data = (2/3) * data_before_1h + (1/3) * data_after_2h

                # 時刻が[2, 5, 8, 11, 14, 17, 20, 23]時の場合
                else:

                    before_2h = date - timedelta(hours=2)
                    file_before_2h = '{target}/met/{year}/{month:02}/{day:02}/{data_type}.3.{year}{month:02}{day:02}{hour:02}.gz'\
                    .format(target=target, year=before_2h.year, month=before_2h.month, day=before_2h.day, hour=before_2h.hour, data_type=data_type)

                    after_1h = date + timedelta(hours=1)
                    file_after_1h = '{target}/met/{year}/{month:02}/{day:02}/{data_type}.3.{year}{month:02}{day:02}{hour:02}.gz'\
                    .format(target=target, year=after_1h.year, month=after_1h.month, day=after_1h.day, hour=after_1h.hour, data_type=data_type)


                    data_before_2h = fill_lack_data(Read_gz_Binary(file_before_2h))
                    data_after_1h = fill_lack_data(Read_gz_Binary(file_after_1h))

                    bin_data = (1/3) * data_before_2h + (2/3) * data_after_1h
                    
               
                    
                # resize=4の場合、リサイズは必要ない。
                if resize == 4:
                    
                    resized_data = bin_data
                    
                else:
                    
                    resized_data = cv2.resize(bin_data, (int(512 / resize), int(672 / resize)),
                                              interpolation=interpolation)
                    
                one_type_in_1day.append(resized_data)
            
            one_type_in_1day = np.array(one_type_in_1day, dtype='float32').reshape(1, 24, int(672 / resize), int(512 / resize), 1)
            
            # 風の成分を正負で分割する
            pos = one_type_in_1day
            pos = np.vectorize(negative_to_zero)(pos)
            
            neg = (- one_type_in_1day)
            neg = np.vectorize(negative_to_zero)(neg)
            
            pos_max, neg_max = pos_neg_max[data_type]
            
            # 正規化: 最小値がもともと0なので、最大値で割ると、すべての値が[0, 1]の間に収まる。
            pos = pos / pos_max
            neg = neg / neg_max
            
            # posとnegをチャンネルの次元でまとめる。
            one_type_in_1day = np.concatenate([pos, neg], axis=4).astype('float32')
            
            # 最初のデータと２個目以降のデータで処理を分ける。
            if data_type == 'UGRD.10m':
                
                all_type_in_1day = one_type_in_1day
            
            else:
                
                all_type_in_1day = np.concatenate([all_type_in_1day, one_type_in_1day], axis=4)
        
        zfilled_count = str(count).zfill(3)
        
        numpy_file_name = f'resize{resize}/{target}/met_Wind/{target}_met_Wind_rs{resize}_f32_{zfilled_count}.npy'
        np.save(numpy_file_name, all_type_in_1day)
        
        count += 1

# make_met_Wind_dataset関数をtrainデータに対して実行
make_met_Wind_dataset(target='train', resize=4, interpolation=cv2.INTER_AREA)
make_met_Wind_dataset(target='train', resize=6, interpolation=cv2.INTER_AREA)
make_met_Wind_dataset(target='train', resize=8, interpolation=cv2.INTER_AREA)
make_met_Wind_dataset(target='train', resize=10, interpolation=cv2.INTER_AREA)
make_met_Wind_dataset(target='train', resize=12, interpolation=cv2.INTER_AREA)
make_met_Wind_dataset(target='train', resize=15, interpolation=cv2.INTER_AREA)

# make_met_Wind_dataset関数をtestデータに対して実行

make_met_Wind_dataset(target='test', resize=4, interpolation=cv2.INTER_AREA)
make_met_Wind_dataset(target='test', resize=6, interpolation=cv2.INTER_AREA)
make_met_Wind_dataset(target='test', resize=8, interpolation=cv2.INTER_AREA)
make_met_Wind_dataset(target='test', resize=10, interpolation=cv2.INTER_AREA)
make_met_Wind_dataset(target='test', resize=12, interpolation=cv2.INTER_AREA)
make_met_Wind_dataset(target='test', resize=15, interpolation=cv2.INTER_AREA)


'''
ここからは、予測時に利用する入力データ(X_test)を、使いやすい状態で
一つのフォルダにまとめる処理を行う。

すでにtestフォルダは作成し、必要なファイルは全てそこに揃っている。
容量は無駄遣いすることになってしまうが、利便性のためにまとめることにする。
'''

# 予測時の入力データ(X_test)を作成してフォルダにまとめる関数
def make_pre24_X_test_file(resize):
    
    os.makedirs('X_test_folder', exist_ok=True)
    
    X_test_sat_dir = f'resize{resize}/test/sat/'
    X_test_met_HPRT_dir = f'resize{resize}/test/met_HPRT/'
    X_test_met_Wind_dir = f'resize{resize}/test/met_Wind/'
    
    X_test_sat_file_list = listdir(X_test_sat_dir)
    X_test_met_HPRT_file_list = listdir(X_test_met_HPRT_dir)
    X_test_met_Wind_file_list = listdir(X_test_met_Wind_dir)
    
    X_test_sat_file_list.sort()
    X_test_met_HPRT_file_list.sort()
    X_test_met_Wind_file_list.sort()
    
    X_test_list = []
    
    for i in range(50):
                
        X_test_sat = np.load(X_test_sat_dir + X_test_sat_file_list[(4*i)+3]).astype('float32') / 255
        X_test_met_HPRT = np.load(X_test_met_HPRT_dir + X_test_met_HPRT_file_list[(4*i)+3])
        X_test_met_Wind = np.load(X_test_met_Wind_dir + X_test_met_Wind_file_list[(4*i)+3])
        
        X_test = np.concatenate([X_test_sat, X_test_met_HPRT, X_test_met_Wind], axis=4)
        
        X_test_list.append(X_test)
        
    X_test_list = np.array(X_test_list, dtype='float32').reshape((50, 24, int(672 / resize), int(512 / resize), 52))
    
    save_file_name = f'X_test_folder/X_test_resize{resize}_float32.npy'
    np.save(save_file_name, X_test_list)

# make_pre24_X_test_fileをそれぞれのサイズに対して実行
make_pre24_X_test_file(resize=4)
make_pre24_X_test_file(resize=6)
make_pre24_X_test_file(resize=8)
make_pre24_X_test_file(resize=10)
make_pre24_X_test_file(resize=12)
make_pre24_X_test_file(resize=15)

'''
モデルの学習をどのように行われるのか
(検証データを使用し、Scoreをモニタリングして
最適な状態のモデルを採用するのか、それとも検証データを使用せずに、
こちらが指定したエポック数だけ繰り返し学習したモデルを採用するのか)
わからないのですが、ここでは念の為、検証データを使用する場合を想定して
検証データを作成しておきます。

必要がなければ修正するので、指摘してください。

ここでは、とりあえず2018年のデータを検証データとして作成しますが、
2015年以前のデータがあるのであれば、汎化性能まで正確に測ることを考慮すると、
そちらを使用するほうがいいかもしれません。

'''

# testデータをvalidフォルダにコピー
def make_pre24_valid_dataset(resize):

    shutil.copytree(f'resize{resize}/test/sat',
                    f'resize{resize}/valid/X_sat')
    shutil.copytree(f'resize{resize}/test/sat',
                    f'resize{resize}/valid/Y')
    shutil.copytree(f'resize{resize}/test/met_HPRT',
                    f'resize{resize}/valid/X_met_HPRT')
    shutil.copytree(f'resize{resize}/test/met_Wind',
                    f'resize{resize}/valid/X_met_Wind')

# make_pre24_valid_dataset関数の実行
make_pre24_valid_dataset(resize=4)
make_pre24_valid_dataset(resize=6)
make_pre24_valid_dataset(resize=8)
make_pre24_valid_dataset(resize=10)
make_pre24_valid_dataset(resize=12)
make_pre24_valid_dataset(resize=15)

'''
validデータから不要なデータを削除し、
同じインデックスにおけるXとYの関係が
入力データと正解データとして成立するようにする
'''

# 不必要な検証データの入力データを削除する関数
def remove_unnecessary_X_valid(resize):

    X_sat_dir = f'resize{resize}/valid/X_sat/'
    X_met_HPRT_dir = f'resize{resize}/valid/X_met_HPRT/'
    X_met_Wind_dir = f'resize{resize}/valid/X_met_Wind/'

    X_sat_file_list = listdir(X_sat_dir)
    X_met_HPRT_file_list = listdir(X_met_HPRT_dir)
    X_met_Wind_file_list = listdir(X_met_Wind_dir)

    X_sat_file_list.sort()
    X_met_HPRT_file_list.sort()
    X_met_Wind_file_list.sort()

    for i, file_name in enumerate(X_sat_file_list):

        if (i % 4) == 3:
            os.remove(X_sat_dir + file_name)

    for i, file_name in enumerate(X_met_HPRT_file_list):

        if (i % 4) == 3:
            os.remove(X_met_HPRT_dir + file_name)

    for i, file_name in enumerate(X_met_Wind_file_list):

        if (i % 4) == 3:
            os.remove(X_met_Wind_dir + file_name)

# remove_unnecessary_X_valid関数を実行
remove_unnecessary_X_valid(resize=4)
remove_unnecessary_X_valid(resize=6)
remove_unnecessary_X_valid(resize=8)
remove_unnecessary_X_valid(resize=10)
remove_unnecessary_X_valid(resize=12)
remove_unnecessary_X_valid(resize=15)

# 不必要な検証データの正解データを削除する関数
def remove_unnecessary_Y_valid(resize):
    
    Y_dir = f'resize{resize}/valid/Y/'

    Y_file_list = listdir(Y_dir)

    Y_file_list.sort()

    for i, file_name in enumerate(Y_file_list):

        if (i % 4 == 0):
            os.remove(Y_dir + file_name)

# remove_unnecessary_Y_valid関数を実行
remove_unnecessary_Y_valid(resize=4)
remove_unnecessary_Y_valid(resize=6)
remove_unnecessary_Y_valid(resize=8)
remove_unnecessary_Y_valid(resize=10)
remove_unnecessary_Y_valid(resize=12)
remove_unnecessary_Y_valid(resize=15)