from sklearn import preprocessing
import numpy as np
from skmultiflow.data import FileStream
import csv
import scipy.io as sio
import warnings


# silence the warning
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
def read(dataset):
        X=[]
        y=[]
        '''
        ./imbalance_dataset/synthesize/{dataset}.csv
        '''
        if dataset[:-1]=='synthesize':
            stream = FileStream(f'imbalance_dataset/synthesize/{dataset}.csv')
            with open(f'imbalance_dataset/synthesize/{dataset}.csv', 'r') as file:
                    reader = csv.reader(file)
                    line_count = len(list(reader))
            for i in range(0, line_count-1):
                    feature, label = stream.next_sample()
                    X.append(feature[0])
                    y.append(int(label[0]))
            X=np.array(X)
            # 使用preprocessing模块中的MinMaxScaler进行归一化
            scaler = preprocessing.MinMaxScaler()
            X= scaler.fit_transform(X)
            y=np.array(y)
            return X,y
        '''
        ./imbalance_dataset/chess/data.mat
        '''
        if dataset =='chess':
            file_name = './imbalance_dataset/chess/data.mat'
            data = sio.loadmat(file_name)
            X = data['X']
            X = X[:, :-1]
            y = data['y']
            X = X.astype(np.double)
            y = y.astype(np.int32)
            X = np.squeeze(X)
            scaler = preprocessing.MinMaxScaler()
            X= scaler.fit_transform(X)
            y = np.squeeze(y)
            return X,y            
        '''
        ./imbalance_dataset/yeast1/yeast1.dat
        '''       
        
        if dataset =='yeast1':
                # 打开数据集文件
                with open('./imbalance_dataset/yeast1/yeast1.dat', 'r') as file:
                    # 逐行读取数据
                    data = file.readlines()



                # 遍历每一行数据
                for line in data:
                    # 去除换行符并按逗号分割数据
                    line = line.strip().split(',')

                    # 提取X，将每个特征转换为浮点数并添加到X列表
                    X.append([float(x) for x in line[:-1]])

                    # 提取y，将标签添加到y列表
                    if line[-1] == ' negative':
                        y.append(0)
                    else:
                        y.append(1)
                        
                X=np.array(X)
                        # 使用preprocessing模块中的MinMaxScaler进行归一化
                scaler = preprocessing.MinMaxScaler()
                X= scaler.fit_transform(X)
                y=np.array(y)
                return X,y
        '''
        ./imbalance_dataset/segment0/segment0.dat
        '''       
        if dataset =='segment0':
                # 打开数据集文件
                with open('./imbalance_dataset/segment0/segment0.dat', 'r') as file:
                    # 逐行读取数据
                    data = file.readlines()



                # 遍历每一行数据
                for line in data:
                    # 去除换行符并按逗号分割数据
                    line = line.strip().split(',')

                    # 提取X，将每个特征转换为浮点数并添加到X列表
                    X.append([float(x) for x in line[:-1]])

                    # 提取y，将标签添加到y列表
                    if line[-1] == ' negative':
                        y.append(0)
                    else:
                        y.append(1)
                        
                X=np.array(X)
                # 使用preprocessing模块中的MinMaxScaler进行归一化
                scaler = preprocessing.MinMaxScaler()
                X= scaler.fit_transform(X)
                y=np.array(y)
                return X,y