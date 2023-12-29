import copy
import random
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
np.float = float
from skmultiflow.trees import HoeffdingTreeClassifier 
import evaluation_online
import read_data

class indi:
    def __init__(self):
        self.x = []
        self.fitx=0.0
        self.clf=HoeffdingTreeClassifier()

threshold=[0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8]
alpha=[0.125, 0.25, 0.5, 1, 2, 4, 8]
n_base_classifier=len(threshold)*len(alpha)
bias=2
POPSIZE=bias+n_base_classifier
DIMENSION=2
individual = []

X_train=[] 
X_test=[] 
y_train=[] 
y_test=[] 
buffer_len=100
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.05            # Minimum exploration probability 
decay_rate = 0.0005            # Exponential decay rate for exploration prob
# np.random.seed(1234)
# random.seed(1234)
pretrain = 1

def initialize1():
    for i in range(POPSIZE):
        individual[i].x[0]=1
        individual[i].x[1]=1
        individual[i].clf = HoeffdingTreeClassifier()

def initialize_metric(n_classifier,class_num, test_len):             
        S = np.zeros((n_classifier, class_num))
        N = np.zeros((n_classifier,class_num))
        cf = np.zeros((n_classifier, class_num, class_num))
        recall = np.zeros((n_classifier,test_len, class_num))
        gmean = np.zeros((n_classifier,test_len))

        return S, N, cf, recall, gmean

if __name__=='__main__':
        
        datasets=['synthesize','chess','yeast','segment']
        test_id=0
        dataset=datasets[test_id]+'4'
        X,y=read_data.read(datasets[test_id],dataset)

        line_count = len(X) 
        class_num = len(np.unique(y))
        class_size = np.zeros(class_num)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-pretrain/(line_count-1), random_state=42)
        for i in range(class_num):
            class_size[i] = len(np.argwhere(y_train == i))
        nb_all=len(y)

        _size = np.zeros(class_num)
        for i in range(class_num):
            _size[i] = len(np.argwhere(y == i))   
        # print(_size,_size[0]/_size[1],_size[1]/_size[0],line_count,X_train.shape[1],class_num)
        test_len=X_test.shape[0]
        class_num=len(np.unique(y))

        individual.clear()
        for i in range(POPSIZE+1):
                a=indi()
                a.x=np.zeros((DIMENSION,))
                individual.append(a)
        initialize1()
    
        S, N, cf, recall, gmean = initialize_metric(1,class_num,test_len)
        S0, N0, cf0, recall0, gmean0 = initialize_metric(bias,class_num,test_len)
        S1, N1, cf1, recall1, gmean1 = initialize_metric(n_base_classifier,class_num,test_len)

        for i in range(POPSIZE):
             individual[i].clf.partial_fit(X_train, y_train,classes=list(range(class_num)))     
        total0=0
        total1=0
        TP=0
        TN=0
        pretrain = 0
        for j in range(X_test.shape[0]):
                #print(j)
                # if (j==0 or j==pretrain):
                #              S, N, cf, recall, gmean = initialize_metric(1,class_num,test_len)
                #              S0, N0, cf0, recall0, gmean0 = initialize_metric(bias,class_num,test_len)
                #              S1, N1, cf1, recall1, gmean1 = initialize_metric(n_base_classifier,class_num,test_len)
                y_pred=[]
                y_pred_proba=[]
                for idx in range(POPSIZE):
                    current_individual = individual[idx]
                    y_pred.append(current_individual.clf.predict(X_test[j].reshape(1, -1)))
                    y_pred_proba.append(current_individual.clf.predict_proba(X_test[j].reshape(1, -1)))
                test_label = y_test[j]
                test_label = np.expand_dims(test_label, 0)
                
                epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*j)
                random_int = random.uniform(0,1)
                # if random_int > greater than epsilon -->  elite strategy
                if random_int > epsilon:
                        top_indices = np.argsort(gmean1[:, j-1])[-len(threshold):][::-1]
                        # max_index = np.argmax(gmean1[:, j-1])
                        # t0= y_pred_proba[max_index+bias][0][0] 
                        # t1= y_pred_proba[max_index+bias][0][1]
                        weight_sum = sum(gmean1[k][j-1] for k in top_indices)
                        t0 = sum(y_pred_proba[k+bias][0][0] * (gmean1[k][j-1] + 1/len(threshold)) / (weight_sum + 1) for k in top_indices) 
                        t1 = sum(y_pred_proba[k+bias][0][1] * (gmean1[k][j-1] + 1/len(threshold)) / (weight_sum + 1) for k in top_indices)
                # else --> mass strategy
                else:
                        weight_sum = sum(gmean1[k][j-1] for k in range(n_base_classifier))
                        weight = [(gmean1[k][j-1] + 1/n_base_classifier) / (weight_sum + 1) for k in range(n_base_classifier)]
                        t0 = sum(y_pred_proba[k][0][0] * weight[k-bias] for k in range(bias, bias + n_base_classifier)) 
                        t1 = sum(y_pred_proba[k][0][1] * weight[k-bias] for k in range(bias, bias + n_base_classifier)) 

                y_pred_ensemble = 1 if t1 > t0 else 0

                print(j,t0,t1,y_pred_ensemble,y_test[j])
                #print(j)

                recall[0][j, :], gmean[0][j], S[0], N[0] = evaluation_online.pf_online(S[0], N[0],test_label,y_pred_ensemble)
                cf[0] = evaluation_online.confusion_online(cf[0], test_label, y_pred_ensemble)

                for k in range(bias):
                    recall0[k][j, :], gmean0[k][j], S0[k], N0[k] = evaluation_online.pf_online(S0[k], N0[k],test_label,y_pred[k])
                    cf0[k] = evaluation_online.confusion_online(cf0[k], test_label, y_pred[k])     
                for k in range(n_base_classifier):
                    recall1[k][j, :], gmean1[k][j], S1[k], N1[k] = evaluation_online.pf_online(S1[k], N1[k],test_label,y_pred[k+bias])
                    cf1[k] = evaluation_online.confusion_online(cf1[k], test_label, y_pred[k+bias])     
                
                class_size[y_test[j]]+=1

                class_dependent_cost=1/(class_size[y_test[j]]/(len(y_train)+j+1))

                individual[0].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)))
                individual[1].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[class_dependent_cost])
                
                FC = [class_dependent_cost * (threshold[k] + (1 - y_pred_proba[bias + k * len(alpha) + l][0][y_test[j]])**alpha[l])
                    for k in range(len(threshold)) for l in range(len(alpha))]

                _ = [individual[k].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]], classes=list(range(class_num)), sample_weight=[FC[k - bias]]) for k in range(bias, bias + n_base_classifier)]
   
        ncol = 1

        fontsize = 15
        legendsize = 13
        limsize = 10
        

        figure1 = plt.figure()
        ax1 = plt.axes()
        plt.xlim(0, nb_all-pretrain+1)
        plt.ylim(0, 1)
        plt.ylabel('G-mean', fontsize=fontsize)
        plt.xlabel('Time step', fontsize=fontsize)
        plt.tick_params(labelsize=limsize)
        plt.grid()
        plt.title(f'Gmean_score over {dataset} in online imbalance learning')


        print(gmean0[0][-1])
        print(gmean0[1][-1])
        print(gmean[0][-1])
        plot_x = range(len(gmean[0][pretrain:]))
        #plot_x = range(nb_all - pretrain)
        plot_y1 = gmean0[0][pretrain:]
        plot_y2 = gmean0[1][pretrain:]
        plot_y3 = gmean[0][pretrain:]

        # print(plot_x)
          
        ax1.plot(plot_x, plot_y1, color='blue',label='no processing')
        ax1.plot(plot_x, plot_y2, color='green',label='class-dependent cost')
        ax1.plot(plot_x, plot_y3, color='red',label='ensemble instance-dependent cost')
        
        ax1.legend(fontsize=legendsize, ncol=ncol) 
        #plt.savefig(f'results/result3/{datasets[test_id]}/{dataset}_ensemble_instance_dependent_focal_cost.png')
        plt.show()
       