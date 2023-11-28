import copy
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
from skmultiflow.data import FileStream
from skmultiflow.trees import HoeffdingTreeClassifier 
import evaluation_online
import csv
import scipy.io as sio

class indi:
    def __init__(self):
        self.x = []
        self.fitx=0.0
        self.clf=HoeffdingTreeClassifier()
majority=1800
minority=200
POPSIZE=3 
DIMENSION=2
UPPERBOUND=10
LOWERBOUND=0
ε=0.000000000000001
Pm=0.02       #Mutation probability
MAXITERA=100 # Number of iterations
start_point=0
individual = []
data = np.arange(0,POPSIZE,1)
data=data.tolist()

BEST=indi()
X_train=[] 
X_test=[] 
y_train=[] 
y_test=[] 
alpha=1.00
buffer_len=100
np.random.seed(1234)
bounds = (0, 1)
pretrain = 100
sample_x=np.zeros(buffer_len)
sample_y=np.zeros(buffer_len)
def initialize1():
    for i in range(POPSIZE):
        individual[i].x[0]=1
        individual[i].x[1]=1
        individual[i].clf = HoeffdingTreeClassifier()

def clip_solution(solution, bounds):
    return np.clip(solution, bounds[0], bounds[1])

def objective_function(x,IR1,test_clf):
#     test_clf = HoeffdingTreeClassifier()
#     test_clf.partial_fit(X_train, y_train, classes=[0, 1]) 
    test_clf=copy.deepcopy(test_clf)
    instance_x=sample_x
    instance_y=sample_y
    S = np.zeros([class_num])
    N = np.zeros([class_num])
    cf = np.zeros([class_num, class_num])
    recall = np.zeros([buffer_len, class_num])
    gmean = np.zeros([buffer_len])
    for j in range(buffer_len):
                y_pred_test = test_clf.predict(instance_x[j].reshape(1,-1))
                y_pred_ptest= test_clf.predict_proba(instance_x[j].reshape(1,-1))
                test_label = instance_y[j]
                test_label = np.expand_dims(test_label, 0)
                test_pre=y_pred_test
                recall[j, :], gmean[j], S, N = evaluation_online.pf_online(S, N,test_label,test_pre)
                cf = evaluation_online.confusion_online(cf, test_label, test_pre)
                class_dependent_cost=1/(IR1[instance_y[j]]/(buffer_len))
                p_t=y_pred_ptest[0][instance_y[j]]
                FC=class_dependent_cost*(-np.abs(p_t-0.5))*np.log(np.abs(p_t-0.5))*(1-p_t)**x
                test_clf.partial_fit(instance_x[j].reshape(1, -1), [instance_y[j]], classes=[0, 1],sample_weight=[FC])
                
    return gmean[-1]

def differential_evolution(objective_function, bounds, IR1, test_clf, population_size=10, max_generations=2, mutation_factor=0.5, crossover_probability=0.7):
    best_solution=0
    best_fitness=0
    population = np.random.uniform(bounds[0], bounds[1], (population_size, 1))
    for generation in range(max_generations):

        for i in range(population_size):
            indices = [index for index in range(population_size) if index != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = population[i] + mutation_factor * (a - b)
            mutant = clip_solution(mutant, bounds)
            crossover = np.random.rand()
            if crossover <= crossover_probability:
                trial = mutant
            else:
                trial = population[i]
            a=objective_function(trial,IR1,test_clf)
            b=objective_function(population[i],IR1,test_clf)
            if  a>b :
                population[i] = trial
                if a>best_fitness:
                      best_fitness=a
                      best_solution=trial[0]
            else:
                if b>best_fitness:
                      best_fitness=b
                      best_solution=population[i][0]
            print(generation, i, best_solution, best_fitness)

    return best_solution, best_fitness

if __name__=='__main__':

        X=[]
        y=[]
        '''
        ./imbalance_dataset/synthesize/{dataset}.csv
        '''
        dataset="synthesize7"
        stream = FileStream(f'imbalance_dataset/synthesize/{dataset}.csv')
        with open(f'imbalance_dataset/synthesize/{dataset}.csv', 'r') as file:
                reader = csv.reader(file)
                line_count = len(list(reader))
        for i in range(0, line_count-1):
                feature, label = stream.next_sample()
                X.append(feature[0])
                y.append(int(label[0]))
        X=np.array(X)
        y=np.array(y)
        '''
        ./imbalance_dataset/chess/data.mat
        '''

        # file_name = './imbalance_dataset/chess/data.mat'
        # data = sio.loadmat(file_name)
        # X = data['X']
        # X = X[:, :-1]
        # y = data['y']
        # X = X.astype(np.double)
        # y = y.astype(np.int32)
        # X = np.squeeze(X)
        # y = np.squeeze(y)
        # line_count = len(X)
        
        # print(len(X), len(y))

        class_num = len(np.unique(y))
        class_size = np.zeros(class_num)
        S = np.zeros([class_num])
        N = np.zeros([class_num])
        cf = np.zeros([class_num, class_num])
        recall = np.zeros([buffer_len, class_num])
        gmean = np.zeros([buffer_len])

 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-pretrain/(line_count-1), random_state=42)
        for i in range(class_num):
            class_size[i] = len(np.argwhere(y_train == i))
        nb_all=len(y)
        test_len=X_test.shape[0]
        class_num=len(np.unique(y))

        individual.clear()
        for i in range(POPSIZE+1):
                a=indi()
                a.x=np.zeros((DIMENSION,))
                individual.append(a)
        initialize1()

        S1 = np.zeros([class_num])
        N1 = np.zeros([class_num])
        cf1 = np.zeros([class_num, class_num])
        recall1 = np.zeros([test_len, class_num])
        gmean1 = np.zeros([test_len])
        S2 = np.zeros([class_num])
        N2 = np.zeros([class_num])
        cf2 = np.zeros([class_num, class_num])
        recall2 = np.zeros([test_len, class_num])
        gmean2 = np.zeros([test_len])
        S3 = np.zeros([class_num])
        N3 = np.zeros([class_num])
        cf3 = np.zeros([class_num, class_num])
        recall3 = np.zeros([test_len, class_num])
        gmean3 = np.zeros([test_len])

        for i in range(POPSIZE):
             individual[i].clf.partial_fit(X_train, y_train)     
        total0=0
        total1=1
        TP=0
        TN=0
        test_clf=copy.deepcopy(individual[2].clf)
        for j in range(X_test.shape[0]):
                print(j)
                y_pred1 = individual[0].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p1=individual[0].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred2= individual[1].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p2=individual[1].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred3 = individual[2].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p3=individual[2].clf.predict_proba(X_test[j].reshape(1,-1))

                test_label = y_test[j]
                test_label = np.expand_dims(test_label, 0)
                test_pre1=y_pred1
                test_pre2=y_pred2
                test_pre3=y_pred3
                recall1[j, :], gmean1[j], S1, N1 = evaluation_online.pf_online(S1, N1,test_label,test_pre1)
                cf1 = evaluation_online.confusion_online(cf1, test_label, test_pre1)
                recall2[j, :], gmean2[j], S2, N2 = evaluation_online.pf_online(S2, N2,test_label,test_pre2)
                cf2 = evaluation_online.confusion_online(cf2, test_label, test_pre2)
                recall3[j, :], gmean3[j], S3, N3 = evaluation_online.pf_online(S3, N3,test_label,test_pre3)
                cf3 = evaluation_online.confusion_online(cf3, test_label, test_pre3)

                class_size[y_test[j]]+=1
                #class_dependent_cost=IR[1-y_test[j]]/(j+1)
                class_dependent_cost=1/(class_size[y_test[j]]/(len(y_train)+j+1))
                individual[0].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]])
                individual[1].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]], sample_weight=[class_dependent_cost])
                p_t=y_pred_p3[0][y_test[j]]
                #FC=1/(IR[y_test[j]]/(j+1))*(-((np.abs(p_t-0.5))**alpha)*np.log(np.abs(p_t-0.5)))
                #FC=class_dependent_cost*(-((np.abs(p_t-0.5))**alpha)*np.log(np.abs(p_t-0.5)))*(1-p_t)
                #FC=class_dependent_cost*(-2*(np.abs(p_t-0.5))*np.log(2*np.abs(p_t-0.5)))
                #*( -(np.abs(p_t-alpha))*np.log(np.abs(p_t-alpha))-(np.abs((1-p_t)-alpha))*np.log(np.abs((1-p_t)-alpha)))
                #alpha=0.00001
                
                # FC=class_dependent_cost*(-np.abs(p_t-0.5))*np.log(np.abs(p_t-0.5))*(1-p_t)**alpha
                FC=class_dependent_cost*(-np.abs(p_t-0.5))*np.log(np.abs(p_t-0.5))*(1-p_t)**0.75

                # FC=0.5*class_dependent_cost*(-(np.abs(p_t-alpha))*np.log(np.abs(p_t-alpha))-(np.abs((1-p_t)-alpha))*np.log(np.abs((1-p_t)-alpha)))
                individual[2].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]], sample_weight=[FC])
                # #use DE
                # if ((j+1)%buffer_len==0):
                #         sample_x = X_test[j+1-buffer_len:j+1]
                #         sample_y = y_test[j+1-buffer_len:j+1]
                #         # IR1={0:buffer_len - np.count_nonzero(sample_y),1:np.count_nonzero(sample_y)}
                #         IR1={0:IR[0]/((j+1)/buffer_len),1:IR[1]/((j+1)/buffer_len)}
                #         alpha,_=differential_evolution(objective_function, bounds, IR1, copy.deepcopy(test_clf))
                #         test_clf=copy.deepcopy(individual[2].clf)
                  
        ncol = 1

        fontsize = 15
        legendsize = 13
        limsize = 10
        

        figure1 = plt.figure()
        ax1 = plt.axes()
        plt.xlim(1, nb_all)
        plt.ylim(0, 1)
        plt.ylabel('G-mean', fontsize=fontsize)
        plt.xlabel('Time step', fontsize=fontsize)
        plt.tick_params(labelsize=limsize)
        plt.grid()
        #plt.title(f'Gmean_score over {dataset}.csv in online imbalance learning')


        print(gmean1[-1])
        print(gmean2[-1])
        print(gmean3[-1])
        plot_x = range(nb_all - pretrain)
        plot_y1 = gmean1 
        plot_y2 = gmean2
        plot_y3 = gmean3
        ax1.plot(plot_x, plot_y1, color='blue',label='no processing')
        ax1.plot(plot_x, plot_y2, color='green',label='class-dependent cost')
        ax1.plot(plot_x, plot_y3, color='red',label=f'instance-dependent cost, alpha={alpha}')
        
        ax1.legend(fontsize=legendsize, ncol=ncol) 
        #plt.savefig(f'results/synthesize/{dataset}_instance_dependent_focal_cost_alpha={alpha}.png')
        plt.show()
       