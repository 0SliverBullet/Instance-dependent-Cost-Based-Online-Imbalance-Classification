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
class_num=2
alpha=0.05
buffer_len=50
np.random.seed(1234)
bounds = (0, 10)

sample_x=np.zeros(buffer_len)
sample_y=np.zeros(buffer_len)
S = np.zeros([class_num])
N = np.zeros([class_num])
cf = np.zeros([class_num, class_num])
recall = np.zeros([buffer_len, class_num])
gmean = np.zeros([buffer_len])
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
                if (y_pred_test[0]!=instance_y[j]):
                             test_clf.partial_fit(instance_x[j].reshape(1, -1), [instance_y[j]], classes=[0, 1],sample_weight=[(1+x*(1/(instance_y[j]*(1-y_pred_ptest[0][1])+(1-instance_y[j])*(y_pred_ptest[0][1]))-1))*1/(IR1[instance_y[j]]/(buffer_len))])
                else: 
                             test_clf.partial_fit(instance_x[j].reshape(1, -1), [instance_y[j]], classes=[0, 1],sample_weight=[(1+x*(1/(instance_y[j]*(y_pred_ptest[0][1])+(1-instance_y[j])*(1-y_pred_ptest[0][1]))-1))*1/(IR1[instance_y[j]]/(buffer_len))])
                             #test_clf.partial_fit(instance_x[j].reshape(1, -1), [instance_y[j]], classes=[0, 1],sample_weight=[1/(IR1[instance_y[j]]/(buffer_len))])
                 
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
        stream = FileStream('imbalance_dataset/synthesize3.csv')
        with open('imbalance_dataset/synthesize3.csv', 'r') as file:
                reader = csv.reader(file)
                line_count = len(list(reader))
        for i in range(0, line_count-1):
                feature, label = stream.next_sample()
                X.append(feature[0])
                y.append(int(label[0]))
        X=np.array(X)
        y=np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-100/(line_count-1), random_state=42)
        
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

        IR={0:0,1:0}
        X, y= stream.next_sample()
        for i in range(POPSIZE):
             individual[i].clf.partial_fit(X_train, y_train, classes=[0, 1])     
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
                print(y_pred_p3)
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

                IR[y_test[j]]+=1
                individual[0].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]], classes=[0, 1])
                individual[1].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]], classes=[0, 1],sample_weight=[1/(IR[y_test[j]]/(j+1))])
                if (y_pred3[0]!=y_test[j]):
                             individual[2].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]], classes=[0, 1],sample_weight=[(1+alpha*(1/(y_test[j]*(1-y_pred_p3[0][1])+(1-y_test[j])*(y_pred_p3[0][1]))-1))*1/(IR[y_test[j]]/(j+1))])
                else: 
                             
                             #individual[2].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]], classes=[0, 1],sample_weight=[1/(IR[y_test[j]]/(j+1))])
                             individual[2].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]], classes=[0, 1],sample_weight=[(1+alpha*(1/(y_test[j]*(y_pred_p3[0][1])+(1-y_test[j])*(1-y_pred_p3[0][1]))-1))*1/(IR[y_test[j]]/(j+1))])
                if ((j+1)%buffer_len==0):
                        sample_x = X_test[j+1-buffer_len:j+1]
                        sample_y = y_test[j+1-buffer_len:j+1]
                        # IR1={0:buffer_len - np.count_nonzero(sample_y),1:np.count_nonzero(sample_y)}
                        IR1={0:IR[0]/((j+1)/buffer_len),1:IR[1]/((j+1)/buffer_len)}
                        alpha,_=differential_evolution(objective_function, bounds, IR1, copy.deepcopy(test_clf))
                        test_clf=copy.deepcopy(individual[2].clf)
                  
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



        print(gmean1[-1])
        print(gmean2[-1])
        print(gmean3[-1])
        plot_x = range(nb_all - 100)
        plot_y1 = gmean1 
        plot_y2 = gmean2
        plot_y3 = gmean3
        ax1.plot(plot_x, plot_y1, color='blue',label='no processing')
        ax1.plot(plot_x, plot_y2, color='green',label='class-dependent cost')
        ax1.plot(plot_x, plot_y3, color='red',label='instance-dependent cost')
        ax1.legend(fontsize=legendsize, ncol=ncol)



        plt.show()
