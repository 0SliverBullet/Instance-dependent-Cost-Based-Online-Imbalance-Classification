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
from skmultiflow.trees import HoeffdingTreeClassifier 
import evaluation_online
import read_data

class indi:
    def __init__(self):
        self.x = []
        self.fitx=0.0
        self.clf=HoeffdingTreeClassifier()
majority=1800
minority=200
POPSIZE=2+7*6
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
pretrain = 1
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
        
        threshold=0.0
        datasets=['synthesize','chess','yeast','segment']
        test_id=3
        dataset=datasets[test_id]+'0-5-5tst'
        X,y=read_data.read(datasets[test_id],dataset)
        alpha=0.25

        line_count = len(X) 
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

        S = np.zeros([class_num])
        N = np.zeros([class_num])
        cf = np.zeros([class_num, class_num])
        recall = np.zeros([test_len, class_num])
        gmean = np.zeros([test_len])
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
        S4 = np.zeros([class_num])
        N4 = np.zeros([class_num])
        cf4 = np.zeros([class_num, class_num])
        recall4 = np.zeros([test_len, class_num])
        gmean4 = np.zeros([test_len])
        S5 = np.zeros([class_num])
        N5 = np.zeros([class_num])
        cf5 = np.zeros([class_num, class_num])
        recall5 = np.zeros([test_len, class_num])
        gmean5 = np.zeros([test_len])
        S6 = np.zeros([class_num])
        N6 = np.zeros([class_num])
        cf6 = np.zeros([class_num, class_num])
        recall6 = np.zeros([test_len, class_num])
        gmean6 = np.zeros([test_len])
        S7 = np.zeros([class_num])
        N7 = np.zeros([class_num])
        cf7 = np.zeros([class_num, class_num])
        recall7 = np.zeros([test_len, class_num])
        gmean7 = np.zeros([test_len])
        S8 = np.zeros([class_num])
        N8 = np.zeros([class_num])
        cf8 = np.zeros([class_num, class_num])
        recall8 = np.zeros([test_len, class_num])
        gmean8 = np.zeros([test_len])
        S9 = np.zeros([class_num])
        N9 = np.zeros([class_num])
        cf9 = np.zeros([class_num, class_num])
        recall9 = np.zeros([test_len, class_num])
        gmean9 = np.zeros([test_len])

        S23 = np.zeros([class_num])
        N23 = np.zeros([class_num])
        cf23 = np.zeros([class_num, class_num])
        recall23 = np.zeros([test_len, class_num])
        gmean23 = np.zeros([test_len])
        S24 = np.zeros([class_num])
        N24 = np.zeros([class_num])
        cf24 = np.zeros([class_num, class_num])
        recall24 = np.zeros([test_len, class_num])
        gmean24 = np.zeros([test_len])
        S25 = np.zeros([class_num])
        N25 = np.zeros([class_num])
        cf25 = np.zeros([class_num, class_num])
        recall25 = np.zeros([test_len, class_num])
        gmean25 = np.zeros([test_len])
        S26 = np.zeros([class_num])
        N26 = np.zeros([class_num])
        cf26 = np.zeros([class_num, class_num])
        recall26 = np.zeros([test_len, class_num])
        gmean26 = np.zeros([test_len])
        S27 = np.zeros([class_num])
        N27 = np.zeros([class_num])
        cf27 = np.zeros([class_num, class_num])
        recall27 = np.zeros([test_len, class_num])
        gmean27 = np.zeros([test_len])
        S28 = np.zeros([class_num])
        N28 = np.zeros([class_num])
        cf28 = np.zeros([class_num, class_num])
        recall28 = np.zeros([test_len, class_num])
        gmean28 = np.zeros([test_len])
        S29 = np.zeros([class_num])
        N29 = np.zeros([class_num])
        cf29 = np.zeros([class_num, class_num])
        recall29 = np.zeros([test_len, class_num])
        gmean29 = np.zeros([test_len])

        S43 = np.zeros([class_num])
        N43 = np.zeros([class_num])
        cf43 = np.zeros([class_num, class_num])
        recall43 = np.zeros([test_len, class_num])
        gmean43 = np.zeros([test_len])
        S44 = np.zeros([class_num])
        N44 = np.zeros([class_num])
        cf44 = np.zeros([class_num, class_num])
        recall44 = np.zeros([test_len, class_num])
        gmean44 = np.zeros([test_len])
        S45 = np.zeros([class_num])
        N45 = np.zeros([class_num])
        cf45 = np.zeros([class_num, class_num])
        recall45 = np.zeros([test_len, class_num])
        gmean45 = np.zeros([test_len])
        S46 = np.zeros([class_num])
        N46 = np.zeros([class_num])
        cf46 = np.zeros([class_num, class_num])
        recall46 = np.zeros([test_len, class_num])
        gmean46 = np.zeros([test_len])
        S47 = np.zeros([class_num])
        N47 = np.zeros([class_num])
        cf47 = np.zeros([class_num, class_num])
        recall47 = np.zeros([test_len, class_num])
        gmean47 = np.zeros([test_len])
        S48 = np.zeros([class_num])
        N48 = np.zeros([class_num])
        cf48 = np.zeros([class_num, class_num])
        recall48 = np.zeros([test_len, class_num])
        gmean48 = np.zeros([test_len])
        S49 = np.zeros([class_num])
        N49 = np.zeros([class_num])
        cf49 = np.zeros([class_num, class_num])
        recall49 = np.zeros([test_len, class_num])
        gmean49 = np.zeros([test_len])

        S63 = np.zeros([class_num])
        N63 = np.zeros([class_num])
        cf63 = np.zeros([class_num, class_num])
        recall63 = np.zeros([test_len, class_num])
        gmean63 = np.zeros([test_len])
        S64 = np.zeros([class_num])
        N64 = np.zeros([class_num])
        cf64 = np.zeros([class_num, class_num])
        recall64 = np.zeros([test_len, class_num])
        gmean64 = np.zeros([test_len])
        S65 = np.zeros([class_num])
        N65 = np.zeros([class_num])
        cf65 = np.zeros([class_num, class_num])
        recall65 = np.zeros([test_len, class_num])
        gmean65 = np.zeros([test_len])
        S66 = np.zeros([class_num])
        N66 = np.zeros([class_num])
        cf66 = np.zeros([class_num, class_num])
        recall66 = np.zeros([test_len, class_num])
        gmean66 = np.zeros([test_len])
        S67 = np.zeros([class_num])
        N67 = np.zeros([class_num])
        cf67 = np.zeros([class_num, class_num])
        recall67 = np.zeros([test_len, class_num])
        gmean67 = np.zeros([test_len])
        S68 = np.zeros([class_num])
        N68 = np.zeros([class_num])
        cf68 = np.zeros([class_num, class_num])
        recall68 = np.zeros([test_len, class_num])
        gmean68 = np.zeros([test_len])
        S69 = np.zeros([class_num])
        N69 = np.zeros([class_num])
        cf69 = np.zeros([class_num, class_num])
        recall69 = np.zeros([test_len, class_num])
        gmean69 = np.zeros([test_len])

        S83 = np.zeros([class_num])
        N83 = np.zeros([class_num])
        cf83 = np.zeros([class_num, class_num])
        recall83 = np.zeros([test_len, class_num])
        gmean83 = np.zeros([test_len])
        S84 = np.zeros([class_num])
        N84 = np.zeros([class_num])
        cf84 = np.zeros([class_num, class_num])
        recall84 = np.zeros([test_len, class_num])
        gmean84 = np.zeros([test_len])
        S85 = np.zeros([class_num])
        N85 = np.zeros([class_num])
        cf85 = np.zeros([class_num, class_num])
        recall85 = np.zeros([test_len, class_num])
        gmean85 = np.zeros([test_len])
        S86 = np.zeros([class_num])
        N86 = np.zeros([class_num])
        cf86 = np.zeros([class_num, class_num])
        recall86 = np.zeros([test_len, class_num])
        gmean86 = np.zeros([test_len])
        S87 = np.zeros([class_num])
        N87 = np.zeros([class_num])
        cf87 = np.zeros([class_num, class_num])
        recall87 = np.zeros([test_len, class_num])
        gmean87 = np.zeros([test_len])
        S88 = np.zeros([class_num])
        N88 = np.zeros([class_num])
        cf88 = np.zeros([class_num, class_num])
        recall88 = np.zeros([test_len, class_num])
        gmean88 = np.zeros([test_len])
        S89 = np.zeros([class_num])
        N89 = np.zeros([class_num])
        cf89 = np.zeros([class_num, class_num])
        recall89 = np.zeros([test_len, class_num])
        gmean89 = np.zeros([test_len])

        Sa3 = np.zeros([class_num])
        Na3 = np.zeros([class_num])
        cfa3 = np.zeros([class_num, class_num])
        recalla3 = np.zeros([test_len, class_num])
        gmeana3 = np.zeros([test_len])
        Sa4 = np.zeros([class_num])
        Na4 = np.zeros([class_num])
        cfa4 = np.zeros([class_num, class_num])
        recalla4 = np.zeros([test_len, class_num])
        gmeana4 = np.zeros([test_len])
        Sa5 = np.zeros([class_num])
        Na5 = np.zeros([class_num])
        cfa5 = np.zeros([class_num, class_num])
        recalla5 = np.zeros([test_len, class_num])
        gmeana5 = np.zeros([test_len])
        Sa6 = np.zeros([class_num])
        Na6 = np.zeros([class_num])
        cfa6 = np.zeros([class_num, class_num])
        recalla6 = np.zeros([test_len, class_num])
        gmeana6 = np.zeros([test_len])
        Sa7 = np.zeros([class_num])
        Na7 = np.zeros([class_num])
        cfa7 = np.zeros([class_num, class_num])
        recalla7 = np.zeros([test_len, class_num])
        gmeana7 = np.zeros([test_len])
        Sa8 = np.zeros([class_num])
        Na8 = np.zeros([class_num])
        cfa8 = np.zeros([class_num, class_num])
        recalla8 = np.zeros([test_len, class_num])
        gmeana8 = np.zeros([test_len])
        Sa9 = np.zeros([class_num])
        Na9 = np.zeros([class_num])
        cfa9 = np.zeros([class_num, class_num])
        recalla9 = np.zeros([test_len, class_num])
        gmeana9 = np.zeros([test_len])

        for i in range(POPSIZE):
             individual[i].clf.partial_fit(X_train, y_train,classes=list(range(class_num)))     
        total0=0
        total1=1
        TP=0
        TN=0
        test_clf=copy.deepcopy(individual[2].clf)
        for j in range(X_test.shape[0]):
                #print(j)
                y_pred1 = individual[0].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p1=individual[0].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred2 = individual[1].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p2=individual[1].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred3 = individual[2].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p3=individual[2].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred4 = individual[3].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p4=individual[3].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred5 = individual[4].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p5=individual[4].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred6 = individual[5].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p6=individual[5].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred7 = individual[6].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p7=individual[6].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred8 = individual[7].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p8=individual[7].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred9 = individual[8].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p9=individual[8].clf.predict_proba(X_test[j].reshape(1,-1))

                y_pred23 = individual[9].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p23=individual[9].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred24 = individual[10].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p24=individual[10].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred25 = individual[11].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p25=individual[11].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred26 = individual[12].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p26=individual[12].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred27 = individual[13].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p27=individual[13].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred28 = individual[14].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p28=individual[14].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred29 = individual[15].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p29=individual[15].clf.predict_proba(X_test[j].reshape(1,-1))

                y_pred43 = individual[16].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p43=individual[16].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred44 = individual[17].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p44=individual[17].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred45 = individual[18].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p45=individual[18].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred46 = individual[19].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p46=individual[19].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred47 = individual[20].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p47=individual[20].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred48 = individual[21].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p48=individual[21].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred49 = individual[22].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p49=individual[22].clf.predict_proba(X_test[j].reshape(1,-1))

                y_pred63 = individual[23].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p63=individual[23].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred64 = individual[24].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p64=individual[24].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred65 = individual[25].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p65=individual[25].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred66 = individual[26].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p66=individual[26].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred67 = individual[27].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p67=individual[27].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred68 = individual[28].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p68=individual[28].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred69 = individual[29].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p69=individual[29].clf.predict_proba(X_test[j].reshape(1,-1))

                y_pred83 = individual[30].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p83=individual[30].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred84 = individual[31].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p84=individual[31].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred85 = individual[32].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p85=individual[32].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred86 = individual[33].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p86=individual[33].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred87 = individual[34].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p87=individual[34].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred88 = individual[35].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p88=individual[35].clf.predict_proba(X_test[j].reshape(1,-1))
                y_pred89 = individual[36].clf.predict(X_test[j].reshape(1,-1))
                y_pred_p89=individual[36].clf.predict_proba(X_test[j].reshape(1,-1))

                y_preda3 = individual[37].clf.predict(X_test[j].reshape(1,-1))
                y_pred_pa3=individual[37].clf.predict_proba(X_test[j].reshape(1,-1))
                y_preda4 = individual[38].clf.predict(X_test[j].reshape(1,-1))
                y_pred_pa4=individual[38].clf.predict_proba(X_test[j].reshape(1,-1))
                y_preda5 = individual[39].clf.predict(X_test[j].reshape(1,-1))
                y_pred_pa5=individual[39].clf.predict_proba(X_test[j].reshape(1,-1))
                y_preda6 = individual[40].clf.predict(X_test[j].reshape(1,-1))
                y_pred_pa6=individual[40].clf.predict_proba(X_test[j].reshape(1,-1))
                y_preda7 = individual[41].clf.predict(X_test[j].reshape(1,-1))
                y_pred_pa7=individual[41].clf.predict_proba(X_test[j].reshape(1,-1))
                y_preda8 = individual[42].clf.predict(X_test[j].reshape(1,-1))
                y_pred_pa8=individual[42].clf.predict_proba(X_test[j].reshape(1,-1))
                y_preda9 = individual[43].clf.predict(X_test[j].reshape(1,-1))
                y_pred_pa9=individual[43].clf.predict_proba(X_test[j].reshape(1,-1))
                 


                test_label = y_test[j]
                test_label = np.expand_dims(test_label, 0)
                test_pre1=y_pred1
                test_pre2=y_pred2
                test_pre3=y_pred3
                test_pre4=y_pred4
                test_pre5=y_pred5
                test_pre6=y_pred6
                test_pre7=y_pred7
                test_pre8=y_pred8
                test_pre9=y_pred9
                
                test_pre23=y_pred23
                test_pre24=y_pred24
                test_pre25=y_pred25
                test_pre26=y_pred26
                test_pre27=y_pred27
                test_pre28=y_pred28
                test_pre29=y_pred29

                test_pre43=y_pred43
                test_pre44=y_pred44
                test_pre45=y_pred45
                test_pre46=y_pred46
                test_pre47=y_pred47
                test_pre48=y_pred48
                test_pre49=y_pred49

                test_pre63=y_pred63
                test_pre64=y_pred64
                test_pre65=y_pred65
                test_pre66=y_pred66
                test_pre67=y_pred67
                test_pre68=y_pred68
                test_pre69=y_pred69

                test_pre83=y_pred83
                test_pre84=y_pred84
                test_pre85=y_pred85
                test_pre86=y_pred86
                test_pre87=y_pred87
                test_pre88=y_pred88
                test_pre89=y_pred89

                test_prea3=y_preda3
                test_prea4=y_preda4
                test_prea5=y_preda5
                test_prea6=y_preda6
                test_prea7=y_preda7
                test_prea8=y_preda8
                test_prea9=y_preda9

                y_pred=0
                test_pre=0
                if j==0:
                     t0=(y_pred_p3[0][0]
                        +y_pred_p4[0][0]
                        +y_pred_p5[0][0]
                        +y_pred_p6[0][0]
                        +y_pred_p7[0][0]
                        +y_pred_p8[0][0]
                        +y_pred_p9[0][0]
                        
                        +y_pred_p23[0][0]
                        +y_pred_p24[0][0]
                        +y_pred_p25[0][0]
                        +y_pred_p26[0][0]
                        +y_pred_p27[0][0]
                        +y_pred_p28[0][0]
                        +y_pred_p29[0][0]
                        
                        +y_pred_p43[0][0]
                        +y_pred_p44[0][0]
                        +y_pred_p45[0][0]
                        +y_pred_p46[0][0]
                        +y_pred_p47[0][0]
                        +y_pred_p48[0][0]
                        +y_pred_p49[0][0]
                        
                        +y_pred_p63[0][0]
                        +y_pred_p64[0][0]
                        +y_pred_p65[0][0]
                        +y_pred_p66[0][0]
                        +y_pred_p67[0][0]
                        +y_pred_p68[0][0]
                        +y_pred_p69[0][0]
                        
                        +y_pred_p83[0][0]
                        +y_pred_p84[0][0]
                        +y_pred_p85[0][0]
                        +y_pred_p86[0][0]
                        +y_pred_p87[0][0]
                        +y_pred_p88[0][0]
                        +y_pred_p89[0][0]
                        
                        +y_pred_pa3[0][0]
                        +y_pred_pa4[0][0]
                        +y_pred_pa5[0][0]
                        +y_pred_pa6[0][0]
                        +y_pred_pa7[0][0]
                        +y_pred_pa8[0][0]
                        +y_pred_pa9[0][0])/42
                     
                     t1=(y_pred_p3[0][1]
                        +y_pred_p4[0][1]
                        +y_pred_p5[0][1]
                        +y_pred_p6[0][1]
                        +y_pred_p7[0][1]
                        +y_pred_p8[0][1]
                        +y_pred_p9[0][1]
                        
                        +y_pred_p23[0][1]
                        +y_pred_p24[0][1]
                        +y_pred_p25[0][1]
                        +y_pred_p26[0][1]
                        +y_pred_p27[0][1]
                        +y_pred_p28[0][1]
                        +y_pred_p29[0][1]
                        
                        +y_pred_p43[0][1]
                        +y_pred_p44[0][1]
                        +y_pred_p45[0][1]
                        +y_pred_p46[0][1]
                        +y_pred_p47[0][1]
                        +y_pred_p48[0][1]
                        +y_pred_p49[0][1]
                        
                        +y_pred_p63[0][1]
                        +y_pred_p64[0][1]
                        +y_pred_p65[0][1]
                        +y_pred_p66[0][1]
                        +y_pred_p67[0][1]
                        +y_pred_p68[0][1]
                        +y_pred_p69[0][1]
                        
                        +y_pred_p83[0][1]
                        +y_pred_p84[0][1]
                        +y_pred_p85[0][1]
                        +y_pred_p86[0][1]
                        +y_pred_p87[0][1]
                        +y_pred_p88[0][1]
                        +y_pred_p89[0][1]
                        
                        +y_pred_pa3[0][1]
                        +y_pred_pa4[0][1]
                        +y_pred_pa5[0][1]
                        +y_pred_pa6[0][1]
                        +y_pred_pa7[0][1]
                        +y_pred_pa8[0][1]
                        +y_pred_pa9[0][1])/42
                     
                     if t1>t0:
                          y_pred=1
                          test_pre=1
                     else:
                          y_pred=0
                          test_pre=0
                else:
                     weight_sum=(gmean3[j-1]
                                +gmean4[j-1]
                                +gmean5[j-1]
                                +gmean6[j-1]
                                +gmean7[j-1]
                                +gmean8[j-1]
                                +gmean9[j-1]
                                
                                +gmean23[j-1]
                                +gmean24[j-1]
                                +gmean25[j-1]
                                +gmean26[j-1]
                                +gmean27[j-1]
                                +gmean28[j-1]
                                +gmean29[j-1]
                                
                                +gmean43[j-1]
                                +gmean44[j-1]
                                +gmean45[j-1]
                                +gmean46[j-1]
                                +gmean47[j-1]
                                +gmean48[j-1]
                                +gmean49[j-1]
                                
                                +gmean63[j-1]
                                +gmean64[j-1]
                                +gmean65[j-1]
                                +gmean66[j-1]
                                +gmean67[j-1]
                                +gmean68[j-1]
                                +gmean69[j-1]
                                
                                +gmean83[j-1]
                                +gmean84[j-1]
                                +gmean85[j-1]
                                +gmean86[j-1]
                                +gmean87[j-1]
                                +gmean88[j-1]
                                +gmean89[j-1]
                                
                                +gmeana3[j-1]
                                +gmeana4[j-1]
                                +gmeana5[j-1]
                                +gmeana6[j-1]
                                +gmeana7[j-1]
                                +gmeana8[j-1]
                                +gmeana9[j-1])
                     
                     weight1=(gmean3[j-1]+1)/(weight_sum+42)
                     weight2=(gmean4[j-1]+1)/(weight_sum+42)
                     weight3=(gmean5[j-1]+1)/(weight_sum+42)
                     weight4=(gmean6[j-1]+1)/(weight_sum+42)
                     weight5=(gmean7[j-1]+1)/(weight_sum+42)
                     weight6=(gmean8[j-1]+1)/(weight_sum+42)
                     weight7=(gmean9[j-1]+1)/(weight_sum+42)

                     weight21=(gmean23[j-1]+1)/(weight_sum+42)
                     weight22=(gmean24[j-1]+1)/(weight_sum+42)
                     weight23=(gmean25[j-1]+1)/(weight_sum+42)
                     weight24=(gmean26[j-1]+1)/(weight_sum+42)
                     weight25=(gmean27[j-1]+1)/(weight_sum+42)
                     weight26=(gmean28[j-1]+1)/(weight_sum+42)
                     weight27=(gmean29[j-1]+1)/(weight_sum+42)

                     weight41=(gmean43[j-1]+1)/(weight_sum+42)
                     weight42=(gmean44[j-1]+1)/(weight_sum+42)
                     weight43=(gmean45[j-1]+1)/(weight_sum+42)
                     weight44=(gmean46[j-1]+1)/(weight_sum+42)
                     weight45=(gmean47[j-1]+1)/(weight_sum+42)
                     weight46=(gmean48[j-1]+1)/(weight_sum+42)
                     weight47=(gmean49[j-1]+1)/(weight_sum+42)

                     weight61=(gmean63[j-1]+1)/(weight_sum+42)
                     weight62=(gmean64[j-1]+1)/(weight_sum+42)
                     weight63=(gmean65[j-1]+1)/(weight_sum+42)
                     weight64=(gmean66[j-1]+1)/(weight_sum+42)
                     weight65=(gmean67[j-1]+1)/(weight_sum+42)
                     weight66=(gmean68[j-1]+1)/(weight_sum+42)
                     weight67=(gmean69[j-1]+1)/(weight_sum+42)

                     weight81=(gmean83[j-1]+1)/(weight_sum+42)
                     weight82=(gmean84[j-1]+1)/(weight_sum+42)
                     weight83=(gmean85[j-1]+1)/(weight_sum+42)
                     weight84=(gmean86[j-1]+1)/(weight_sum+42)
                     weight85=(gmean87[j-1]+1)/(weight_sum+42)
                     weight86=(gmean88[j-1]+1)/(weight_sum+42)
                     weight87=(gmean89[j-1]+1)/(weight_sum+42)


                     weighta1=(gmeana3[j-1]+1)/(weight_sum+42)
                     weighta2=(gmeana4[j-1]+1)/(weight_sum+42)
                     weighta3=(gmeana5[j-1]+1)/(weight_sum+42)
                     weighta4=(gmeana6[j-1]+1)/(weight_sum+42)
                     weighta5=(gmeana7[j-1]+1)/(weight_sum+42)
                     weighta6=(gmeana8[j-1]+1)/(weight_sum+42)
                     weighta7=(gmeana9[j-1]+1)/(weight_sum+42)

                     t0=(y_pred_p3[0][0]*weight1
                        +y_pred_p4[0][0]*weight2
                        +y_pred_p5[0][0]*weight3
                        +y_pred_p6[0][0]*weight4
                        +y_pred_p7[0][0]*weight5
                        +y_pred_p8[0][0]*weight6
                        +y_pred_p9[0][0]*weight7
                        
                        +y_pred_p23[0][0]*weight21
                        +y_pred_p24[0][0]*weight22
                        +y_pred_p25[0][0]*weight23
                        +y_pred_p26[0][0]*weight24
                        +y_pred_p27[0][0]*weight25
                        +y_pred_p28[0][0]*weight26
                        +y_pred_p29[0][0]*weight27
                        
                        +y_pred_p43[0][0]*weight41
                        +y_pred_p44[0][0]*weight42
                        +y_pred_p45[0][0]*weight43
                        +y_pred_p46[0][0]*weight44
                        +y_pred_p47[0][0]*weight45
                        +y_pred_p48[0][0]*weight46
                        +y_pred_p49[0][0]*weight47
                        
                        +y_pred_p63[0][0]*weight61
                        +y_pred_p64[0][0]*weight62
                        +y_pred_p65[0][0]*weight63
                        +y_pred_p66[0][0]*weight64
                        +y_pred_p67[0][0]*weight65
                        +y_pred_p68[0][0]*weight66
                        +y_pred_p69[0][0]*weight67
                        
                        +y_pred_p83[0][0]*weight81
                        +y_pred_p84[0][0]*weight82
                        +y_pred_p85[0][0]*weight83
                        +y_pred_p86[0][0]*weight84
                        +y_pred_p87[0][0]*weight85
                        +y_pred_p88[0][0]*weight86
                        +y_pred_p89[0][0]*weight87
                        
                        +y_pred_pa3[0][0]*weighta1
                        +y_pred_pa4[0][0]*weighta2
                        +y_pred_pa5[0][0]*weighta3
                        +y_pred_pa6[0][0]*weighta4
                        +y_pred_pa7[0][0]*weighta5
                        +y_pred_pa8[0][0]*weighta6
                        +y_pred_pa9[0][0]*weighta7)
                     
                     t1=(y_pred_p3[0][1]*weight1
                        +y_pred_p4[0][1]*weight2
                        +y_pred_p5[0][1]*weight3
                        +y_pred_p6[0][1]*weight4
                        +y_pred_p7[0][1]*weight5
                        +y_pred_p8[0][1]*weight6
                        +y_pred_p9[0][1]*weight7
                        
                        +y_pred_p23[0][1]*weight21
                        +y_pred_p24[0][1]*weight22
                        +y_pred_p25[0][1]*weight23
                        +y_pred_p26[0][1]*weight24
                        +y_pred_p27[0][1]*weight25
                        +y_pred_p28[0][1]*weight26
                        +y_pred_p29[0][1]*weight27
                        
                        +y_pred_p43[0][1]*weight41
                        +y_pred_p44[0][1]*weight42
                        +y_pred_p45[0][1]*weight43
                        +y_pred_p46[0][1]*weight44
                        +y_pred_p47[0][1]*weight45
                        +y_pred_p48[0][1]*weight46
                        +y_pred_p49[0][1]*weight47
                        
                        +y_pred_p63[0][1]*weight61
                        +y_pred_p64[0][1]*weight62
                        +y_pred_p65[0][1]*weight63
                        +y_pred_p66[0][1]*weight64
                        +y_pred_p67[0][1]*weight65
                        +y_pred_p68[0][1]*weight66
                        +y_pred_p69[0][1]*weight67
                        
                        +y_pred_p83[0][1]*weight81
                        +y_pred_p84[0][1]*weight82
                        +y_pred_p85[0][1]*weight83
                        +y_pred_p86[0][1]*weight84
                        +y_pred_p87[0][1]*weight85
                        +y_pred_p88[0][1]*weight86
                        +y_pred_p89[0][1]*weight87
                        
                        +y_pred_pa3[0][1]*weighta1
                        +y_pred_pa4[0][1]*weighta2
                        +y_pred_pa5[0][1]*weighta3
                        +y_pred_pa6[0][1]*weighta4
                        +y_pred_pa7[0][1]*weighta5
                        +y_pred_pa8[0][1]*weighta6
                        +y_pred_pa9[0][1]*weighta7)
                     if t1>t0:
                          y_pred=1
                          test_pre=1
                     else:
                          y_pred=0
                          test_pre=0
                                             
                print(j,t0,t1,y_pred,y_test[j])
                #print(j)
                recall[j, :], gmean[j], S, N = evaluation_online.pf_online(S, N,test_label,test_pre)
                cf = evaluation_online.confusion_online(cf, test_label, test_pre)
                recall1[j, :], gmean1[j], S1, N1 = evaluation_online.pf_online(S1, N1,test_label,test_pre1)
                cf1 = evaluation_online.confusion_online(cf1, test_label, test_pre1)
                recall2[j, :], gmean2[j], S2, N2 = evaluation_online.pf_online(S2, N2,test_label,test_pre2)
                cf2 = evaluation_online.confusion_online(cf2, test_label, test_pre2)
                recall3[j, :], gmean3[j], S3, N3 = evaluation_online.pf_online(S3, N3,test_label,test_pre3)
                cf3 = evaluation_online.confusion_online(cf3, test_label, test_pre3)
                recall4[j, :], gmean4[j], S4, N4 = evaluation_online.pf_online(S4, N4,test_label,test_pre4)
                cf4 = evaluation_online.confusion_online(cf4, test_label, test_pre4)
                recall5[j, :], gmean5[j], S5, N5 = evaluation_online.pf_online(S5, N5,test_label,test_pre5)
                cf5 = evaluation_online.confusion_online(cf5, test_label, test_pre5)
                recall6[j, :], gmean6[j], S6, N6 = evaluation_online.pf_online(S6, N6,test_label,test_pre6)
                cf6 = evaluation_online.confusion_online(cf6, test_label, test_pre6)
                recall7[j, :], gmean7[j], S7, N7 = evaluation_online.pf_online(S7, N7,test_label,test_pre7)
                cf7 = evaluation_online.confusion_online(cf7, test_label, test_pre7)
                recall8[j, :], gmean8[j], S8, N8 = evaluation_online.pf_online(S8, N8,test_label,test_pre8)
                cf8 = evaluation_online.confusion_online(cf8, test_label, test_pre8)
                recall9[j, :], gmean9[j], S9, N9 = evaluation_online.pf_online(S9, N9,test_label,test_pre9)
                cf9 = evaluation_online.confusion_online(cf9, test_label, test_pre9)

                recall23[j, :], gmean23[j], S23, N23 = evaluation_online.pf_online(S23, N23,test_label,test_pre23)
                cf23 = evaluation_online.confusion_online(cf23, test_label, test_pre23)
                recall24[j, :], gmean24[j], S24, N24 = evaluation_online.pf_online(S24, N24,test_label,test_pre24)
                cf24 = evaluation_online.confusion_online(cf24, test_label, test_pre24)
                recall25[j, :], gmean25[j], S25, N25 = evaluation_online.pf_online(S25, N25,test_label,test_pre25)
                cf25 = evaluation_online.confusion_online(cf25, test_label, test_pre25)
                recall26[j, :], gmean26[j], S26, N26 = evaluation_online.pf_online(S26, N26,test_label,test_pre26)
                cf26 = evaluation_online.confusion_online(cf26, test_label, test_pre26)
                recall27[j, :], gmean27[j], S27, N27 = evaluation_online.pf_online(S27, N27,test_label,test_pre27)
                cf27 = evaluation_online.confusion_online(cf27, test_label, test_pre27)
                recall28[j, :], gmean28[j], S28, N28 = evaluation_online.pf_online(S28, N28,test_label,test_pre28)
                cf28 = evaluation_online.confusion_online(cf28, test_label, test_pre28)
                recall29[j, :], gmean29[j], S29, N29 = evaluation_online.pf_online(S29, N29,test_label,test_pre29)
                cf29 = evaluation_online.confusion_online(cf29, test_label, test_pre29)

                recall43[j, :], gmean43[j], S43, N43 = evaluation_online.pf_online(S43, N43,test_label,test_pre43)
                cf43 = evaluation_online.confusion_online(cf43, test_label, test_pre43)
                recall44[j, :], gmean44[j], S44, N44 = evaluation_online.pf_online(S44, N44,test_label,test_pre44)
                cf44 = evaluation_online.confusion_online(cf44, test_label, test_pre44)
                recall45[j, :], gmean45[j], S45, N45 = evaluation_online.pf_online(S45, N45,test_label,test_pre45)
                cf45 = evaluation_online.confusion_online(cf45, test_label, test_pre45)
                recall46[j, :], gmean46[j], S46, N46 = evaluation_online.pf_online(S46, N46,test_label,test_pre46)
                cf46 = evaluation_online.confusion_online(cf46, test_label, test_pre46)
                recall47[j, :], gmean47[j], S47, N47 = evaluation_online.pf_online(S47, N47,test_label,test_pre47)
                cf47 = evaluation_online.confusion_online(cf47, test_label, test_pre47)
                recall48[j, :], gmean48[j], S48, N48 = evaluation_online.pf_online(S48, N48,test_label,test_pre48)
                cf48 = evaluation_online.confusion_online(cf48, test_label, test_pre48)
                recall49[j, :], gmean49[j], S49, N49 = evaluation_online.pf_online(S49, N49,test_label,test_pre49)
                cf49 = evaluation_online.confusion_online(cf49, test_label, test_pre49)

                recall63[j, :], gmean63[j], S63, N63 = evaluation_online.pf_online(S63, N63,test_label,test_pre63)
                cf63 = evaluation_online.confusion_online(cf63, test_label, test_pre63)
                recall64[j, :], gmean64[j], S64, N64 = evaluation_online.pf_online(S64, N64,test_label,test_pre64)
                cf64 = evaluation_online.confusion_online(cf64, test_label, test_pre64)
                recall65[j, :], gmean65[j], S65, N65 = evaluation_online.pf_online(S65, N65,test_label,test_pre65)
                cf65 = evaluation_online.confusion_online(cf65, test_label, test_pre65)
                recall66[j, :], gmean66[j], S66, N66 = evaluation_online.pf_online(S66, N66,test_label,test_pre66)
                cf66 = evaluation_online.confusion_online(cf66, test_label, test_pre66)
                recall67[j, :], gmean67[j], S67, N67 = evaluation_online.pf_online(S67, N67,test_label,test_pre67)
                cf67 = evaluation_online.confusion_online(cf67, test_label, test_pre67)
                recall68[j, :], gmean68[j], S68, N68 = evaluation_online.pf_online(S68, N68,test_label,test_pre68)
                cf68 = evaluation_online.confusion_online(cf68, test_label, test_pre68)
                recall69[j, :], gmean69[j], S69, N69 = evaluation_online.pf_online(S69, N69,test_label,test_pre69)
                cf69 = evaluation_online.confusion_online(cf69, test_label, test_pre69)

                recall83[j, :], gmean83[j], S83, N83 = evaluation_online.pf_online(S83, N83,test_label,test_pre83)
                cf83 = evaluation_online.confusion_online(cf83, test_label, test_pre83)
                recall84[j, :], gmean84[j], S84, N84 = evaluation_online.pf_online(S84, N84,test_label,test_pre84)
                cf84 = evaluation_online.confusion_online(cf84, test_label, test_pre84)
                recall85[j, :], gmean85[j], S85, N85 = evaluation_online.pf_online(S85, N85,test_label,test_pre85)
                cf85 = evaluation_online.confusion_online(cf85, test_label, test_pre85)
                recall86[j, :], gmean86[j], S86, N86 = evaluation_online.pf_online(S86, N86,test_label,test_pre86)
                cf86 = evaluation_online.confusion_online(cf86, test_label, test_pre86)
                recall87[j, :], gmean87[j], S87, N87 = evaluation_online.pf_online(S87, N87,test_label,test_pre87)
                cf87 = evaluation_online.confusion_online(cf87, test_label, test_pre87)
                recall88[j, :], gmean88[j], S88, N88 = evaluation_online.pf_online(S88, N88,test_label,test_pre88)
                cf88 = evaluation_online.confusion_online(cf88, test_label, test_pre88)
                recall89[j, :], gmean89[j], S89, N89 = evaluation_online.pf_online(S89, N89,test_label,test_pre89)
                cf89 = evaluation_online.confusion_online(cf89, test_label, test_pre89)

                recalla3[j, :], gmeana3[j], Sa3, Na3 = evaluation_online.pf_online(Sa3, Na3,test_label,test_prea3)
                cfa3 = evaluation_online.confusion_online(cfa3, test_label, test_prea3)
                recalla4[j, :], gmeana4[j], Sa4, Na4 = evaluation_online.pf_online(Sa4, Na4,test_label,test_prea4)
                cfa4 = evaluation_online.confusion_online(cfa4, test_label, test_prea4)
                recalla5[j, :], gmeana5[j], Sa5, Na5 = evaluation_online.pf_online(Sa5, Na5,test_label,test_prea5)
                cfa5 = evaluation_online.confusion_online(cfa5, test_label, test_prea5)
                recalla6[j, :], gmeana6[j], Sa6, Na6 = evaluation_online.pf_online(Sa6, Na6,test_label,test_prea6)
                cfa6 = evaluation_online.confusion_online(cfa6, test_label, test_prea6)
                recalla7[j, :], gmeana7[j], Sa7, Na7 = evaluation_online.pf_online(Sa7, Na7,test_label,test_prea7)
                cfa7 = evaluation_online.confusion_online(cfa7, test_label, test_prea7)
                recalla8[j, :], gmeana8[j], Sa8, Na8 = evaluation_online.pf_online(Sa8, Na8,test_label,test_prea8)
                cfa8 = evaluation_online.confusion_online(cfa8, test_label, test_prea8)
                recalla9[j, :], gmeana9[j], Sa9, Na9 = evaluation_online.pf_online(Sa9, Na9,test_label,test_prea9)
                cfa9 = evaluation_online.confusion_online(cfa9, test_label, test_prea9)

                class_size[y_test[j]]+=1
                #class_dependent_cost=IR[1-y_test[j]]/(j+1)
                class_dependent_cost=1/(class_size[y_test[j]]/(len(y_train)+j+1))
                individual[0].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)))
                individual[1].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[class_dependent_cost])
                p_t1=y_pred_p3[0][y_test[j]]
                p_t2=y_pred_p4[0][y_test[j]]
                p_t3=y_pred_p5[0][y_test[j]]
                p_t4=y_pred_p6[0][y_test[j]]
                p_t5=y_pred_p7[0][y_test[j]]
                p_t6=y_pred_p8[0][y_test[j]]
                p_t7=y_pred_p9[0][y_test[j]]

                p_t21=y_pred_p23[0][y_test[j]]
                p_t22=y_pred_p24[0][y_test[j]]
                p_t23=y_pred_p25[0][y_test[j]]
                p_t24=y_pred_p26[0][y_test[j]]
                p_t25=y_pred_p27[0][y_test[j]]
                p_t26=y_pred_p28[0][y_test[j]]
                p_t27=y_pred_p29[0][y_test[j]]

                p_t41=y_pred_p43[0][y_test[j]]
                p_t42=y_pred_p44[0][y_test[j]]
                p_t43=y_pred_p45[0][y_test[j]]
                p_t44=y_pred_p46[0][y_test[j]]
                p_t45=y_pred_p47[0][y_test[j]]
                p_t46=y_pred_p48[0][y_test[j]]
                p_t47=y_pred_p49[0][y_test[j]]

                p_t61=y_pred_p63[0][y_test[j]]
                p_t62=y_pred_p64[0][y_test[j]]
                p_t63=y_pred_p65[0][y_test[j]]
                p_t64=y_pred_p66[0][y_test[j]]
                p_t65=y_pred_p67[0][y_test[j]]
                p_t66=y_pred_p68[0][y_test[j]]
                p_t67=y_pred_p69[0][y_test[j]]


                p_t81=y_pred_p83[0][y_test[j]]
                p_t82=y_pred_p84[0][y_test[j]]
                p_t83=y_pred_p85[0][y_test[j]]
                p_t84=y_pred_p86[0][y_test[j]]
                p_t85=y_pred_p87[0][y_test[j]]
                p_t86=y_pred_p88[0][y_test[j]]
                p_t87=y_pred_p89[0][y_test[j]]

                p_ta1=y_pred_pa3[0][y_test[j]]
                p_ta2=y_pred_pa4[0][y_test[j]]
                p_ta3=y_pred_pa5[0][y_test[j]]
                p_ta4=y_pred_pa6[0][y_test[j]]
                p_ta5=y_pred_pa7[0][y_test[j]]
                p_ta6=y_pred_pa8[0][y_test[j]]
                p_ta7=y_pred_pa9[0][y_test[j]]


                #FC=1/(IR[y_test[j]]/(j+1))*(-((np.abs(p_t-0.5))**alpha)*np.log(np.abs(p_t-0.5)))
                #FC=class_dependent_cost*(-((np.abs(p_t-0.5))**alpha)*np.log(np.abs(p_t-0.5)))*(1-p_t)
                #FC=class_dependent_cost*(-2*(np.abs(p_t-0.5))*np.log(2*np.abs(p_t-0.5)))
                #
                #alpha=0.000001
                # if y_pred3!=y_test[j]:
                #      FC=class_dependent_cost*(1+p_t)
                # else:
                
                threshold=0.0
                FC1=class_dependent_cost*(threshold+(1-p_t1)**0.125)
                FC2=class_dependent_cost*(threshold+(1-p_t2)**0.25)
                FC3=class_dependent_cost*(threshold+(1-p_t3)**0.5)
                FC4=class_dependent_cost*(threshold+(1-p_t4)**1)
                FC5=class_dependent_cost*(threshold+(1-p_t5)**2)
                FC6=class_dependent_cost*(threshold+(1-p_t6)**4)
                FC7=class_dependent_cost*(threshold+(1-p_t7)**8)

                threshold=0.2
                FC21=class_dependent_cost*(threshold+(1-p_t21)**0.125)
                FC22=class_dependent_cost*(threshold+(1-p_t22)**0.25)
                FC23=class_dependent_cost*(threshold+(1-p_t23)**0.5)
                FC24=class_dependent_cost*(threshold+(1-p_t24)**1)
                FC25=class_dependent_cost*(threshold+(1-p_t25)**2)
                FC26=class_dependent_cost*(threshold+(1-p_t26)**4)
                FC27=class_dependent_cost*(threshold+(1-p_t27)**8)

                threshold=0.4
                FC41=class_dependent_cost*(threshold+(1-p_t41)**0.125)
                FC42=class_dependent_cost*(threshold+(1-p_t42)**0.25)
                FC43=class_dependent_cost*(threshold+(1-p_t43)**0.5)
                FC44=class_dependent_cost*(threshold+(1-p_t44)**1)
                FC45=class_dependent_cost*(threshold+(1-p_t45)**2)
                FC46=class_dependent_cost*(threshold+(1-p_t46)**4)
                FC47=class_dependent_cost*(threshold+(1-p_t47)**8)

                threshold=0.6
                FC61=class_dependent_cost*(threshold+(1-p_t61)**0.125)
                FC62=class_dependent_cost*(threshold+(1-p_t62)**0.25)
                FC63=class_dependent_cost*(threshold+(1-p_t63)**0.5)
                FC64=class_dependent_cost*(threshold+(1-p_t64)**1)
                FC65=class_dependent_cost*(threshold+(1-p_t65)**2)
                FC66=class_dependent_cost*(threshold+(1-p_t66)**4)
                FC67=class_dependent_cost*(threshold+(1-p_t67)**8)

                threshold=0.8
                FC81=class_dependent_cost*(threshold+(1-p_t81)**0.125)
                FC82=class_dependent_cost*(threshold+(1-p_t82)**0.25)
                FC83=class_dependent_cost*(threshold+(1-p_t83)**0.5)
                FC84=class_dependent_cost*(threshold+(1-p_t84)**1)
                FC85=class_dependent_cost*(threshold+(1-p_t85)**2)
                FC86=class_dependent_cost*(threshold+(1-p_t86)**4)
                FC87=class_dependent_cost*(threshold+(1-p_t87)**8)

                threshold=1.0
                FCa1=class_dependent_cost*(threshold+(1-p_ta1)**0.125)
                FCa2=class_dependent_cost*(threshold+(1-p_ta2)**0.25)
                FCa3=class_dependent_cost*(threshold+(1-p_ta3)**0.5)
                FCa4=class_dependent_cost*(threshold+(1-p_ta4)**1)
                FCa5=class_dependent_cost*(threshold+(1-p_ta5)**2)
                FCa6=class_dependent_cost*(threshold+(1-p_ta6)**4)
                FCa7=class_dependent_cost*(threshold+(1-p_ta7)**8)
                # FC=class_dependent_cost*(-np.abs(p_t-0.5))*np.log(np.abs(p_t-0.5))*(1-p_t)**alpha
                #FC=class_dependent_cost*(-np.abs(p_t-0.5+alpha))*np.log(np.abs(p_t-0.5+alpha))*(1-p_t)**0.75
                #print(FC)

                # FC=0.5*class_dependent_cost*(-(np.abs(p_t-alpha))*np.log(np.abs(p_t-alpha))-(np.abs((1-p_t)-alpha))*np.log(np.abs((1-p_t)-alpha)))
                individual[2].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC1])
                individual[3].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC2])
                individual[4].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC3])
                individual[5].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC4])
                individual[6].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC5])
                individual[7].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC6])
                individual[8].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC7])

                individual[9].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC21])
                individual[10].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC22])
                individual[11].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC23])
                individual[12].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC24])
                individual[13].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC25])
                individual[14].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC26])
                individual[15].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC27])

                individual[16].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC41])
                individual[17].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC42])
                individual[18].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC43])
                individual[19].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC44])
                individual[20].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC45])
                individual[21].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC46])
                individual[22].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC47])

                individual[23].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC61])
                individual[24].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC62])
                individual[25].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC63])
                individual[26].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC64])
                individual[27].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC65])
                individual[28].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC66])
                individual[29].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC67])

                individual[30].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC81])
                individual[31].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC82])
                individual[32].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC83])
                individual[33].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC84])
                individual[34].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC85])
                individual[35].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC86])
                individual[36].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FC87])

                individual[37].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FCa1])
                individual[38].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FCa2])
                individual[39].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FCa3])
                individual[40].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FCa4])
                individual[41].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FCa5])
                individual[42].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FCa6])
                individual[43].clf.partial_fit(X_test[j].reshape(1, -1), [y_test[j]],classes=list(range(class_num)), sample_weight=[FCa7])

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
        plt.title(f'Gmean_score over {dataset} in online imbalance learning')


        print(gmean1[-1])
        print(gmean2[-1])
        print(gmean[-1])
        plot_x = range(nb_all - pretrain)
        plot_y1 = gmean1 
        plot_y2 = gmean2
        plot_y3 = gmean
        #, $\\alpha$={alpha}
        ax1.plot(plot_x, plot_y1, color='blue',label='no processing')
        ax1.plot(plot_x, plot_y2, color='green',label='class-dependent cost')
        ax1.plot(plot_x, plot_y3, color='red',label=f'instance-dependent cost')
        
        ax1.legend(fontsize=legendsize, ncol=ncol) 
        plt.savefig(f'results/result1/{datasets[test_id]}/{dataset}_instance_dependent_focal_cost.png')
        plt.show()
       