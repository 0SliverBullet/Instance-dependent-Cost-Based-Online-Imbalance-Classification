# Instance-dependent Cost Based Online Imbalance Classification
Mass and elite strategy based instance-dependent cost online classification


gamma=0.00005, N=56

```python
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
np.random.seed(1234)
random.seed(1234)
pretrain = 1
```

