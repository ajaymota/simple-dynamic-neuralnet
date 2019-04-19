from neuralnet import *
from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {
    'X': (0, 0),
    'Y': (0, 0),
    'learning_rate': (0.01, 1.0),
    'dropout_rate': (0.1, 1.0),
    'batch_size': (1, 4),
    'epochs': (100, 1000),
    'num_hidden_layers': (3, 5),
    'hidden_nodes_per_layer': (4, 6),
    'iterations': (1, 1)
}

optimizer = BayesianOptimization(
    f=train,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=2,
    n_iter=4,
)

print(optimizer.max)

