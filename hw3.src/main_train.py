# from lz import *
import numpy as np
import matplotlib.pyplot as plt
from cnn.classifiers.conv_net import *
from cnn.data_utils import *
from cnn.solver import Solver

choice_classes = (5, 7, 9)
# choice_classes = range(10)
num_classes = len(choice_classes)

data = get_MNIST_data('./data/mnist', num_training=300, num_validation=100, num_test=100, choice_classes=choice_classes)
for k, v in data.items():
    print(('%s: ' % k, v.shape))

model = ThreeLayerConvNet(
    input_dim=(1, 28, 28),
    weight_scale=0.001, hidden_dim=500, num_classes=num_classes,
    filter_size=3, num_filters=64, reg=0.002)

solver = Solver(model, data,
                num_epochs=10, batch_size=50,
                update_rule='adam',
                optim_config={
                    'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
solver.train()
acc = solver.check_accuracy(data['X_test'], data['y_test'])
print('final test acc is ', acc)
