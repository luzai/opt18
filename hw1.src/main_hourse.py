# -*- coding: utf-8 -*-
from lz import *

import numpy as np
import matplotlib.pyplot as plt

from net import TwoLayerNet
from opt import Solver
from utils import *

N, D, H, C = 1168, 270, 300, 1
# loss = 'mse'
loss = 'huber'

std = 1e-4
# Here is an example use neural network
# but please implement regression model

###########################################################################
# TODO: Implement regression model
###########################################################################

###########################################################################
#                             END OF YOUR CODE                            #
###########################################################################

model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std,
                    loss_name=loss,
                    reg = 5e-4)

print('----------------')
print('Training ')

db = Database('input/cleaned.h5')
small_data = {'X_train': db['train'], 'y_train': db['train_label'],
              'X_val': db['val'], 'y_val': db['val_label'], }

# from clean_hourse_data import train,train_label,val,val_label


return
for k, v in list(small_data.items()):
    print(('%s: ' % k, v.shape))


def r2score(y_scores, y_true):
    from sklearn.metrics import r2_score

    y_scores = y_scores.squeeze()
    y_true = y_true.squeeze().astype(float)
    return r2_score(y_true=y_true, y_pred=y_scores)


learning_rate = 1e-3
solver = Solver(model, small_data,
                print_every=10, num_epochs=100, batch_size=128,
                update_rule='sgd_momentum',
                optim_config={
                    'learning_rate': learning_rate,
                    'momentum': 0.9
                },
                metric=r2score
                )
solver.train()

plt.subplot(2, 1, 1)
plt.title('Training loss')
plt.plot(solver.loss_history, 'o')
plt.xlabel('Iteration')

plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history, '-o', label='train')
plt.plot(solver.val_acc_history, '-o', label='val')
plt.plot([0.5] * len(solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()
