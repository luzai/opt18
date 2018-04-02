import numpy as np
import matplotlib.pyplot as plt

from net import TwoLayerNet
from data import *
from opt import Solver

print('==================')
print('Training ')
small_data = get_data()
#small_data = get_CIFAR10_data(num_training=500, num_validation=100, num_test=100, subtract_mean=True)

# for k, v in small_data.items():
#     if len(v.shape) == 4:
#         small_data[k] = v.reshape(v.shape[0], -1)

for k, v in list(small_data.items()):
    print(('%s: ' % k, v.shape))

N, D, H, C = small_data['X_train'].shape[0], 2, 10, 1
# N, D, H, C = small_data['X_train'].shape[0], 3072, 256, 10
loss = 'mse'

N, D, H, C = small_data['X_train'].shape[0], 2, 10, 2
loss = 'softmax'

std = 1e-3
model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C,
                    weight_scale=std, loss_name=loss,
                    reg=1e-5)


def acc(y_scores, y_true):
    if y_scores.shape[1]==2:
        y_pred = np.argmax(y_scores, axis=1)
    else:
        y_pred = y_scores>0.5
        y_pred = y_pred.astype(int)
        y_pred = y_pred.squeeze()
    return np.mean((y_pred == y_true).astype(float))


learning_rate = 1e-3
solver = Solver(model, small_data,
                print_every=10, num_epochs=5000, batch_size=128,
                update_rule='sgd_momentum',
                optim_config={
                    'learning_rate': learning_rate,
                    'momentum': 0.9
                },
                metric=acc,
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
