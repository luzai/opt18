import numpy as np
import matplotlib.pyplot as plt

from net import TwoLayerNet
from data import get_data
from opt import Solver
from utils import eval_numerical_gradient, eval_numerical_gradient_array, rel_error

# test forward

np.random.seed(16)
N, D, H, C = 3, 2, 3, 2
loss = 'softmax'

# N, D, H, C = 3, 2, 3, 1
# loss = 'mse'
# loss = 'huber'

std = 1e-3

X = np.random.randn(N, D)
y = np.random.randint(C, size=N)

model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std, loss_name=loss)
print('loss use {}'.format(loss))
print('----------------')
print('Testing test-time forward pass')
model.params['W1'] = np.linspace(-0.7, 0.3, num=D * H).reshape(D, H)
model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
model.params['W2'] = np.linspace(-0.3, 0.4, num=H * C).reshape(H, C)
model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
X = np.linspace(-5.5, 4.5, num=N * D).reshape(D, N).T
scores = model.loss(X)
print('scores are {}'.format(scores))

# test backward
print('----------------')
print('Testing training loss')
y = np.asarray([0, 1, 0])
loss, grads = model.loss(X, y)
print('loss is {}'.format(loss))

# gradient check

for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)
    print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))

