from __future__ import absolute_import, print_function

import matplotlib.pyplot as plt
from cnn.classifiers.fc_net import *
from cnn.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# naive conv forwad pass
from scipy.misc import imread, imresize

kitten, puppy = imread('data/kitten.jpg'), imread('data/puppy.jpg')
# kitten is wide, and puppy is already square
d = kitten.shape[1] - kitten.shape[0]
kitten_cropped = kitten[:, d // 2:-d // 2, :]

img_size = 200  # Make this smaller if it runs too slow
x = np.zeros((2, 3, img_size, img_size))
x[0, :, :, :] = imresize(puppy, (img_size, img_size)).transpose((2, 0, 1))
x[1, :, :, :] = imresize(kitten_cropped, (img_size, img_size)).transpose((2, 0, 1))

# Set up a convolutional weights holding 2 filters, each 3x3
w = np.zeros((2, 3, 3, 3))

# The first filter converts the image to grayscale.
# Set up the red, green, and blue channels of the filter.
w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]
w[0, 1, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]
w[0, 2, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]

# Second filter detects horizontal edges in the blue channel.
w[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

# Vector of biases. We don't need any bias for the grayscale
# filter, but for the edge detection filter we want to add 128
# to each output so that nothing is negative.
b = np.array([0, 128])

# Compute the result of convolving each input in x with each filter in w,
# offsetting by b, and storing the results in out.
out, _ = conv_forward_naive(x, w, b, {'stride': 1, 'pad': 1})


def imshow_noax(img, normalize=True):
    """ Tiny helper to show images as uint8 and remove axis labels """
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255.0 * (img - img_min) / (img_max - img_min)
    plt.imshow(img.astype('uint8'), cmap='gray')
    plt.gca().axis('off')


# Show the original images and the results of the conv operation
plt.subplot(2, 3, 1)
imshow_noax(puppy, normalize=False)
plt.title('Original image')
plt.subplot(2, 3, 2)
imshow_noax(out[0, 0])
plt.title('Grayscale')
plt.subplot(2, 3, 3)
imshow_noax(out[0, 1])
plt.title('Edges')
plt.subplot(2, 3, 4)
imshow_noax(kitten_cropped, normalize=False)
plt.subplot(2, 3, 5)
imshow_noax(out[1, 0])
plt.subplot(2, 3, 6)
imshow_noax(out[1, 1])
plt.show()

# check conv backward pass
x = np.random.randn(4, 3, 5, 5)
w = np.random.randn(2, 3, 3, 3)
b = np.random.randn(2, )
dout = np.random.randn(4, 2, 5, 5)
conv_param = {'stride': 1, 'pad': 1}

dx_num = eval_numerical_gradient_array(
    lambda x: conv_forward_naive(x, w, b, conv_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(
    lambda w: conv_forward_naive(x, w, b, conv_param)[0], w, dout)
db_num = eval_numerical_gradient_array(
    lambda b: conv_forward_naive(x, w, b, conv_param)[0], b, dout)

out, cache = conv_forward_naive(x, w, b, conv_param)
dx, dw, db = conv_backward_naive(dout, cache)

# Your errors should be around 1e-9'
print('Testing conv_backward_naive function')
print('dx error: ', rel_error(dx, dx_num))
print('dw error: ', rel_error(dw, dw_num))
print('db error: ', rel_error(db, db_num))

# check max pooling backward pass
x = np.random.randn(3, 2, 8, 8)
dout = np.random.randn(3, 2, 4, 4)
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

dx_num = eval_numerical_gradient_array(
    lambda x: max_pool_forward_naive(x, pool_param)[0], x, dout)

out, cache = max_pool_forward_naive(x, pool_param)
dx = max_pool_backward_naive(dout, cache)

# Your error should be around 1e-12
print('Testing max_pool_backward_naive function:')
print('dx error: ', rel_error(dx, dx_num))

# check fast conv

from cnn.fast_layers import conv_forward_fast, conv_backward_fast, use_cython
from time import time

if not use_cython:
    print('did not compile cython , try to use tf')
    import tensorflow as tf

    with tf.InteractiveSession().as_default():
        x = tf.random_normal((100, 31, 31, 3))  # data format = NCHW seems only implemented on linux platform.
        x_np = x.eval()
        w = tf.random_normal((3, 3, 3, 25))
        b = tf.random_normal((25,))
        dout = tf.random_normal((100, 16, 16, 25))
        tf.global_variables_initializer()  # intialize will also take some time.
        t0 = time()
        out = tf.nn.conv2d(x, w, strides=(1, 2, 2, 1), padding='SAME')
        out_np = out.eval()
        t1 = time()
        print('forward out shape is', out_np.shape)

        dw = tf.nn.conv2d_backprop_filter(x, (3, 3, 3, 25), dout,
                                          strides=(1, 2, 2, 1), padding='SAME', )
        dw_np = dw.eval()
        dx = tf.nn.conv2d_backprop_input((100, 31, 31, 3), w, dout,
                                         strides=(1, 2, 2, 1), padding='SAME', )
        dx_np = dx.eval()
        t2 = time()
        print('backward grad dx shape is ', dx_np.shape)
        print('conv forward takes %fs' % (t1 - t0))
        print('conv backward takes %fs' % (t2 - t1))
else:
    x = np.random.randn(100, 3, 31, 31)
    w = np.random.randn(25, 3, 3, 3)
    b = np.random.randn(25, )
    dout = np.random.randn(100, 25, 16, 16)
    conv_param = {'stride': 2, 'pad': 1}

    t0 = time()
    out_naive, cache_naive = conv_forward_naive(x, w, b, conv_param)
    t1 = time()
    out_fast, cache_fast = conv_forward_fast(x, w, b, conv_param)
    t2 = time()

    print('Testing conv_forward_fast:')
    print('Naive: %fs' % (t1 - t0))
    print('Fast: %fs' % (t2 - t1))
    print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))
    print('Difference: ', rel_error(out_naive, out_fast))

    t0 = time()
    dx_naive, dw_naive, db_naive = conv_backward_naive(dout, cache_naive)
    t1 = time()
    dx_fast, dw_fast, db_fast = conv_backward_fast(dout, cache_fast)
    t2 = time()

    print('\nTesting conv_backward_fast:')
    print('Naive: %fs' % (t1 - t0))
    print('Fast: %fs' % (t2 - t1))
    print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))
    print('dx difference: ', rel_error(dx_naive, dx_fast))
    print('dw difference: ', rel_error(dw_naive, dw_fast))
    print('db difference: ', rel_error(db_naive, db_fast))
