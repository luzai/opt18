# from lz import *
# init_dev((0,))
# allow_growth()
import tensorflow as tf
from tensorflow import contrib
import matplotlib.pyplot as plt
import math
from cnn.data_utils import *

tf.reset_default_graph()
choice_classes = (5, 7, 9)
# choice_classes = range(10)
num_classes = len(choice_classes)

data = get_MNIST_data('./data/mnist',
                      num_training=None, num_validation=100, num_test=100,
                      choice_classes=choice_classes)
for k, v in data.items():
    if len(v.shape) == 4:
        v = v.transpose((0, 2, 3, 1))
        data[k] = v
    print(('%s: ' % k, v.shape))
(X_train, y_train,
 X_val, y_val,
 X_test, y_test) = (data['X_train'], data['y_train'],
                    data['X_val'], data['y_val'],
                    data['X_test'], data['y_test'])

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)


# conv (3x3x64) - relu - 2x2 max pool - affine - relu - affine - softmax

def get_conv_net(X):
    conv1 = tf.layers.conv2d(X, filters=32, kernel_size=3, padding='SAME', activation=tf.nn.relu)
    conv1 = tf.layers.batch_normalization(conv1, training=is_training)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)
    embed = tf.reshape(pool1, [-1, 14 * 14 * 32])
    fc1 = tf.layers.dense(embed, 300, activation=tf.nn.relu, )
    fc1 = tf.layers.batch_normalization(fc1, training=is_training)
    out = tf.layers.dense(fc1, num_classes)
    return out


def get_conv_net2(X):
    conv1 = tf.layers.conv2d(X, filters=32, kernel_size=3, padding='SAME', activation=tf.nn.relu)
    conv1 = tf.layers.batch_normalization(conv1, training=is_training)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)
    conv2 = contrib.layers.seperable_conv2d(pool1, num_outputs=64,
                                            kernel_size=3,
                                            depth_multiplier=1,
                                            activation_fn=tf.nn.relu,
                                            normalizer_fn=tf.layers.batch_normalization)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)
    conv3 = contrib.layers.seperable_conv2d(pool2, num_outputs=64, kernel_size=3,
                                            depth_multiplier=1,
                                            activation_fn=tf.nn.relu,
                                            normalizer_fn=tf.layers.batch_normalization
                                            )
    pool3 = tf.layers.average_pooling2d(conv3, (7, 7), 1)
    embed = tf.reshape(pool3, [-1, 64])
    out = tf.layers.dense(embed, num_classes)
    return out


y_out = get_conv_net(X)
mean_loss = tf.losses.softmax_cross_entropy(tf.one_hot(y, num_classes), y_out)
optimizer = tf.train.AdamOptimizer(1e-3)

# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)


def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None

    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss, correct_prediction, accuracy]
    if training_now:
        variables[-1] = training

    # counter
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0] / batch_size))):
            # generate indicies for the batch
            start_idx = (i * batch_size) % Xd.shape[0]
            idx = train_indicies[start_idx:start_idx + batch_size]

            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx, :],
                         y: yd[idx],
                         is_training: training_now}
            # get batch size
            actual_batch_size = yd[idx].shape[0]

            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables, feed_dict=feed_dict)

            # aggregate performance stats
            losses.append(loss * actual_batch_size)
            correct += np.sum(corr)

            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}" \
                      .format(iter_cnt, loss, np.sum(corr) / actual_batch_size))
            iter_cnt += 1
        total_correct = correct / Xd.shape[0]
        total_loss = np.sum(losses) / Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}" \
              .format(total_loss, total_correct, e + 1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e + 1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss, total_correct


with tf.Session() as sess:
    with tf.device("/cpu:0"):  # "/cpu:0" or "/gpu:0"
        sess.run(tf.global_variables_initializer())
        print('Training')
        run_model(sess, y_out, mean_loss, X_train, y_train, 1, 64, 100, train_step, True)
        print('Validation')
        run_model(sess, y_out, mean_loss, X_val, y_val, 1, 64)
