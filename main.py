#%%
import tensorflow as tf
import numpy as np
import cv2
import pylab
from tqdm import tqdm

#%%
def data(batch_n, samples_n, seq_n):
    x = np.array([np.around(np.random.uniform(-1, 1, sample_n + batch_n)).cumsum() for x in range(seq_n)]).T
    x = (x - x.mean()) / x.std()
    t = x[sample_n:, ...]
    x = np.array([x[i:i+sample_n] for i in range(batch_n)])
    return x, t

#%%
sample_n = 100 # time stamp
seq_n = 1 # data shape
output_n = seq_n

inputs = tf.placeholder(tf.float32, [None, sample_n, seq_n])
y = tf.placeholder(tf.float32, [None, output_n])

# rnn = tf.contrib.rnn.BasicRNNCell(128, reuse=tf.AUTO_REUSE) # neurons
# cell = tf.contrib.rnn.MultiRNNCell([lstm, rnn])
lstm = tf.contrib.rnn.BasicLSTMCell(128, reuse=tf.AUTO_REUSE) # neurons
cell = tf.contrib.rnn.MultiRNNCell([lstm])

outputs, states = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

h = tf.contrib.slim.fully_connected(outputs[:, -1], output_n, activation_fn=None)

with tf.name_scope("optimize"):
    loss = tf.losses.mean_squared_error(y, h)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

#%%
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#%% create data

data_n = 10
batch_n = 100
input_data = np.ones((0, sample_n, seq_n))
label_data = np.ones((0, output_n))
for i in range(data_n):
    x, t = data(batch_n, sample_n, seq_n)
    input_data = np.concatenate((input_data, x), 0)
    label_data = np.concatenate((label_data, t), 0)
input_data.shape
label_data.shape

# pylab.plot(input_data[0]) and pylab.show()
# pylab.plot(label_data[0]) and pylab.show()

#%%
i = 0
r = sess.run(h, feed_dict={inputs: input_data[i : i + batch_n]})
pylab.plot(r)
pylab.plot(label_data[i : i + batch_n])

#%%
tmp = np.expand_dims(input_data[0], 0)
for i in range(10):
    tmp.shape, i
    r = sess.run(h, feed_dict={inputs: tmp[:, i:]})
    # pylab.plot(r[i]) and pylab.plot(label_data[i]); i = (i + 1) % data_n
    tmp = np.concatenate((tmp, np.expand_dims(r, 0)), 1)
pylab.plot(tmp[..., 0].flatten())
pylab.show()

#%% train
# label_data.shape
# input_data.shape
# sess.run(states, feed_dict={inputs: input_data, y: label_data})
for i in (range(50)):
    _, lossv = sess.run([optimizer, loss], feed_dict={inputs: input_data, y: label_data})
    print(i, lossv)
    # print(lossv, result[0][:10])
