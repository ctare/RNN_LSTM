#%%
import tensorflow as tf
import numpy as np
import cv2
import pylab
from tqdm import tqdm

#%%
def data(samples_n, seq_n):
    x = np.sin(np.array([np.linspace(0, 10, sample_n + 1) + np.random.uniform(0, 1000) for x in range(seq_n)]).T)
    t = np.squeeze(x[-1, ...])
    x = x[:-1, ...]
    # x = np.zeros((samples_n, seq_n))
    # for i in range(samples_n):
    #     x[i, :] = np.around(np.random.rand(seq_n)).astype(int)
    # t = np.sum(x, axis=1)
    return x, t

#%%
sample_n = 100 # time stamp
seq_n = 2 # data shape
output_n = 2

inputs = tf.placeholder(tf.float32, [None, sample_n, seq_n])
y = tf.placeholder(tf.float32, [None, output_n])

lstm = tf.contrib.rnn.BasicLSTMCell(128, reuse=tf.AUTO_REUSE) # neurons
cell = tf.contrib.rnn.MultiRNNCell([lstm])
# rnn = tf.contrib.rnn.BasicRNNCell(128, reuse=tf.AUTO_REUSE) # neurons
# cell = tf.contrib.rnn.MultiRNNCell([lstm, rnn])

outputs, states = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

h = tf.contrib.slim.fully_connected(outputs[:, -1], 2, activation_fn=None)

with tf.name_scope("optimize"):
    loss = tf.losses.mean_squared_error(y, h)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

#%%
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#%% create data
input_data = []
label_data = []
data_n = 10
for i in range(data_n):
    x, t = data(sample_n, seq_n)
    input_data.append(x)
    label_data.append(t)
input_data = np.asarray(input_data)
label_data = np.asarray(label_data)
input_data.shape
label_data.shape

# pylab.plot(input_data[0]) and pylab.show()
# pylab.plot(label_data[0]) and pylab.show()

#%%
tmp = np.expand_dims(input_data[2], 0)
for i in range(500):
    r = sess.run(h, feed_dict={inputs: tmp[:, i:]})
    # pylab.plot(r[i]) and pylab.plot(label_data[i]); i = (i + 1) % data_n
    tmp = np.concatenate((tmp, np.expand_dims(r, 0)), 1)
pylab.plot(tmp[..., 0].flatten())
pylab.plot(tmp[..., 1].flatten())
pylab.show()

#%% train
# label_data.shape
# input_data.shape
# sess.run(states, feed_dict={inputs: input_data, y: label_data})
for i in (range(400)):
    _, lossv = sess.run([optimizer, loss], feed_dict={inputs: input_data, y: label_data})
    print(i, lossv)
    # print(lossv, result[0][:10])

#%% test
input_test = []
label_test = []
for i in range(1):
    x, t = data(sample_n, seq_n)
    input_test.append(x)
    label_test.append(t)
input_test = np.asarray(input_test)
label_test = np.asarray(label_test)

result, l = sess.run([h, loss], feed_dict={inputs: input_test, y: label_test})
print(input_test[0][:10])
print(input_test[0][:10].sum(axis=1))
print(np.around(result[0][:10]), l)
print(result[0][:10])
print(input_data[0][:10].sum(axis=1))
