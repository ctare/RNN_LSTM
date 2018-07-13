#%%
import tensorflow as tf
import numpy as np
import cv2
import pylab
from tqdm import tqdm

#%%
def data(samples_n, seq_n):
    x = np.sin(np.array([np.linspace(0, 10, sample_n + 1) + np.random.uniform(0, 1000) for x in range(seq_n)]).T)
    t = np.squeeze(x[1:, ...], -1)
    x = x[:-1, ...]
    # x = np.zeros((samples_n, seq_n))
    # for i in range(samples_n):
    #     x[i, :] = np.around(np.random.rand(seq_n)).astype(int)
    # t = np.sum(x, axis=1)
    return x, t

#%%
sample_n = 100 # time stamp
seq_n = 1 # data shape
output_n = 1

inputs = tf.placeholder(tf.float32, [None, sample_n, seq_n])
y = tf.placeholder(tf.float32, [None, output_n])

lstm = tf.contrib.rnn.BasicLSTMCell(128, reuse=tf.AUTO_REUSE) # neurons
cell = tf.contrib.rnn.MultiRNNCell([lstm])
# rnn = tf.contrib.rnn.BasicRNNCell(128, reuse=tf.AUTO_REUSE) # neurons
# cell = tf.contrib.rnn.MultiRNNCell([lstm, rnn])

outputs, states = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

hs = [tf.contrib.slim.fully_connected(outputs[:, i], 1, activation_fn=None) for i in range(sample_n)]
output = tf.concat(hs, 1)

with tf.name_scope("optimize"):
    loss = tf.losses.mean_squared_error(y, output)
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

# pylab.plot(input_data[0]) and pylab.show()
# pylab.plot(label_data[0]) and pylab.show()

#%%
r = sess.run(output, feed_dict={inputs: input_data})
r.shape
input_data[0][:10]
r[0][:10]

input_data.shape
label_data.shape
r.shape
# np.expand_dims(r, axis=1)
#
# cat_result = np.concatenate((input_data, np.expand_dims(r, axis=1)), axis=1)
# cat_label = np.concatenate((input_data, np.expand_dims(label_data, axis=1)), axis=1)
# i = 0
# pylab.plot(cat_result[i]) and pylab.plot(cat_label[i]); i = (i + 1) % data_n

#%%
tmp = np.expand_dims(input_data[0], 0)
for i in range(20):
    r = sess.run(output, feed_dict={inputs: tmp})
    # pylab.plot(r[i]) and pylab.plot(label_data[i]); i = (i + 1) % data_n
    pylab.plot(r[0])
    tmp = np.expand_dims(r, -1)
pylab.show()

#%% train
# label_data.shape
# input_data.shape
# sess.run(states, feed_dict={inputs: input_data, y: label_data})
for i in tqdm(range(400)):
    _, lossv, result = sess.run([optimizer, loss, hs[-1]], feed_dict={inputs: input_data, y: label_data})
    # print(lossv, result[0][:10])

tmp = np.expand_dims(input_data[0], 0)
for i in range(100):
    r = sess.run(output, feed_dict={inputs: tmp})
    # pylab.plot(r[i]) and pylab.plot(label_data[i]); i = (i + 1) % data_n
    pylab.plot(r[0])
    tmp = np.expand_dims(r, -1)
pylab.show()
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
