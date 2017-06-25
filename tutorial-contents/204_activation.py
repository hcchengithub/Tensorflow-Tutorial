"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
matplotlib

把幾個 Activation Function 都畫了出來.

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# fake data
x = np.linspace(-5, 5, 200)     # x data, shape=(100, 1)

# following are popular activation functions
y_relu = tf.nn.relu(x)
y_sigmoid = tf.nn.sigmoid(x)
y_tanh = tf.nn.tanh(x)
y_softplus = tf.nn.softplus(x)
# y_softmax = tf.nn.softmax(x)  # softmax is a special kind of
                                # activation function, it is about probability

sess = tf.Session()
'''
sess = {Session} <tensorflow.python.client.session.Session object at 0x000002277E322E10>
x = {ndarray} [-5.         -4.94974874 -4.89949749 -4.84924623 -4.79899497 -4.74874372\n -4.69849246 -4.64824121 -4.59798995 -4.54773869 -4.49748744 -4.44723618\n -4.39698492 -4.34673367 -4.29648241 -4.24623116 -4.1959799  -4.14572864\n -4.09547739 -4.04522613 -3.99497487 
y_relu = {Tensor} Tensor("Relu:0", shape=(200,), dtype=float64)
y_sigmoid = {Tensor} Tensor("Sigmoid:0", shape=(200,), dtype=float64)
y_softplus = {Tensor} Tensor("Softplus:0", shape=(200,), dtype=float64)
y_tanh = {Tensor} Tensor("Tanh:0", shape=(200,), dtype=float64)
'''

y_relu, y_sigmoid, y_tanh, y_softplus = sess.run([y_relu, y_sigmoid, y_tanh, y_softplus])
    # 前幾課 202 placeholder 處講過, sess.run() 可以一次 run 一個 array 的 tensors
    # 這裡 run 完又塞給自己... 斷進去看看前後的變化
'''
sess = {Session} <tensorflow.python.client.session.Session object at 0x0000023D86B0D828>
x = {ndarray} [-5.         -4.94974874 -4.89949749 -4.84924623 -4.79899497 -4.74874372\n -4.69849246 -4.64824121 -4.59798995 -4.54773869 -4.49748744 -4.44723618\n -4.39698492 -4.34673367 -4.29648241 -4.24623116 -4.1959799  -4.14572864\n -4.09547739 -4.04522613 -3.99497487 
y_relu = {ndarray} [ 0.          0.          0.          0.          0.          0.          0.\n  0.          0.          0.          0.          0.          0.          0.\n  0.          0.          0.          0.          0.          0.          0.\n  0.          0.         
y_sigmoid = {ndarray} [ 0.00669285  0.00703534  0.00739523  0.00777338  0.00817071  0.00858818\n  0.00902677  0.00948756  0.00997163  0.01048013  0.01101428  0.01157533\n  0.01216461  0.01278351  0.01343346  0.01411598  0.01483266  0.01558515\n  0.01637519  0.01720457  0.01807518 
y_softplus = {ndarray} [ 0.00671535  0.00706021  0.00742271  0.00780375  0.00820428  0.00862527\n  0.00906776  0.00953285  0.01002168  0.01053543  0.01107538  0.01164285\n  0.01223921  0.01286592  0.0135245   0.01421656  0.01494377  0.01570788\n  0.01651074  0.01735429  0.01824053 
y_tanh = {ndarray} [-0.9999092  -0.99989961 -0.99988899 -0.99987726 -0.99986428 -0.99984993\n -0.99983407 -0.99981652 -0.99979713 -0.99977568 -0.99975197 -0.99972575\n -0.99969676 -0.9996647  -0.99962926 -0.99959007 -0.99954674 -0.99949883\n -0.99944586 -0.99938729 -0.99932253 
一看就懂了!!!
'''
# 試試最簡式, 沒錯, 就這樣, 其他都是周邊的東西.
plt.plot(x, y_relu)
plt.show()

# plt to visualize these activation function
plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()