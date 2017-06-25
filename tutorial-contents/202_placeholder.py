"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
"""
import tensorflow as tf
import pdb

x1 = tf.placeholder(dtype=tf.float32, shape=None, name='one')
    # name 只能這時候給, 然後就是 read-only 了. default 是 'placeholder:0', placeholder_1, _2...etc
y1 = tf.placeholder(dtype=tf.float32, shape=None, name='two')
z1 = x1 + y1  # 這個 + 應該是 tf.add() 從 z1.name 是 'add:0' 可見得, 不如試試看? --> 果然!
z11 = tf.add(x1, y1)  # 看看產生的東西跟上一行一不一樣? --> [x] z11.op 我看是一樣的

x2 = tf.placeholder(dtype=tf.float32, shape=[2, 1])
y2 = tf.placeholder(dtype=tf.float32, shape=[1, 2])
z2 = tf.matmul(x2, y2)

with tf.Session() as sess:
    # when only one operation to run
    z1_value = sess.run(z1, feed_dict={x1: 1, y1: 2})

    # when run multiple operations
    z1_value, z2_value = sess.run(
        [z1, z2],       # run them together
        feed_dict={
            x1: 1, y1: 2,
            x2: [[2], [2]], y2: [[3, 3]]
        })
    print(z1_value)
    print(z2_value)