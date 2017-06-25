"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0

這是修訂版的第一課，先介紹 tf 的 constant ，這個教法很好。

"""
import tensorflow as tf
import pdb

m1 = tf.constant([[2, 2]])
m2 = tf.constant([[3],
                  [3]])
dot_operation = tf.matmul(m1, m2)
pdb.set_trace()
print(dot_operation)  # wrong! no result
'''
    不是 no result 而是這種東西
    (Pdb) dot_operation 
    <tf.Tensor 'MatMul:0' shape=(1, 1) dtype=int32>
    (Pdb) n
    Tensor("MatMul:0", shape=(1, 1), dtype=int32)
    (Pdb)    type(dot_operation)
    <class 'tensorflow.python.framework.ops.Tensor'>   這種東西就是 tensor 
    (Pdb)    type(m1)
    <class 'tensorflow.python.framework.ops.Tensor'>  # m1, m2 也都是 tensor 
    (Pdb) type(tf)
    <class 'module'>
    (Pdb) type(tf.matmul)
    <class 'function'>
    (Pdb)
    我覺得，TensorFlow 是一台 Virtual Machine 
'''
# method1 use session
sess = tf.Session()  # 大寫的應該是 constructor
result = sess.run(dot_operation)
print(result)  # [[12]]
sess.close()

# method2 use session
with tf.Session() as sess:
    result_ = sess.run(dot_operation)
    print(result_)  # [[12]]