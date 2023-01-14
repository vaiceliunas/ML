import tensorflow as tf

print(tf.version)

t = tf.zeros([5,5,5,5])
#print(t)
t = tf.reshape(t, [5, 1, 125])
print(t)