# !/usr/bin/python3.5

import tensorflow as tf

sess = tf.Session()

with sess.as_default():
    v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
    v = tf.Print(v, [v])
    assignment = v.assign_add(1)
    v = tf.Print(assignment, [assignment])

    tf.global_variables_initializer().run()

    # assignment.run()
