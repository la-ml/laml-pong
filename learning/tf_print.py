import tensorflow as tf

sess = tf.Session()

with sess.as_default():
    ones = tf.ones([4, 4], dtype=tf.int32)

    print("ones First Print")
    ones.eval()

    print("ones")
    ones = tf.Print(ones, [ones])
    ones.eval()

    print("ones_")
    ones_ = tf.Print(ones, [ones])
    ones_.eval()

    result = ones + 1
    result_ = ones_ + 1

    ones = ones + 1  # Prediction: TF Error

    result = ones + 1
    result_ = ones_ + 1

    # print(ones.eval())
    # print(ones_.eval())
    #
    # print(result.eval())
    # print(result_.eval())


