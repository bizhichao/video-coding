import tensorflow as tf


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(**feature):
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example
