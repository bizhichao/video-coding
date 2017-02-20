import os

import tensorflow as tf

from dataflow.tools.misc import n_positive_integers


class ImageNetLoader(object):
    def __init__(self, data_dir, gt_file_path):
        with open(gt_file_path, 'r') as f:
            gt_lines = f.readlines()
            gt_pairs = [line.split() for line in gt_lines]
            self.paths = [os.path.join(data_dir, p[0]) for p in gt_pairs]
        print('%d samples in dataset set.' % len(self.paths))

    def inputs(self, image_shape, batch_size, n_epochs, shuffle=True):
        if len(image_shape) == 2:
            image_shape = image_shape + [3]
        # create a queue that produces the image paths to read
        if not n_epochs:
            n_epochs = None
        filename_queue = tf.train.string_input_producer(self.paths, num_epochs=n_epochs)

        # Read a record, getting filenames from the filename_queue.
        reader = tf.WholeFileReader()
        _, value = reader.read(filename_queue)

        height, width, depth = image_shape
        uint8image = tf.image.decode_jpeg(value, channels=depth)

        float_image = tf.cast(uint8image, tf.float32)

        # Image processing for training the network. Note the many random
        # distortions applied to the image.
        # Randomly crop a [height, width] section of the image.

        distorted_image = tf.image.resize_images(float_image, (height, width))

        min_queue_examples = 1000
        num_preprocess_threads = 16
        print('Filling queue with %d ImageNet images before starting to train. '
              'This will take a few minutes.' % min_queue_examples)

        # Generate a batch of images and labels by building up a queue of examples.
        if shuffle:
            feature_batch = tf.train.shuffle_batch(
                [distorted_image],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples)
        else:
            feature_batch = tf.train.batch(
                [distorted_image],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size)
        return feature_batch

    def inputs_in_cu(self, cu_shape, batch_size, n_epochs, shuffle=True, min_queue_examples=1000,
                     num_preprocess_threads=16, num_cu_per_image=5):
        # create a queue that produces the image paths to read
        if not n_epochs:
            n_epochs = None
        filename_queue = tf.train.string_input_producer(self.paths, num_epochs=n_epochs)

        # Read a record, getting filenames from the filename_queue.
        reader = tf.WholeFileReader()
        _, value = reader.read(filename_queue)

        height, width, depth = cu_shape
        try:
            uint8image = tf.image.decode_jpeg(value, channels=depth)
        except tf.errors.InvalidArgumentError:
            uint8image = tf.zeros(cu_shape, dtype=tf.uint8)

        float32image = tf.cast(uint8image, tf.float32)

        IMG_WIDTH = 224
        IMG_HEIGHT = 224
        float32image = tf.image.resize_image_with_crop_or_pad(float32image, IMG_HEIGHT, IMG_WIDTH)

        cus = split_image_to_cus(float32image, (height, width))
        reshaped_cus = tf.reshape(cus, (-1, height, width, depth))

        num_cus_in_image = reshaped_cus.get_shape()[0]
        _, variance = tf.nn.moments(reshaped_cus, [1, 2, 3])
        _, indices = tf.nn.top_k(variance, k=num_cu_per_image, sorted=False)
        cu_batch = tf.stack([reshaped_cus[indices[i]]
                             for i in range(indices.get_shape().as_list()[0])], 0)

        print('Filling queue with %d ImageNet images before starting to train. '
              'This will take a few minutes.' % min_queue_examples)

        # Generate a batch of images and labels by building up a queue of examples.
        if shuffle:
            feature_batch = tf.train.shuffle_batch(
                [cu_batch],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples,
                enqueue_many=True)
        else:
            feature_batch = tf.train.batch(
                [cu_batch],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size,
                enqueue_many=True)
        return feature_batch


""" Split Image to Coding Units, and Merge Coding Units to Image"""


def split_image_to_cus(inputs, cu_size):
    height, width, depth = inputs.get_shape().as_list()
    cu_h, cu_w = n_positive_integers(2, cu_size)
    height_in_cu = height // cu_h
    width_in_cu = width // cu_w

    inputs = tf.image.resize_image_with_crop_or_pad(inputs, height_in_cu * cu_h, width_in_cu * cu_w)

    slices = tf.split(0, height_in_cu, inputs)
    slices = tf.stack(slices, 0)  # [height_in_cu, 1, width, c]
    tiles = tf.split(2, width_in_cu, slices)  # width_in_cu [height_in_cu, 1, 1, c]
    tiles = tf.stack(tiles, 1)  # [height_in_cu, width_in_cu, 1, 1, c]
    return tiles


def merge_cus_to_image(tiles):
    # [height_in_cu, width_in_cu, cu_h, cu_w, C]
    height_in_cu, width_in_cu, cu_h, cu_w, depth = tiles.get_shape().as_list()
    slices = tf.concat(2, tf.split(0, height_in_cu, tiles))
    image = tf.concat(3, tf.split(1, width_in_cu, slices))
    image = tf.squeeze(image, axis=[0, 1])
    return image
