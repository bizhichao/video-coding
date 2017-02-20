import glob
import os

import numpy as np
import tensorflow as tf

from dataflow.tools.misc import n_positive_integers


class CelebALoader(object):
    def __init__(self, data_dir, image_size, ratio_of_test=0.0, ratio_of_validate=0.0):
        self.data_dir = data_dir
        self.image_size = image_size

        self.ratio_of_test = ratio_of_test
        self.ratio_of_validate = ratio_of_validate

        self.images = {}
        self.read_dataset(ratio_of_test, ratio_of_validate)

    def read_dataset(self, ratio_of_test=0.0, ratio_of_validate=0.0):
        if not os.path.exists(self.data_dir):
            raise ValueError("DataSet CelebA not found")

        # create images list
        extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG']
        images = []
        for extension in extensions:
            file_glob = os.path.join(self.data_dir, '*' + extension)
            images.extend(glob.glob(file_glob))

        if not images:
            raise ValueError('No images found in {}'.format(self.data_dir))

        np.random.shuffle(images)

        nb_of_images = len(images)
        nb_of_validate = int(ratio_of_validate * nb_of_images)
        nb_of_test = int(ratio_of_test * nb_of_images)

        self.images['validate'] = images[: nb_of_validate]
        self.images['test'] = images[nb_of_validate: nb_of_validate + nb_of_test]
        self.images['train'] = images[nb_of_validate + nb_of_test:]
        print('Read DataSet CelebA: #train-{}, #test-{}, #validate-{}'.format(len(self.images['train']),
                                                                              len(self.images['test']),
                                                                              len(self.images['validate'])))

    def preprocess_image(self, image):
        # image cropping
        cropped_image = tf.image.crop_to_bounding_box(image,
                                                      offset_height=55, offset_width=35,
                                                      target_height=100, target_width=100)
        # image resize
        decoded_image_4d = tf.expand_dims(cropped_image, axis=0)
        resized_image_size = n_positive_integers(2, self.image_size)
        resized_image_4d = tf.image.resize_bilinear(decoded_image_4d, resized_image_size)
        resized_image = tf.squeeze(resized_image_4d, axis=[0])

        return resized_image

    def inputs(self, data_key, batch_size, num_epochs=None, shuffle=False):
        filenames = self.images[data_key]
        filenames_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)

        reader = tf.WholeFileReader()
        key, value = reader.read(filenames_queue)
        decoded_image = tf.image.decode_jpeg(value, channels=3)
        decoded_image = tf.cast(decoded_image, tf.float32)

        # image preprocess
        distorted_image = self.preprocess_image(decoded_image)

        # generate batch
        num_preprocess_threads = 4
        num_examples_per_epoch = len(filenames)
        min_queue_examples = int(0.1 * num_examples_per_epoch)

        print('Filling queue with %d images before starting to train. '
              'This will take a few minutes.' % min_queue_examples)

        if shuffle:
            images = tf.train.shuffle_batch([distorted_image],
                                            batch_size=batch_size,
                                            num_threads=num_preprocess_threads,
                                            capacity=min_queue_examples + 3 * batch_size,
                                            min_after_dequeue=min_queue_examples)
        else:
            images = tf.train.batch([distorted_image],
                                    batch_size=batch_size,
                                    num_threads=num_preprocess_threads,
                                    capacity=min_queue_examples + 3 * batch_size)
        return images
