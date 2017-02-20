import tensorflow as tf
from dataflow.loader import celeba
from dataflow.models.gan import DCGAN_G, DCGAN_D, DCGAN, WGAN

flags = tf.flags

tf.flags.DEFINE_string("logdir", "/tmp/gan", "directory of save checkpoint and summaries")
tf.flags.DEFINE_string("data_dir", "F:/img_align_celeba", "path to dataset")

tf.flags.DEFINE_integer("model", "0", "Model to train. 0 - GAN, 1 - WassersteinGAN")

tf.flags.DEFINE_string("optimizer", "RMSProp", "Optimizer to use for training")
tf.flags.DEFINE_float("lr", "0.01", "Learning rate for Adam Optimizer")

tf.flags.DEFINE_integer("batch_size", 64, "batch size for training")
tf.flags.DEFINE_integer("ngf", 64, "# of genrator's first filters")
tf.flags.DEFINE_integer("ndf", 64, "# of critic's first filters")
tf.flags.DEFINE_integer("nc", 3, "# of channels of image")
tf.flags.DEFINE_integer("nz", 100, "size of the latent z vector")
tf.flags.DEFINE_integer("isize", 64, "image size")
tf.flags.DEFINE_integer("n_extra_layers", 0, "# of extra layers on gen and disc")
tf.flags.DEFINE_integer("n_iters", 100000, "# of iterations to train for")
tf.flags.DEFINE_integer("n_D_iters", 5, "# of D iters per each G iter")
tf.flags.DEFINE_integer("clamp_lower", -0.01, "# of D iters per each G iter")
tf.flags.DEFINE_integer("clamp_upper", 0.01, "# of D iters per each G iter")
FLAGS = flags.FLAGS


def main(argv=None):
    # read image from CelebA
    inputs = celeba.CelebALoader(FLAGS.data_dir, FLAGS.isize, 0, 0)
    # build gan
    generator = DCGAN_G(FLAGS.isize, FLAGS.nc, FLAGS.ngf, FLAGS.n_extra_layers, activation=tf.nn.relu)
    critic = DCGAN_D(FLAGS.isize, FLAGS.ndf, FLAGS.n_extra_layers, activation=tf.nn.relu)
    if FLAGS.model == 0:
        gan = DCGAN(FLAGS.isize, FLAGS.nz, FLAGS.batch_size)
    else:
        gan = WGAN(FLAGS.isize, FLAGS.nz, FLAGS.batch_size,
                   clamp_lower=FLAGS.clamp_lower, clamp_upper=FLAGS.clamp_upper)

    gan.build_network(inputs, generator, critic, FLAGS.optimizer, learning_rate=FLAGS.lr)
    gan.build_supervisor(FLAGS.logdir)

    gan.fit(FLAGS.n_D_iters, FLAGS.n_iters)


if __name__ == '__main__':
    tf.app.run()
