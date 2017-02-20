import tensorflow as tf
import numpy as np

class DCGAN_G(object):
    def __init__(self, isize, nc, ngf, n_extra_layers=0, activation=tf.nn.relu, scope=None):
        self._isize = isize
        assert self._isize % 16 == 0, "image size should be a multiple of 16"
        self._nc = nc
        self._ngf = ngf
        self._n_extra_layers = n_extra_layers

        self._activation = activation
        self._scope = scope or 'Generator'

    def __call__(self, z, training=True, reuse=None):
        with tf.variable_scope(self._scope, reuse=reuse) as scope:
            self._scope_name = scope.name

            cngf, tisize = self._ngf // 2, 4
            while tisize != self._isize:
                tisize = tisize * 2
                cngf = cngf * 2

            # first conv layer : 1x1 -> 4x4
            net = tf.expand_dims(z, 1)
            net = tf.expand_dims(net, 1)  # now, z has shape [N, 1, 1, nz]
            net = tf.layers.conv2d_transpose(net, cngf, kernel_size=(4, 4), strides=(1, 1),
                                             padding='valid', use_bias=False)
            net = tf.layers.batch_normalization(net, training=training)
            net = tf.nn.relu(net)

            csize, cndf = 4, cngf
            while csize < self._isize // 2:
                cngf = cngf // 2
                csize = csize * 2

                net = tf.layers.conv2d_transpose(net, cngf, kernel_size=(4, 4), strides=(2, 2),
                                                 padding='same', use_bias=False)
                net = tf.layers.batch_normalization(net, training=training)
                net = self._activation(net)

            # extra layers
            for t in range(self._n_extra_layers):
                net = tf.layers.conv2d(net, cngf, kernel_size=(3, 3), strides=(1, 1),
                                       padding='same', use_bias=False)
                net = tf.layers.batch_normalization(net, training=training)
                net = self._activation(net)

            # last convolution
            net = tf.layers.conv2d_transpose(net, self._nc, kernel_size=(4, 4), strides=(2, 2),
                                             padding='same', use_bias=False)
            net = tf.layers.batch_normalization(net, training=training)
            net = self._activation(net)
        return net

    def scope(self):
        if not self._scope_name:
            print('[!]WARNING: You must call function __call__ first before get scope')
        return self._scope_name


class DCGAN_D(object):
    def __init__(self, isize, ndf, n_extra_layers=0, activation=tf.nn.relu, scope=None):
        self._isize = isize
        assert self._isize % 16 == 0, "image size should be a multiple of 16"
        self._ndf = ndf
        self._n_extra_layers = n_extra_layers
        self._activation = activation
        self._scope = scope or 'Critic'
        self._scope_name = None

    def __call__(self, image, training=True, reuse=None):
        with tf.variable_scope(self._scope, reuse=reuse) as scope:
            self._scope_name = scope.name

            # first conv layer
            net = tf.layers.conv2d(image, self._ndf, kernel_size=(4, 4), strides=(2, 2),
                                   padding='same', use_bias=False)
            net = self._activation(net)

            csize, cndf = self._isize // 2, self._ndf

            # extra layers
            for t in range(self._n_extra_layers):
                net = tf.layers.conv2d(net, cndf, kernel_size=(3, 3), strides=(1, 1),
                                       padding='same', use_bias=False)
                net = tf.layers.batch_normalization(net, training=training)
                net = self._activation(net)

            while csize > 4:
                cndf = cndf * 2
                csize = csize // 2

                net = tf.layers.conv2d(net, cndf, kernel_size=(4, 4), strides=(2, 2),
                                       padding='same', use_bias=False)
                net = tf.layers.batch_normalization(net, training=training)
                net = self._activation(net)

            # state size. 4 x 4 x N
            net = tf.layers.conv2d(net, 1, kernel_size=(4, 4), strides=(1, 1),
                                   padding='valid', use_bias=False)
        return net

    def scope(self):
        if not self._scope_name:
            print('[!]WARNING: You must call function __call__ first before get scope')
        return self._scope_name


class DCGAN(object):
    def __init__(self, isize, nz, batch_size, scope=None):
        self.isize = isize
        self.nz = nz

        self.scope = scope or 'GAN'
        self.batch_size = batch_size

    def build_network(self, image_inputer, generator, discriminator, optimizer='Adam', **optimizer_params):
        print("[*]Setting up model...")
        with tf.variable_scope(self.scope) as scope:
            self.scope_name = scope.name

            # set up placeholder
            self.training = tf.placeholder(tf.bool)
            self.z = tf.placeholder(tf.float32, [None, self.nz])
            tf.summary.histogram("z", self.z)

            self.image_real = image_inputer.inputs("train", self.batch_size)
            tf.summary.image('image_real', self.image_real)
            image_real = (self.image_real - 127.5) / 127.5
            logit_real = discriminator(image_real, training=self.training, reuse=None)

            # generate image from random vector: z
            image_fake = generator(self.z, training=self.training)
            logit_fake = discriminator(image_fake, training=self.training, reuse=True)
            self.image_fake = (image_fake * 127.5) + 127.5
            tf.summary.image('image_fake', self.image_fake)

            # criticize real image and fake image using discriminator/critic
            self.errD, self.errG = self._build_loss(logit_real, logit_fake)

            # set up optimizer to optimize loss
            if optimizer == 'Adam':
                self.optimizer = tf.train.AdamOptimizer(**optimizer_params)
            elif optimizer == 'RMSProp':
                self.optimizer = tf.train.RMSPropOptimizer(**optimizer_params)
            else:
                raise NotImplementedError('[ERROR]Specified optimizer {} has not been implemented'.format(optimizer))

            scopeD = discriminator.scope()
            scopeG = generator.scope()
            scope = self.scope_name
            # set up trainable variables of Generator, and Discriminator/Critic
            self.varsD = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopeD)
            self.varsG = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopeG)
            # set up train op
            self.trainD_op, self.trainG_op, self.global_step = self._build_train_op()
            # set up init op
            global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            local_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=scope)
            self.init_op = tf.variables_initializer(global_vars + local_vars)
            # set up summary op and writer op
            summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)
            self.summary_op = None if not summary_ops else tf.summary.merge(summary_ops)
            # set up saver
            vars_to_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            vars_to_save += tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS, scope=scope)
            self.saver = tf.train.Saver(vars_to_save)

    def _build_train_op(self):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        grads_and_vars = self.optimizer.compute_gradients(self.errD, var_list=self.varsD)
        for grad, var in grads_and_vars:
            tf.summary.histogram('grads_and_varsD/' + var.op.name + '/value', var)
            tf.summary.histogram('grads_and_varsD/' + var.op.name + '/gradient', grad)
        trainD = self.optimizer.apply_gradients(grads_and_vars)

        grads_and_vars = self.optimizer.compute_gradients(self.errG, var_list=self.varsG)
        for grad, var in grads_and_vars:
            tf.summary.histogram('grads_and_varsG/' + var.op.name + '/value', var)
            tf.summary.histogram('grads_and_varsG/' + var.op.name + '/gradient', grad)
        trainG = self.optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        return trainD, trainG, global_step

    def _build_loss(self, logit_real, logit_fake):
        # get loss for discriminator/critic
        errD_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_real, labels=tf.ones_like(logit_real)))
        errD_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_fake, labels=tf.zeros_like(logit_fake)))
        errD = errD_real + errD_fake
        tf.summary.scalar('errD_real', errD_real)
        tf.summary.scalar('errD_fake', errD_fake)
        tf.summary.scalar('errD', errD)

        # get loss for generator
        use_d_trick = True
        if use_d_trick:
            errG = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_fake, labels=tf.ones_like(logit_fake)))
        else:
            errG = -errD_fake
        tf.summary.scalar('errG', errG)
        return errD, errG

    def build_supervisor(self, logdir):
        self.logdir = logdir

        def get_feed_dict(training=True):
            batch_z = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.nz]).astype(np.float32)
            feed_dict = {self.z: batch_z, self.training: training}
            return feed_dict

        self.sv = tf.train.Supervisor(init_op=self.init_op, logdir=self.logdir, summary_op=None,
                                      saver=self.saver, global_step=self.global_step,
                                      init_feed_dict=get_feed_dict(True))

    def fit(self, Diters, max_iters):
        with self.sv.managed_session() as sess:
            for _ in range(max_iters):
                if self.sv.should_stop():
                    break

                def get_feed_dict(training=True):
                    batch_z = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.nz]).astype(np.float32)
                    feed_dict = {self.z: batch_z, self.training: training}
                    return feed_dict

                [gs] = sess.run([self.global_step])
                # Update D network
                for i in range(Diters):
                    _, errD = sess.run([self.trainD_op, self.errD], feed_dict=get_feed_dict(True))
                    print('step: {} - errD: {}'.format(gs, errD))

                # Update G network
                if gs % 100 == 0:
                    _, errG, summ = sess.run([self.trainG_op, self.errG, self.summary_op],
                                             feed_dict=get_feed_dict(True))
                    self.sv.summary_computed(sess, summ)
                else:
                    _, errG = sess.run([self.trainG_op, self.errG],
                                       feed_dict=get_feed_dict(True))
                print('step: {} - errG: {}'.format(gs, errG))


class WGAN(DCGAN):
    def __init__(self, isize, nz, batch_size, scope=None, **kwargs):
        super(WGAN, self).__init__(isize, nz, batch_size, scope)
        self.clamp_lower = kwargs['clamp_lower']
        self.clamp_upper = kwargs['clamp_upper']

    def _build_loss(self, logit_real, logit_fake, **kwargs):
        # get loss for discriminator/critic
        errD = tf.reduce_mean(logit_real - logit_fake)
        tf.summary.scalar('errD', errD)
        # get loss for generator
        errG = tf.reduce_mean(logit_fake)
        tf.summary.scalar('errG', errG)
        return errD, errG

    def _build_train_op(self):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        grads_and_vars = self.optimizer.compute_gradients(self.errD, var_list=self.varsD)
        for grad, var in grads_and_vars:
            tf.summary.histogram('grads_and_varsD/' + var.op.name + '/value', var)
            tf.summary.histogram('grads_and_varsD/' + var.op.name + '/gradient', grad)
        trainD = self.optimizer.apply_gradients(grads_and_vars)

        grads_and_vars = self.optimizer.compute_gradients(self.errG, var_list=self.varsG)
        for grad, var in grads_and_vars:
            tf.summary.histogram('grads_and_varsG/' + var.op.name + '/value', var)
            tf.summary.histogram('grads_and_varsG/' + var.op.name + '/gradient', grad)
        trainG = self.optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        with tf.control_dependencies([trainD]):
            clip_varD = [tf.assign(var, tf.clip_by_value(var, self.clamp_lower, self.clamp_upper)) for var in
                         self.varsD]
            clip_varD = tf.group(*clip_varD)

        return clip_varD, trainG, global_step

    def fit(self, Diters, max_iters):
        with self.sv.managed_session() as sess:
            for _ in range(max_iters):
                if self.sv.should_stop():
                    break

                def get_feed_dict(training=True):
                    batch_z = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.nz]).astype(np.float32)
                    feed_dict = {self.z: batch_z, self.training: training}
                    return feed_dict

                [gs] = sess.run([self.global_step], feed_dict=get_feed_dict(True))
                if gs < 25 or gs % 500 == 0:
                    t_Diters = 100
                else:
                    t_Diters = Diters
                # Update D network
                for i in range(t_Diters):
                    _, errD = sess.run([self.trainD_op, self.errD], feed_dict=get_feed_dict(True))
                    print('step: {} - errD: {}'.format(gs, errD))

                # Update G network
                if gs % 100 == 0:
                    _, errG, summ = sess.run([self.trainG_op, self.errG, self.summary_op],
                                             feed_dict=get_feed_dict(True))
                    self.sv.summary_computed(sess, summ)
                else:
                    _, errG = sess.run([self.trainG_op, self.errG],
                                       feed_dict=get_feed_dict(True))
                print('step: {} - errG: {}'.format(gs, errG))


