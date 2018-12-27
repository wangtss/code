import tensorflow as tf


class Models:
    def __init__(self, reg=1e-8):
        self.reg_rate = reg

    def createCNN2DBlock(self, input, filters, kernel_size, phase, padding='same', batchnorm=False, dropout=False):
        output = tf.layers.conv2d(
            inputs=input,
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            data_format='channels_last',
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_rate),
            bias_regularizer=tf.contrib.layers.l2_regularizer(self.reg_rate),
        )

        if batchnorm:
            output = tf.layers.batch_normalization(
                inputs=output,
                training=phase,
            )

        output = tf.nn.relu(output)

        if dropout:
            output = tf.layers.dropout(output, rate=0.5, training=phase)

        return output

    def createCNN3DBlock(self, input, filters, kernel_size, phase, padding='same', batchnorm=False):
        output = tf.layers.conv3d(
            inputs=input,
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            data_format='channels_last',
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_rate),
            bias_regularizer=tf.contrib.layers.l2_regularizer(self.reg_rate),
        )

        if batchnorm:
            output = tf.layers.batch_normalization(
                inputs=output,
                training=phase,
            )

        output = tf.nn.relu(output)

        return output

    def createDenseBlock(self, input, units, phase, batchnorm=False, relu=True):
        output = tf.layers.dense(
            inputs=input,
            units=units,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_rate),
            bias_regularizer=tf.contrib.layers.l2_regularizer(self.reg_rate),
        )

        if batchnorm:
            output = tf.layers.batch_normalization(
                inputs=output,
                training=phase,
            )

        if relu:
            output = tf.nn.relu(output)

        return output

    def createSTNet(self, model_input, phase):
        reshaped_input = tf.expand_dims(model_input, -1)
        with tf.name_scope('3DConvs'), tf.variable_scope('3DConvs', reuse=tf.AUTO_REUSE):
            activation = self.createCNN3DBlock(reshaped_input, 8, 3, phase, batchnorm=True)
            activation = self.createCNN3DBlock(activation, 8, 3, phase, batchnorm=True)
            activation = self.createCNN3DBlock(activation, 3, 1, phase, batchnorm=True)

        with tf.name_scope('2DConvs'), tf.variable_scope('2DConvs', reuse=tf.AUTO_REUSE):
            activation = tf.reshape(activation, [-1, 64, 64, 192])
            activation = self.createCNN2DBlock(activation, 192, 3, phase, batchnorm=True)
            activation = tf.layers.max_pooling2d(activation, 2, 2, padding='same')
            activation = self.createCNN2DBlock(activation, 192, 3, phase, batchnorm=True)
            activation = tf.layers.max_pooling2d(activation, 2, 2, padding='same')
            activation = self.createCNN2DBlock(activation, 192, 3, phase, batchnorm=True)
            activation = tf.layers.average_pooling2d(activation, [16, 16], 1)

        with tf.name_scope('Dense'), tf.variable_scope('Dense', reuse=tf.AUTO_REUSE):
            activation = tf.reshape(activation, [-1, 192])
            activation = self.createDenseBlock(activation, 1, phase, relu=False)

            return tf.reshape(activation, [-1])


if __name__ == '__main__':
    x = tf.random_normal((2, 64, 64, 64))
    model = Models()
    y = model.createSTNet(x, True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    y = sess.run(y)
    print(y)
    from collections import deque
    q = deque()


