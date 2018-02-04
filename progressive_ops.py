import tensorflow as tf

from neuralgym.ops.summary_ops import scalar_summary, images_summary


act = lambda x: pixel_norm(wscale_layer(tf.nn.leaky_relu(x)))


def nn_block(x, cnum, name):
    """Basic block for progressive gan consists a stack of
    3x3 conv, leaky_relu, 3x3 conv and leaky_relu.

    Args:
        x (tf.Tensor): tensor input
        cnum (int): channel number

    Returns:
        tf.Tensor: tensor output

    """
    y = x
    y = tf.layers.conv2d(
        y, cnum, 3, 1, padding='same', activation=act, name=name+'_layer1')
    y = tf.layers.conv2d(
        y, cnum, 3, 1, padding='same', activation=act, name=name+'_layer2')
    return y


def pixel_norm(x):
    """Pixel normalization proposed in progressive GAN.

    Args:
        x (tf.Tensor): tensor input

    Returns:
        tf.Tensor: tensor output

    """
    y =  x / tf.sqrt(
        tf.reduce_mean(
            x**2, axis=[1, 2, 3], keep_dims=True) + 1e-8)
    return y


def wscale_layer(x):
    """Applies equalized learning rate to the preceding layer, proposed
    in progressive GAN.

    Args:
        x (tf.Tensor): tensor input

    Returns:
        tf.Tensor: tensor output

    """
    y = x
    # raise NotImplementedError, "wscale layer is not implemented yet."
    return y


def progressive_kt(name, steps=5000):
    """ Claim a progressive changing scalar variable of kt, where steps
    indicates how many steps are required to progressively changing kt from 0
    to 1.

    Args:
        name (string): the name of kt will be 'kt/'+name
        steps (int): how many steps (batches) are required to change kt from 0
            to 1

    Returns:
        tf.Tensor: scalar tensor kt

    """
    kt = tf.get_variable(
        name, dtype=tf.float32, initializer=0.0, trainable=False)
    scalar_summary('kt/'+name, kt)
    update_kt = tf.assign(kt, tf.clip_by_value(kt + 1. / steps, 0., 1.))
    with tf.control_dependencies([update_kt]):
        kt = tf.identity(kt)
    return kt
