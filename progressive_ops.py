import tensorflow as tf

from neuralgym.ops.summary_ops import scalar_summary, images_summary


act = lambda x: pixel_norm(wscale_layer(tf.nn.leaky_relu(x)))


def nn_block(x, cnum, name):
    """Basic block for progressive gan consists a stack of
    3x3 conv, leaky_relu, 3x3 conv and leaky_relu.

    Args:
        x (TODO): TODO
        cnum (TODO): TODO

    Returns: TODO

    """
    y = x
    y = tf.layers.conv2d(y, cnum, 3, 1, activation=act, name=name+'_layer1')
    y = tf.layers.conv2d(y, cnum, 3, 1, activation=act, name=name+'_layer2')
    return y


def upsample(x, scale=2):
    """Basic upsample layer.

    Args:
        x (TODO): TODO

    Returns: TODO

    """
    xs = tf.shape(x)
    return tf.image.resize_bilinear(
        x, [scale*xs[1], scale*xs[2]], align_corners=False)


def downsample(x, scale=.5):
    """Basic downsample layer.

    Args:
        x (TODO): TODO

    Returns: TODO

    """
    xs = tf.shape(x)
    return tf.image.resize_bilinear(
        x,
        [tf.cast(scale*xs[1], tf.int32), tf.cast(scale*xs[2], tf.int32)],
        align_corners=False)


def pixel_norm(x):
    """Pixel normalization proposed in progressive GAN.

    Args:
        x (TODO): TODO

    Returns: TODO

    """
    y =  x / tf.sqrt(
        tf.reduce_mean(
            x**2, axis=[1, 2, 3], keep_dims=True) + 1e-8)
    return y


def wscale_layer(x):
    """Applies equalized learning rate to the preceding layer, proposed
    in progressive GAN.

    Args:
        x (TODO): TODO

    Returns: TODO

    """
    y = x
    return y


def progressive_kt(name, steps=10000):
    kt = tf.get_variable(
        name + '_kt', dtype=tf.float32, initializer=0.0, trainable=False)
    scalar_summary('kt/'+name, kt)
    update_kt = tf.assign(kt, tf.clip_by_value(kt + 1. / steps, 0., 1.))
    with tf.control_dependencies([update_kt]):
        return kt
