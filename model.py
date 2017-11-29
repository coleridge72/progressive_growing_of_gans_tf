import logging

import tensorflow as tf
import neuralgym as ng

from neuralgym.models import Model
from neuralgym.ops.summary_ops import scalar_summary, images_summary
from neuralgym.ops.summary_ops import gradients_summary
from neuralgym.ops.layers import flatten, resize_nearest_neighbor

from ops import gen_conv, gen_deconv, dis_conv


class ProgressiveGAN(Model):

    """Tensorflow model of progressive gan.
        Args:
            num_channels (TODO): TODO
            resolution (TODO): TODO
            label_size (TODO): TODO
            config (TODO): TODO
    """

    def __init__(self, num_channels, resolution, label_size, config):
        super().__init__('ProgressiveGAN')
        self._num_channels = num_channels
        self._resolution = resolution
        self._label_size = label_size
        self.cfg = config

    def G_paper(self, z, name):
        """Build graph for generator.
        Returns: TODO

        """
        x = z
        with tf.variable_scope(name, reuse=True):
            for i in range()

        with tf.variable_scope(name, reuse=False):
            for i in range()
        return x

    def D_paper(self, x):
        """Build graph for discriminator.
        Returns: TODO

        """
        pass

    def build_graph_with_losses(self, data, config):
        """Build training graph and losses.

        Args:
            data (TODO): TODO
            config (TODO): TODO

        Returns: TODO

        """
        kt = tf.get_variable(
            'curr_iter', dtype=tf.int32, initializer=0, trainable=False)
        pass
