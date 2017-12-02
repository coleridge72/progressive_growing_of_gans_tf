import logging

import tensorflow as tf
import neuralgym as ng
import numpy as np

from neuralgym.models import Model
from neuralgym.ops.summary_ops import scalar_summary, images_summary
from neuralgym.ops.summary_ops import gradients_summary
from neuralgym.ops.layers import flatten, resize_nearest_neighbor

from progressive_ops import nn_block, act, upsample, downsample, progressive_kt


class ProgressiveGAN(Model):

    """Tensorflow model of progressive gan.
        Args:
            resolution (TODO): TODO
            config (TODO): TODO
    """

    def __init__(self, resolution, config):
        super().__init__('ProgressiveGAN')
        self._resolution = resolution
        self.cfg = config

    def G_paper(self, z, last_resolution, current_resolution, name='G_paper'):
        """Build graph for generator.
        Returns: TODO

        """
        assert last_resolution in [4, 8, 16, 32, 64, 128, 256, 512]
        assert current_resolution == last_resolution * 2
        get_cnum = lambda x: int(min(512, 2 ** (13 - np.log2(x))))

        x = z
        with tf.variable_scope(name, reuse=current_resolution != 8):
            # [-1, 4, 4, 512]
            x = tf.reshape(x, [-1, 1, 1, 512])
            x = tf.layers.conv2d_transpose(
                x, 512, 4, padding="same", activation=act, name='deconv_in')
            x = tf.layers.conv2d(
                x, 512, 3, padding='same', activation=act, name='conv_in')
        block_resolution = 4

        with tf.variable_scope(name, reuse=True):
            for i in range(int(np.log2(current_resolution) - 3)):
                cnum = get_cnum(block_resolution)
                logger.info('Restore block, input resolution: {}, cnum: {}, '
                            'output resolution: {}.'.format(
                                block_resolution, cnum, block_resolution*2))
                x = upsample(x)
                block_resolution *= 2
                x = nn_block(x, cnum, name='block%s' % block_resolution)
            if current_resolution != 8:
                last_x = tf.layers.conv2d(
                    x, 3, 1, padding='same', name='%s_out' % block_resolution)

        with tf.variable_scope(name, reuse=False):
            cnum = get_cnum(block_resolution)
            logger.info('Add block, input resolution: {}, cnum: {}, '
                        'output resolution: {}.'.format(
                            block_resolution, cnum, block_resolution*2))
            x = upsample(x)
            block_resolution *= 2
            x = nn_block(x, cnum, name='block%s' % block_resolution)

            x = tf.layers.conv2d(
                x, 3, 1, padding='same', name='%s_out' % block_resolution)

        if current_resolution != 8:
            kt = progressive_kt('%s_kt' % block_resolution)
            x = kt * x + (1. - kt) * upsample(x)
        return x

    def D_paper(self, x, last_resolution, current_resolution, name='D_paper'):
        """Build graph for discriminator.
        Returns: TODO

        """
        assert last_resolution in [4, 8, 16, 32, 64, 128, 256, 512]
        assert current_resolution == last_resolution * 2
        get_cnum = lambda x: int(min(512, 2 ** (13 - np.log2(x))))

        x_in = x

        block_resolution = current_resolution
        with tf.variable_scope(name, reuse=False):
            # additional layer to be replaced during progressive training
            cnum = get_cnum(block_resolution * 2)
            x = tf.layers.conv2d(
                x, cnum, 3, padding='same', name='%s_in')
            # block
            cnum = get_cnum(block_resolution)
            if current_resolution != 8:
                last_x = tf.layers.conv2d(
                    x, cnum, 1, padding='same', name='%s_in'%block_resolution)
            logger.info('Add block, input resolution: {}, cnum: {}, '
                        'output resolution: {}.'.format(
                            block_resolution, cnum, block_resolution//2))
            block_resolution //= 2
            x = nn_block(x, cnum, name='block%s' % block_resolution)
            current_x = x

        with tf.variable_scope(name, reuse=True):
            if current_resolution != 8:
                cnum = get_cnum(block_resolution * 2)
                x = tf.layers.conv2d(
                    x_in, cnum, 3, padding='same', name='%s_in')
                kt = progressive_kt('%s_kt' % block_resolution)
                x = kt * current_x + (1. - kt) * downsample(x)

            for i in range(int(np.log2(current_resolution) - 3)):
                cnum = get_cnum(block_resolution)
                logger.info('Restore block, input resolution: {}, cnum: {}, '
                            'output resolution: {}.'.format(
                                block_resolution, cnum, block_resolution*2))
                x = nn_block(x, cnum, name='block%s' % block_resolution)
                x = downsample(x)
                block_resolution //= 2

        with tf.variable_scope(name, reuse=current_resolution != 8):
            x = tf.layers.conv2d(
                x, 512, 3, 2, padding='same', activation=act, name='conv_out')
            x = tf.layers.conv2d(
                x, 512, 3, 2, padding="same", activation=act, name='conv_out')
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, 1)
        return x

    def build_graph_with_losses(self, data, config):
        """Build training graph and losses.

        Args:
            data (TODO): TODO
            config (TODO): TODO

        Returns: TODO

        """
