import logging

import tensorflow as tf
import neuralgym as ng
import numpy as np

from neuralgym.models import Model
from neuralgym.ops.summary_ops import scalar_summary, images_summary
from neuralgym.ops.summary_ops import gradients_summary
from neuralgym.ops.layers import resize, avg_pool
from neuralgym.ops.gan_ops import gan_wgan_loss, gradients_penalty
from neuralgym.ops.gan_ops import random_interpolates

from progressive_ops import nn_block, act, progressive_kt


logger = logging.getLogger()


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

    def G_paper(self, z, last_resolution, current_resolution, name='G_paper',
                reuse=False):
        """Build graph for generator.
        Returns: TODO

        """
        assert last_resolution in [4, 8, 16, 32, 64, 128, 256, 512]
        assert current_resolution == last_resolution * 2
        get_cnum = lambda x: int(min(512, 2 ** (13 - np.log2(x))))

        x = z
        # with tf.variable_scope(name, reuse=(reuse or current_resolution!=8)):
        with tf.variable_scope(name, reuse=(reuse or False)):
            # [-1, 4, 4, 512]
            x = tf.reshape(x, [-1, 1, 1, 512])
            x = tf.layers.conv2d_transpose(
                x, 512, 4, 4, padding="same", activation=act, name='deconv_in')
            x = tf.layers.conv2d(
                x, 512, 3, padding='same', activation=act, name='conv_in')
        block_resolution = 4

        # with tf.variable_scope(name, reuse=True):
        with tf.variable_scope(name, reuse=(reuse or False)):
            for i in range(int(np.log2(current_resolution) - 3)):
                cnum = get_cnum(block_resolution)
                logger.info('Restore block, input resolution: {}, cnum: {}, '
                            'output resolution: {}.'.format(
                                block_resolution, cnum, block_resolution*2))
                x = resize(x, 2)
                block_resolution *= 2
                x = nn_block(x, cnum, name='block%s' % block_resolution)
            if current_resolution != 8:
                last_x = tf.layers.conv2d(
                    x, 3, 1, padding='same', name='%s_out' % block_resolution)

        with tf.variable_scope(name, reuse=(reuse or False)):
            cnum = get_cnum(block_resolution)
            logger.info('Add block, input resolution: {}, cnum: {}, '
                        'output resolution: {}.'.format(
                            block_resolution, cnum, block_resolution*2))
            x = resize(x, 2)
            block_resolution *= 2
            x = nn_block(x, cnum, name='block%s' % block_resolution)

            x = tf.layers.conv2d(
                x, 3, 1, padding='same', name='%s_out' % block_resolution)
            kt = progressive_kt('%s_kt' % block_resolution)

        if current_resolution != 8:
            x = kt * x + (1. - kt) * resize(last_x, 2)
        return x

    def D_paper(self, x, last_resolution, current_resolution, reuse=False,
                name='D_paper'):
        """Build graph for discriminator.
        Returns: TODO

        """
        assert last_resolution in [4, 8, 16, 32, 64, 128, 256, 512]
        assert current_resolution == last_resolution * 2
        get_cnum = lambda x: int(min(512, 2 ** (13 - np.log2(x))))

        x_in = x

        block_resolution = current_resolution
        with tf.variable_scope(name, reuse=(reuse or False)):
            # additional layer to be replaced during progressive training
            cnum = get_cnum(block_resolution * 2)
            x = tf.layers.conv2d(
                x, cnum, 3, padding='same', activation=act,
                name='%s_in' % block_resolution)
            # block
            cnum = get_cnum(block_resolution)
            logger.info('Add block, input resolution: {}, cnum: {}, '
                        'output resolution: {}.'.format(
                            block_resolution, cnum, block_resolution//2))
            x = nn_block(x, cnum, name='block%s' % block_resolution)
            x = avg_pool(x)
            current_x = x
            block_resolution //= 2

            kt = progressive_kt('%s_kt' % block_resolution)

        # with tf.variable_scope(name, reuse=True):
        with tf.variable_scope(name, reuse=(reuse or False)):
            if current_resolution != 8:
                cnum = get_cnum(current_resolution)
                x = tf.layers.conv2d(
                    avg_pool(x_in), cnum, 3, padding='same', activation=act,
                    name='%s_in' % block_resolution)
                x = kt * current_x + (1. - kt) * x

            for i in range(int(np.log2(current_resolution) - 3)):
                cnum = get_cnum(block_resolution)
                logger.info('Restore block, input resolution: {}, cnum: {}, '
                            'output resolution: {}.'.format(
                                block_resolution, cnum, block_resolution//2))
                x = nn_block(x, cnum, name='block%s' % block_resolution)
                x = avg_pool(x)
                block_resolution //= 2

        # with tf.variable_scope(name, reuse=(reuse or current_resolution!=8)):
        with tf.variable_scope(name, reuse=(reuse or False)):
            x = tf.layers.conv2d(
                x, 512, 3, 2, padding='same', activation=act, name='conv_out1')
            x = tf.layers.conv2d(
                x, 512, 3, 2, padding="same", activation=act, name='conv_out2')
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, 1, name='fcout')
        return x

    def build_graph_with_losses(self, data, config, reuse=True, summary=False):
        """Build training graph and losses.

        Args:
            data (TODO): TODO
            config (TODO): TODO

        Returns: TODO

        """
        images = data.data_pipeline(config.BATCH_SIZE)
        images = images/127.5 - 1.
        z = tf.random_uniform([config.BATCH_SIZE, 1, 1, 512], -1, 1, name='z')
        fake = self.G_paper(
            z,
            config.LAST_RESOLUTION, config.CURRENT_RESOLUTION,
            reuse=reuse)

        if summary:
            images_summary(images, 'real_images', config.VIZ_MAX_OUT)
            images_summary(fake, 'fake_images', config.VIZ_MAX_OUT)

        neg = self.D_paper(
            fake,
            config.LAST_RESOLUTION, config.CURRENT_RESOLUTION,
            reuse=reuse)
        pos = self.D_paper(
            images,
            config.LAST_RESOLUTION, config.CURRENT_RESOLUTION,
            reuse=True)
        g_loss, d_loss = gan_wgan_loss(pos, neg)

        ri = random_interpolates(images, fake)
        ri_out = self.D_paper(
            ri,
            config.LAST_RESOLUTION, config.CURRENT_RESOLUTION,
            reuse=True)
        ri_loss = gradients_penalty(ri, ri_out)
        d_loss = d_loss + config.LOSS['iwass_lambda'] * ri_loss
        losses = {'g_loss': g_loss, 'd_loss': d_loss, 'ri_loss': ri_loss}

        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'G_paper')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'D_paper')
        return g_vars, d_vars, losses
