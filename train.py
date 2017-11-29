import logging
import numpy as np
import neuralgym as ng

from neuralgym.callbacks import DiscriminatorMultiGPUTrainer
import network


logger = logging.getLogger()


def train_gan(config):
    """main function for training progressive gan

    Args:
        config (ng.Config): all hyperparameters

    Returns:
    """
    logger.info('Start to train progressive gan.')
    # get dataset
    with open(config.DATA_FLIST[config.DATASET][0]) as f:
        fnames = f.read().splitlines()
    data = ng.data.DataFromFNames(
        fnames, config.IMG_SHAPES, random_crop=config.RANDOM_CROP)

    # init networks
    model = model.ProgressiveGAN(
        num_channels=None, resolution=None,
        label_size=None, config)
    g_vars, d_vars, losses = model.build_graph_with_losses(data, config)

    g_optimizer = tf.train.AdamOptimizer(
        lr,
        beta1=config.TRAIN['adam_beta1'],
        beta2=config.TRAIN['adam_beta2'],
        epsilon=config.TRAIN['adam_epsilon'])
    d_optimizer = g_optimizer

    discriminator_training_callback = DiscriminatorMultiGPUTrainer(
        1, config.NUM_GPUS, multigpu_graph_def,
        {'model': model, 'data': data, 'config': config, 'loss_type': 'd',
         'summary': False},
        d_optimizer, d_vars, steps=config.TRAIN['D_training_repeats'])

    log_prefix = 'model_logs/' + '_'.join([
        str(ng.date_uid()),socket.gethostname(), config.DATASET,
        config.LOG_DIR])

    trainer = MultiGPUTrainer(
        config=config,
        optimizer=g_optimizer,
        var_list=g_vars,
        gpu_num=config.NUM_GPUS,
        async_train=True,
        graph_def=multigpu_graph_def,
        grads_summary=config.GRADS_SUMMARY,
        gradient_processor=gradient_processor,
        graph_def_kwargs={
            'model': model, 'data': data, 'config': config, 'loss_type': 'g'},
        spe=config.TRAIN_SPE,
        max_iters=config.MAX_ITERS,
        log_dir=log_prefix,
    )

    trainer.add_callbacks([
        ng.callbacks.WeightsViewer(),
        # ng.callbacks.ModelRestorer(trainer.context['saver'], dump_prefix='', optimistic=True),
        discriminator_training_callback,
        ng.callbacks.ModelSaver(config.TRAIN_SPE, trainer.context['saver'], log_prefix+'/snap'),
        ng.callbacks.SummaryWriter((config.VAL_PSTEPS//1), trainer.context['summary_writer'], tf.summary.merge_all()),
    ])

    trainer.train()


def multigpu_graph_def(gpu_id, model, data, config, loss_type='g'):
    with tf.device('/cpu:0'):
        images = data.data_pipeline(config.BATCH_SIZE)

    if summary and gpu_id == 0:
        _, _, losses = model.build_graph_with_losses(
            images, config, summary=True)
    else:
        _, _, losses = model.build_graph_with_losses(
            images, config)
    if loss_type == 'g':
        return losses['g_loss']
    elif loss_type == 'd':
        return losses['d_loss']
    else:
        raise ValueError('loss type is not supported.')


if __name__ == "__main__":
    config = ng.Config('progressive_gan.yml')
    if config.GPU_ID != -1:
        ng.set_gpus(config.GPU_ID)
    else:
        ng.get_gpus(config.NUM_GPUS)
    np.random.seed(config.RANDOM_SEED)

    eval(config.TRAIN.func + '(config)')
