import logging
import numpy as np
import neuralgym as ng

import progressive_model


logger = logging.getLogger()


def multigpu_graph_def(model, data, config, gpu_id=0, loss_type='g'):
    if gpu_id == 0:
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
        fnames, config.IMG_SHAPES)

    # init model
    model = progressive_model.ProgressiveGAN(1024, config)
    from IPython import embed; embed()
    g_vars, d_vars, losses = model.build_graph_with_losses(data, config)

    g_optimizer = tf.train.AdamOptimizer(
        lr,
        beta1=config.TRAIN['adam_beta1'],
        beta2=config.TRAIN['adam_beta2'],
        epsilon=config.TRAIN['adam_epsilon'])
    d_optimizer = g_optimizer

    discriminator_training_callback = ng.callbacks.SecondaryTrainer(
        pstep=1,
        optimizer=d_optimizer,
        var_list=d_vars,
        max_iters=config.TRAIN['D_training_repeats'],
        graph_def=multigpu_graph_def,
        graph_def_kwargs={
            'model': model, 'data': data, 'config': config, 'loss_type': 'd'},
        )

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


if __name__ == "__main__":
    config = ng.Config('progressive_gan.yml')
    if config.GPU_ID != -1:
        ng.set_gpus(config.GPU_ID)
    else:
        ng.get_gpus(config.NUM_GPUS)
    np.random.seed(config.RANDOM_SEED)

    eval(config.TRAIN.func + '(config)')
