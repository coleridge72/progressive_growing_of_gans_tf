import logging

import tensorflow as tf
import neuralgym as ng

from neuralgym.ops.summary_ops import scalar_summary
from neuralgym.callbacks import ScheduledCallback, CallbackLoc
from neuralgym.utils.logger import ProgressBar, callback_log


logger = logging.getLogger()


class ProgressiveGrowingGAN(ScheduledCallback):

    """This class is for automatically change training image resolution from
    8x8 to 1024x1024 for progressive growing of gans.

    Rebuild graph for progressive growing of gan.
    """

    def __init__(self, schedule,
                 g_trainer, g_graph_def, g_graph_def_kwargs,
                 d_trainer, d_graph_def, d_graph_def_kwargs):
        super().__init__(CallbackLoc.step_end, schedule)
        self._g_trainer = g_trainer  # primary trainer
        self._g_graph_def = g_graph_def
        self._g_graph_def_kwargs = g_graph_def_kwargs
        self._d_trainer = d_trainer  # secondary trainer
        self._d_graph_def = d_graph_def
        self._d_graph_def_kwargs = d_graph_def_kwargs

    def run(self, sess, step):
        """Automatic run function for callback.

        Args:
            sess (tf.Session): session
            step (int): current training step

        """
        callback_log(
            'Trigger ProgressiveGrowingGAN callback at Step-%d: build '
            'gan with higher resolution of %d X %d.' % (
                step, self.schedule[step], self.schedule[step]))
        resolution = self.schedule[step]
