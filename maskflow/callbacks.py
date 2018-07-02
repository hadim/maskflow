import logging
import os

import tensorflow as tf


class BasicLogger(tf.keras.callbacks.Callback):
    """Callback that prints metrics to stdout.
    # Arguments
        count_mode: One of "steps" or "samples".
            Whether the progress bar should
            count samples seen or steps (batches) seen.
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over an epoch.
            Metrics in this list will be logged as-is.
            All others will be averaged over time (e.g. loss, etc).
    # Raises
        ValueError: In case of invalid `count_mode`.
    """

    def __init__(self, log_on_batch_end=True, count_mode='samples', stateful_metrics=None):
        super().__init__()

        if count_mode == 'samples':
            self.use_steps = False
        elif count_mode == 'steps':
            self.use_steps = True
        else:
            raise ValueError('Unknown `count_mode`: ' + str(count_mode))
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()
            
        self.log_on_batch_end = log_on_batch_end
        self.print_func = print

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        
        self.epoch = epoch
        self.print_func(f'Start epoch {epoch + 1}/{self.epochs}')
        
        if self.use_steps:
            target = self.params['steps']
        else:
            target = self.params['samples']
        self.target = target
        self.seen = 0

    def on_batch_begin(self, batch, logs=None):
        if self.seen < self.target:
            self.log_values = []

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        if self.use_steps:
            self.seen += 1
        else:
            self.seen += batch_size

        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.log_on_batch_end and self.seen < self.target:
            mess = f"Epoch: {self.epoch + 1} | Batch: {self.seen}/{self.target} | {self.log_values}"
            self.print_func(mess)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
                
        mess = f"End epoch {self.epoch+ 1} | Batch: {self.seen}/{self.target} | {self.log_values}"
        self.print_func(mess)


class FileLogger(BasicLogger):
    
    def __init__(self, filename, log_on_batch_end=True, count_mode='samples', stateful_metrics=None):
        super().__init__(log_on_batch_end, count_mode, stateful_metrics)
        
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(logging.DEBUG)

        handler = logging.handlers.RotatingFileHandler(filename, mode="a", maxBytes=10000000, backupCount=1)
        self.logger.addHandler(handler)

        formatter = logging.Formatter("%(asctime)s:%(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        
        self.print_func = self.logger.info


class TelegramLogger(BasicLogger):
    
    def __init__(self, log_on_batch_end=False, count_mode='samples', stateful_metrics=None):
        super().__init__(log_on_batch_end, count_mode, stateful_metrics)
                
        if "TELEGRAM_TOKEN" not in os.environ.keys():
            raise Exception("Set TELEGRAM_TOKEN env variable.")

        if "TELEGRAM_CHAT_ID" not in os.environ.keys():
            raise Exception("Set TELEGRAM_CHAT_ID env variable.")
            
        self.token = os.environ["TELEGRAM_TOKEN"]
        self.chat_id = os.environ["TELEGRAM_CHAT_ID"]

        import telegram
        self.bot = telegram.Bot(token=self.token)
        
        self.print_func = lambda mess: self.bot.send_message(self.chat_id, mess)
