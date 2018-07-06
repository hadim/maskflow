from pathlib import Path
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

    def __init__(self, log_on_batch_end=True, count_mode='steps', stateful_metrics=None):
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
        self.print(f'Start epoch {epoch}/{self.epochs}')
        
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
            mess = f"Epoch: {self.epoch} | Batch: {self.seen}/{self.target} | {self.log_values}"
            self.print(mess)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
                
        mess = f"End epoch {self.epoch} | Batch: {self.seen}/{self.target} | {self.log_values}"
        self.print(mess)
        
    def print(self, mess):
        try:
            self.print_func(mess)
        except Exception as ex:
            print(f"Error with base logger: {ex}")


class FileLogger(BasicLogger):
    
    def __init__(self, filename, log_on_batch_end=True, count_mode='steps', stateful_metrics=None):
        super().__init__(log_on_batch_end, count_mode, stateful_metrics)
        
        self.logger = logging.getLogger(f"FileLogger_{filename.parent}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        handler = logging.handlers.RotatingFileHandler(filename, mode="a", maxBytes=10000000, backupCount=1)
        self.logger.addHandler(handler)

        formatter = logging.Formatter("%(asctime)s:%(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        
        self.print_func = self.logger.info


class TelegramLogger(BasicLogger):
    
    def __init__(self, log_on_batch_end=False, count_mode='steps', stateful_metrics=None):
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

        
class TrainValTensorBoard(tf.keras.callbacks.TensorBoard):
    
    def __init__(self, log_dir, **kwargs):
        
        self.training_log_dir = Path(log_dir) / "training"
        self.validation_log_dir = Path(log_dir) / "validation"
        
        self.training_log_dir.mkdir(parents=True, exist_ok=True)
        self.validation_log_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(str(self.training_log_dir), **kwargs)

    def set_model(self, model):
        # Setup writer for validation metrics
        self.validation_writer = tf.summary.FileWriter(str(self.validation_log_dir))
        super().set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.validation_writer.add_summary(summary, epoch)
        self.validation_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super().on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        self.validation_writer.close()