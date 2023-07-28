from tkinter import TRUE
from importlib_metadata import SelectableGroups
import tensorflow as tf
#from tensorflow.keras import utils as np_utils
import numpy as np

class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42,augmentFlag=TRUE):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.rotate_inputs = tf.keras.layers.RandomRotation(factor=(-0.1,0.1), seed=seed)
    self.rotate_labels = tf.keras.layers.RandomRotation(factor=(-0.1,0.1), seed=seed)
    self.translate_inputs = tf.keras.layers.RandomTranslation(height_factor=(-0.1,0.1),width_factor=(-0.1,0.1), seed=seed)
    self.translate_labels = tf.keras.layers.RandomTranslation(height_factor=(-0.1,0.1),width_factor=(-0.1,0.1), seed=seed)
    self.augmentFlag = augmentFlag
  def call(self, inputs, labels):
    if self.augmentFlag == False:
        return inputs, labels
    inputs = self.rotate_inputs(inputs)
    labels = self.rotate_labels(labels)
    inputs = self.translate_inputs(inputs)
    labels = self.translate_labels(labels)
    return inputs, labels

class anatomicalview():
    def __init__(self,string,max_slices,out_ch):
        self.string = string
        self.max_slices = max_slices
        self.out_ch = out_ch
        
def make_callbacks(weight_name, history_name, epoch_size,batch_size, view_string, monitor='val_loss', patience=20, mode='min', save_weights_only=True,SGR_schedule=False, warm_restarts = False,min_lr=0.00001,max_lr=0.001, use_tensorboard=False):
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from tensorflow.keras.callbacks import TerminateOnNaN, ReduceLROnPlateau
    import six, io, time, csv, numpy as np, json, warnings
    from collections import deque
    from collections import OrderedDict
    from collections import Iterable
    from collections import defaultdict
    from tensorflow import keras
    from tensorflow.keras.utils import Progbar
    from tensorflow.keras import backend as K
    from tensorflow.python.keras.engine.training_utils_v1 import standardize_input_data
    class CSVLogger_time_SGDR(Callback):
        """Callback that streams epoch results to a csv file.
                Supports all values that can be represented as a string,
                including 1D iterables such as np.ndarray.
                # Example
                ```python
                csv_logger = CSVLogger('training.log')
                model.fit(X_train, Y_train, callbacks=[csv_logger])
                ```
                # Arguments
                    filename: filename of the csv file, e.g. 'run/log.csv'.
                    separator: string used to separate elements in the csv file.
                    append: True: append if file exists (useful for continuing
                        training). False: overwrite existing file,
                """
        def __init__(self, 
                    filename, 
                    min_lr,
                    max_lr,
                    steps_per_epoch,
                    lr_decay=1,
                    cycle_length=10,
                    mult_factor=2,
                    separator=',', 
                    append=False,):

            self.min_lr = min_lr
            self.max_lr = max_lr
            self.lr_decay = lr_decay
            self.batch_since_restart = 0
            self.next_restart = cycle_length
            self.steps_per_epoch = steps_per_epoch
            self.cycle_length = cycle_length
            self.mult_factor = mult_factor
            self.history = {}
            self.sep = separator
            self.filename = filename
            self.append = append
            self.writer = None
            self.keys = None
            self.append_header = True

            if six.PY2:
                self.file_flags = 'b'
                self._open_args = {}
            else:
                self.file_flags = ''
                self._open_args = {'newline': '\n'}
            super(CSVLogger_time_SGDR, self).__init__()

        def clr(self):
            '''Calculate the learning rate.'''
            fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
            return lr

        def on_train_begin(self, logs=None):
            import os
            if self.append:
                if os.path.exists(self.filename):
                    with open(self.filename, 'r' + self.file_flags) as f:
                        self.append_header = not bool(len(f.readline()))
                mode = 'a'
            else:
                mode = 'w'
            self.csv_file = io.open(self.filename,
                                    mode + self.file_flags,
                                    **self._open_args)
            '''Initialize the learning rate to the maximum value at the start of training.'''
            K.set_value(self.model.optimizer.lr, self.max_lr)

        def on_batch_end(self, batch, logs={}):
            '''Record previous batch statistics and update the learning rate.'''
            logs = logs or {}
            self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)
            self.batch_since_restart += 1
            K.set_value(self.model.optimizer.lr, self.clr())

        def on_epoch_begin(self, epoch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}

            def handle_value(k):
                is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
                if isinstance(k, six.string_types):
                    return k
                elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                    return '"[%s]"' % (', '.join(map(str, k)))
                else:
                    return k

            if self.keys is None:
                self.keys = sorted(logs.keys())

            if self.model.stop_training:
                # We set NA so that csv parsers do not fail for this last epoch.
                logs = dict([(k, logs[k] if k in logs else 'NA') for k in self.keys])

            if not self.writer:
                class CustomDialect(csv.excel):
                    delimiter = self.sep
                fieldnames = ['epoch'] + self.keys +['time'] + ['lr'] 
                if six.PY2:
                    fieldnames = [str(x) for x in fieldnames]
                self.writer = csv.DictWriter(self.csv_file,
                                             fieldnames=fieldnames,
                                             dialect=CustomDialect)
                if self.append_header:
                    self.writer.writeheader()
            '''Check for end of current cycle, apply restarts when necessary.'''
            if epoch + 1 == self.next_restart:
                self.batch_since_restart = 0
                self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
                self.next_restart += self.cycle_length
                
                ##Descent
                #self.max_lr *= self.lr_decay
                
                ##Use warm restart
                self.max_lr = self.max_lr
                
                self.best_weights = self.model.get_weights()

            row_dict = OrderedDict({'epoch': epoch})
            logs['time']=time.time() - self.epoch_time_start
            self.keys.append('time')
            #logs['lr']=float(self.model.optimizer.lr)
            #self.keys.append('time')
            #self.keys.append('lr')
            row_dict.update((key, handle_value(logs[key])) for key in self.keys)
            self.writer.writerow(row_dict)
            self.csv_file.flush()
        def on_train_end(self, logs=None):
            self.csv_file.close()
            self.writer = None
            self.model.set_weights(self.best_weights)

        def __del__(self):
            if hasattr(self, 'csv_file') and not self.csv_file.closed:
                self.csv_file.close()
                    
    class CSVLogger_time(Callback):
        """Callback that streams epoch results to a csv file.
        Supports all values that can be represented as a string,
        including 1D iterables such as np.ndarray.
        # Example
        ```python
        csv_logger = CSVLogger('training.log')
        model.fit(X_train, Y_train, callbacks=[csv_logger])
        ```
        # Arguments
            filename: filename of the csv file, e.g. 'run/log.csv'.
            separator: string used to separate elements in the csv file.
            append: True: append if file exists (useful for continuing
                training). False: overwrite existing file,
        """

        def __init__(self, filename, separator=',', append=False):
            self.sep = separator
            self.filename = filename
            self.append = append
            self.writer = None
            self.keys = None
            self.append_header = True
            if six.PY2:
                self.file_flags = 'b'
                self._open_args = {}
            else:
                self.file_flags = ''
                self._open_args = {'newline': '\n'}
            super(CSVLogger_time, self).__init__()

        def on_train_begin(self, logs=None):
            import os
            if self.append:
                if os.path.exists(self.filename):
                    with open(self.filename, 'r' + self.file_flags) as f:
                        self.append_header = not bool(len(f.readline()))
                mode = 'a'
            else:
                mode = 'w'
            self.csv_file = io.open(self.filename,
                                    mode + self.file_flags,
                                    **self._open_args)

        def on_epoch_begin(self, epoch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}

            def handle_value(k):
                is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
                if isinstance(k, six.string_types):
                    return k
                elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                    return '"[%s]"' % (', '.join(map(str, k)))
                else:
                    return k

            if self.keys is None:
                self.keys = sorted(logs.keys())

            if self.model.stop_training:
                # We set NA so that csv parsers do not fail for this last epoch.
                logs = dict([(k, logs[k] if k in logs else 'NA') for k in self.keys])

            if not self.writer:
                class CustomDialect(csv.excel):
                    delimiter = self.sep
                fieldnames = ['epoch'] + self.keys +['time']
                if six.PY2:
                    fieldnames = [str(x) for x in fieldnames]
                self.writer = csv.DictWriter(self.csv_file,
                                             fieldnames=fieldnames,
                                             dialect=CustomDialect)
                if self.append_header:
                    self.writer.writeheader()

            row_dict = OrderedDict({'epoch': epoch})
            logs['time']=time.time() - self.epoch_time_start
            self.keys.append('time')
            row_dict.update((key, handle_value(logs[key])) for key in self.keys)
            self.writer.writerow(row_dict)
            self.csv_file.flush()

        def on_train_end(self, logs=None):
            self.csv_file.close()
            self.writer = None

        def __del__(self):
            if hasattr(self, 'csv_file') and not self.csv_file.closed:
                self.csv_file.close()
    if 'loss' in monitor:
        mode = 'min'
    else:
        mode = 'max'
    earlystop=EarlyStopping(monitor=monitor, patience=patience, verbose=0, mode=mode)
    checkpoint=ModelCheckpoint(filepath=weight_name, monitor=monitor, mode=mode, save_best_only=True, save_weights_only=save_weights_only, verbose=0)
    if SGR_schedule==True:
        csvlog=CSVLogger_time_SGDR(history_name, 
                        min_lr=min_lr,
                        max_lr=max_lr,
                        steps_per_epoch=np.ceil(epoch_size/batch_size),
                        lr_decay=1.0,
                        cycle_length=15,
                        mult_factor=1.5,
                        separator='\t',append=True)
    else:
        csvlog=CSVLogger_time(history_name, separator='\t')
        print('no schedule')
    term = TerminateOnNaN()
    reduce = ReduceLROnPlateau(monitor=monitor)
    if use_tensorboard:
        from tensorflow.keras.callbacks import TensorBoard
        import os
        tensorboard_callback = TensorBoard(log_dir=os.path.dirname(history_name)+str('/')+view_string,
                         update_freq='epoch')
        return [earlystop, checkpoint, csvlog, term,tensorboard_callback,reduce]
    else:
        return [earlystop, checkpoint, csvlog, term,reduce]

from tensorflow.keras.callbacks import Callback    
class LRFinder(Callback):
    '''
    A simple callback for finding the optimal learning rate range for your model + dataset. 
    
    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5, 
                                 max_lr=1e-2, 
                                 steps_per_epoch=np.ceil(epoch_size/batch_size), 
                                 epochs=3)
            model.fit(X_train, Y_train, callbacks=[lr_finder])
            
            lr_finder.plot_loss()
        ```
    
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient. 
        
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: https://arxiv.org/abs/1506.01186
    '''

    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        x = self.iteration / self.total_iterations 
        return self.min_lr + (self.max_lr-self.min_lr) * x

    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the maximum value at the start of training.'''
        from tensorflow.keras import backend as K

        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)

    def on_batch_end(self, epoch, logs=None):
        from tensorflow.keras import backend as K

        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())
    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        import matplotlib.pyplot as plt
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        plt.show()
        
    def plot_loss(self):
        '''Helper function to quickly observe the learning rate experiment results.'''
        import matplotlib.pyplot as plt
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.show()