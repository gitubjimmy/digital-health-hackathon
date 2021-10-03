"""src: https://github.com/kdha0727/easyrun-pytorch/blob/main/easyrun.py"""
# Separate basic class of trainer, for readability.

import contextlib
import time
import functools

import torch
import torch.nn.functional
import torch.utils.data


@functools.lru_cache(maxsize=None)  # Remember prior inputs
def get_loader_information(loader):
    batch_size = getattr(loader, 'batch_size', 1)
    loader_length = len(loader)
    dataset_length = len(getattr(loader, 'dataset', loader))
    return batch_size, loader_length, dataset_length


class CheckpointMixin(object):

    _closed = True

    #
    # De-constructor: executed in buffer-cleaning in python exit
    #

    def __del__(self):
        self._close()

    #
    # Context manager magic methods
    #

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_info, exc_class, exc_traceback):
        try:
            self._close()
        except Exception as exc:
            if (exc_info or exc_class or exc_traceback) is not None:
                pass  # executed in exception handling - just let python raise that exception
            else:
                raise exc

    def _require_context(self):
        if self._closed:
            raise ValueError('Already closed: %r' % self)

    @contextlib.contextmanager
    def _with_context(self):
        prev = True
        try:
            prev = self._open()
            yield
        finally:
            self._close(prev)

    def _open(self, *args, **kwargs):
        raise NotImplementedError

    def _close(self, *args, **kwargs):
        raise NotImplementedError


class MovableMixin(object):

    __to_parse = (None, None, False, None)

    #
    # Device-moving Methods
    #

    def to(self, *args, **kwargs):  # overwrite this in subclass, for further features
        self._to_set(*args, **kwargs)
        return self

    # Internal Device-moving Methods

    def _to_set(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)  # noqa
        device = device or self.__to_parse[0]
        dtype = dtype or self.__to_parse[1]
        non_blocking = non_blocking or self.__to_parse[2]
        convert_to_format = convert_to_format or self.__to_parse[3]
        self.__to_parse = (device, dtype, non_blocking, convert_to_format)

    def _to_apply_module(self, v):
        device, dtype, non_blocking, convert_to_format = self.__to_parse
        return v.to(device, dtype, non_blocking, memory_format=convert_to_format)

    def _to_apply_tensor(self, v):
        device, dtype, _, convert_to_format = self.__to_parse
        return v.to(device, dtype, memory_format=convert_to_format)

    def _to_apply_multi_tensor(self, *v):
        return tuple(map(self._to_apply_tensor, v))


class TimerLoggerMixin(object):

    train_iter: ...
    total_epoch: int
    verbose: bool
    use_timer: bool
    progress: bool
    save_and_load: bool
    _best_loss: float

    _time_start = None
    _time_stop = None

    # Internal Timing Functions

    def _timer_start(self):
        if self.use_timer:
            self._time_start = time.time()

    def _timer_stop(self):
        if self.use_timer:
            self._time_stop = time.time()

    # Internal Logging Methods

    def _log_start(self):
        if self.verbose:
            log = f"\n<Start Learning> "
            if self.total_epoch is not None:
                log += f"\t\t\t\tTotal {self.total_epoch} epochs"
            self.log_function(log)

    def _log_step(self, epoch: int):
        if self.verbose:
            if self.progress:
                self.log_function(f'\nEpoch {epoch}')
            else:
                self.log_function(f'Epoch {epoch}', end=' ')

    def _log_train_doing(self, loss, iteration, whole=None):
        if self.verbose and self.progress:
            if isinstance(whole, int):
                batch_size = 1
                loader_length = dataset_length = whole
            else:
                batch_size, loader_length, dataset_length = get_loader_information(whole or self.train_iter)
            self.log_function(
                f'\r[Train]\t '
                f'Progress: {iteration * batch_size}/{dataset_length} '
                f'({100. * iteration / loader_length:05.2f}%), \tLoss: {loss:.6f}',
                end=' '
            )

    def _log_train_done(self, loss, whole=None):
        if self.verbose:
            if isinstance(whole, int):
                dataset_length = whole
            else:
                _, _, dataset_length = get_loader_information(whole or self.train_iter)
            if self.progress:
                log = f'\r[Train]\t Progress: {dataset_length}/{dataset_length} (100.00%), \t'
            else:
                log = f'[Train]\t '
            log += f'Average Loss: {loss:.6f}'
            if self.progress:
                self.log_function(log)
            else:
                self.log_function(log, end='\t ')

    def _log_eval(self, loss, test=False):
        if self.verbose:
            log = '\n[Test]\t ' if test else '[Eval]\t '
            log += f'Average loss: {loss:.6f}, '
            if self.use_timer:
                log += "\tTime Elapsed: "
                duration = time.time() - self._time_start
                if duration > 60:
                    log += f"{int(duration // 60):02}m {duration % 60:05.2f}s"
                else:
                    log += f"{duration:05.2f}s"
            self.log_function(log)

    def _log_stop(self):
        if self.verbose:
            log = "\n<Stop Learning> "
            if self.save_and_load:
                log += f"\tLeast loss: {self._best_loss:.4f}"
            if self.use_timer:
                log += "\tDuration: "
                duration = self._time_stop - self._time_start
                if duration > 60:
                    log += f"{int(duration // 60):02}m {duration % 60:05.2f}s"
                else:
                    log += f"{duration:05.2f}s"
            self.log_function(log)

    # Log function: overwrite this to use custom logging hook
    log_function = staticmethod(print)


class TrainerMixin(CheckpointMixin, MovableMixin, TimerLoggerMixin):

    @staticmethod
    def _check_criterion(criterion):
        if not callable(criterion):  # allow string: convert string to callable function or module
            assert isinstance(criterion, str), \
                "Invalid criterion type: %s" % criterion.__class__.__name__
            assert (hasattr(torch.nn, criterion) or hasattr(torch.nn.functional, criterion)), \
                "Invalid criterion string: %s" % criterion
            criterion = getattr(torch.nn.functional, criterion, getattr(torch.nn, criterion)())
        return criterion

    @staticmethod
    def _check_data(dataset_or_loader):
        if dataset_or_loader is None:
            return
        assert isinstance(dataset_or_loader, (torch.utils.data.Dataset, torch.utils.data.DataLoader)), \
            "Invalid test_iter type: %s" % dataset_or_loader.__class__.__name__
        return dataset_or_loader

    _open: ...
    _close: ...
