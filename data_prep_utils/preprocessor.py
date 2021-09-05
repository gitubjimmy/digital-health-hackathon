from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_array
import numpy as np

from typing import cast


class StandardYProcessor:
    """
    truncate (convert to nan or given value) y whose event is 0, and min-max scale remaining data.
    (y: time, event)
    """

    def __init__(self, *, feature_range=(-1, 1), default_value=np.nan, copy=True, clip=False):
        self.feature_range = feature_range
        self.default_value = default_value
        self.copy = copy
        self.clip = clip

    def _reset(self):
        if hasattr(self, 'scaler_'):
            del self.scaler_
            del self.mask_
            del self.time_
            del self.event_

    def fit(self, time, event):
        self._reset()
        self.time_ = time
        self.event_ = event
        time = self._validate_data(time)
        event = self._validate_data(event)
        self.mask_ = event != 0
        self.scaler_ = MinMaxScaler(feature_range=self.feature_range, copy=self.copy, clip=self.clip)
        self.scaler_.fit(cast(np.ndarray, time)[self.mask_])
        return self

    def transform(self, time=None, event=None):
        time, event = self._get_new_or_cached_data(time, event)
        mask = event != 0
        time[mask] = self.scaler_.transform(time[mask])
        time[mask == 0] = self.default_value
        return time

    def _get_new_or_cached_data(self, time, event):
        if time is None and event is None:
            time, event = self.time_, self.event_
        elif time is not None and event is not None:
            time, event = self._validate_data(time), self._validate_data(event)
        else:
            raise TypeError("transform() takes no or two arguments (1 given)")
        return time, event

    def fit_transform(self, time, event):
        return self.fit(time, event).transform(time, event)

    @staticmethod
    def _validate_data(data):
        return cast(np.ndarray, check_array(data, accept_sparse=True, force_all_finite="allow-nan"))


class TanhYProcessor(StandardYProcessor):
    """
    tanh processor.
    (y: time, event)
    """

    def __init__(self, *, feature_range=(-2, 2), default_value=1, copy=True, clip=False):  # tanh(2) = 0.964..
        super().__init__(feature_range=feature_range, default_value=default_value, copy=copy, clip=clip)

    def transform(self, time=None, event=None):
        time, event = self._get_new_or_cached_data(time, event)
        mask = event != 0
        time[mask] = np.tanh(self.scaler_.transform(time[mask]))
        time[mask == 0] = self.default_value
        return time


__all__ = ['MinMaxScaler', 'StandardYProcessor', 'TanhYProcessor']
