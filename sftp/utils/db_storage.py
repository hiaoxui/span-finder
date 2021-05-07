import pickle
import warnings

import h5py
import numpy as np


class Cache:
    def __init__(self, file: str, mode: str = 'a', overwrite=False):
        self.db_file = h5py.File(file, mode=mode)
        self.overwrite = overwrite

    @staticmethod
    def _key(key):
        if isinstance(key, str):
            return key
        elif isinstance(key, list):
            ret = []
            for k in key:
                ret.append(Cache._key(k))
            return ' '.join(ret)
        else:
            return str(key)

    @staticmethod
    def _value(value: np.ndarray):
        if isinstance(value, h5py.Dataset):
            value: np.ndarray = value[()]
        if value.dtype.name.startswith('bytes'):
            value = pickle.loads(value)
        return value

    def __getitem__(self, key):
        key = self._key(key)
        if key not in self:
            raise KeyError
        return self._value(self.db_file[key])

    def __setitem__(self, key, value) -> None:
        key = self._key(key)
        if key in self:
            del self.db_file[key]
        if not isinstance(value, np.ndarray):
            value = np.array(pickle.dumps(value))
        self.db_file[key] = value

    def __delitem__(self, key) -> None:
        key = self._key(key)
        if key in self:
            del self.db_file[key]

    def __len__(self) -> int:
        return len(self.db_file)

    def close(self) -> None:
        self.db_file.close()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __contains__(self, item):
        item = self._key(item)
        return item in self.db_file

    def __enter__(self):
        return self

    def __call__(self, function):
        """
        The object of the class could also be used as a decorator. Provide an additional
        argument `cache_id' when calling the function, and the results will be cached.
        """

        def wrapper(*args, **kwargs):
            if 'cache_id' in kwargs:
                cache_id = kwargs['cache_id']
                del kwargs['cache_id']
                if cache_id in self and not self.overwrite:
                    return self[cache_id]
                rst = function(*args, **kwargs)
                self[cache_id] = rst
                return rst
            else:
                warnings.warn("`cache_id' argument not found. Cache is disabled.")
                return function(*args, **kwargs)

        return wrapper
