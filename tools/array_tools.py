"""array_tools.py"""

import numpy as np


def sum_nan_arrays(array1: np.array, array2: np.array):
    """
    Sums the elements of af array, ignoring nan unless both elements are nan
    source: https://stackoverflow.com/questions/42209838/treat-nan-as-zero-in-numpy-array-summation-except-for-nan-in-all-arrays
    :param array1: array
    :param array2: array
    :return: array
    """
    ma = np.isnan(array1)
    mb = np.isnan(array2)
    m_keep_a = ~ma & mb
    m_keep_b = ma & ~mb
    out = array1 + array2
    out[m_keep_a] = array1[m_keep_a]
    out[m_keep_b] = array2[m_keep_b]
    return out
