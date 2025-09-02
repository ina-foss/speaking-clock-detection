# Copyright (c) 2008 Cournapeau David

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Note : this code is derived from scikits.talkbox project
# https://github.com/cournape/talkbox
# original project is no more maintained and not compatible with python3

import numpy as np
from scipy import fft
from scipy.signal import lfilter
from scipy.signal.windows import hamming
import warnings

def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis
    into overlapping frames.
    This code has been implemented by Anne Archibald, and has been discussed on the
    ML.
    This function has been taken and adapted from the scikits.talkbox module
    by Cournapeau David:
    https://github.com/cournape/talkbox/blob/ee0ec30a6a6d483eb9284f72bdaf26bd99765f80/scikits/talkbox/tools/segmentaxis.py#L9
    example:
    >>> segment_axis(arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

    arguments:
    a       The array to segment
    length  The length of each frame
    overlap The number of array elements by which the frames should overlap
    axis    The axis to operate on; if None, act on the flattened array
    end     What to do with the last frame, if the array is not evenly
            divisible into pieces. Options are:

            'cut'   Simply discard the extra values
            'wrap'  Copy values from the beginning of the array
            'pad'   Pad with a constant value

    endvalue    The value to use for end='pad'

    The array is not copied unless necessary (either because it is unevenly
    strided and being flattened or because end is set to 'pad' or 'wrap').
    """

    if axis is None:
        a = np.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]

    if overlap >= length:
        raise ValueError("frames cannot overlap by more than 100%")
    if overlap < 0 or length <= 0:
        raise ValueError("overlap must be nonnegative and length must "\
                         "be positive")

    if l < length or (l-length) % (length-overlap):
        if l>length:
            roundup = length + (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown < l < roundup
        assert roundup == rounddown + (length-overlap) \
            or (roundup == length and rounddown == 0)
        a = a.swapaxes(-1,axis)

        if end == 'cut':
            a = a[..., :rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1] = roundup
            b = np.empty(s,dtype=a.dtype)
            b[..., :l] = a
            if end == 'pad':
                b[..., l:] = endvalue
            elif end == 'wrap':
                b[..., l:] = a[..., :roundup-l]
            a = b

        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l == 0:
        raise ValueError(
            "Not enough data points to segment array in 'cut' mode; "\
            "try 'pad' or 'wrap'"
        )
    assert l >= length
    assert (l-length) % (length-overlap) == 0
    n = 1 + (l-length) // (length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n,length) + a.shape[axis+1:]
    newstrides = a.strides[:axis] + ((length-overlap)*s,s) + a.strides[axis+1:]

    try:
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length-overlap)*s,s) \
            + a.strides[axis+1:]
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)


def preemp(input, p):
    """Pre-emphasis filter."""
    return lfilter([1., -p], 1, input)

def my_specgram(data, winlen=256, steplen=256, nfft=512):
    """ Custom spectrogram (256 = 16ms for 16k signal) """
    preemp_fact = 0.97
    data = preemp(data, preemp_fact)
    w = hamming(winlen, sym=0)
    framed = segment_axis(data, winlen, winlen-steplen) * w
    ret = np.abs(fft.fft(framed, nfft, axis=-1))
    ret = ret[:, 0:(ret.shape[1] // 2)]
    return ret
