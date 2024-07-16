#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
""" Speaking Clock Detection - version 1.2 2017-04-19
Author: David Doukhan <ddoukhan@ina.fr>

The detection of Speaking clock is based on the detection of bips
Bips are 1kHz impulses, of duration variying between 80 and 160 ms
The pattern corresponding to bips is 0 10 20 30 40 57 58 59
Which leads to diff bip pattern of 17, 3*1; 4*10
"""

import sys
import os.path
import argparse
import warnings
import soundfile
import numpy as np

from scipy import fft
from scipy.signal import lfilter
from scipy.signal.windows import hamming
from subprocess import check_output, STDOUT, CalledProcessError

def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis
    into overlapping frames.
    This code has been implemented by Anne Archibald, and has been discussed on the
    ML.

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
def decode_media(infname, tmpdir, ffmpeg='ffmpeg', outsr=4000):
    """
    Decode any media to a numpy array sampled at 'outsr' Hz
    Args:
    * infname: full path to input media
    * outfname: full path to decoded media.
    * ffmpeg: full path to ffmpeg binary.
    * outsr: output sampling rate.
    """

    # check input arguments
    assert os.path.exists(infname), 'input media %s cannot be accessed!' % infname
    assert os.path.exists(tmpdir), 'temp directory %s should exist!' % tmpdir
    # set temp wav file name
    _, tail = os.path.split(infname)
    tmp_wav = '%s/%s.wav' % (tmpdir, tail)
    assert not os.path.exists(tmp_wav), 'Temp Wav %s already exists! Remove it first' % tmp_wav

    # performs media decoding to wav with ffmpeg
    cmd = [ffmpeg, '-i', infname, '-acodec', 'pcm_s16le', '-ar', str(outsr), tmp_wav]
    try:
        check_output(cmd, stderr=STDOUT)
    except CalledProcessError as err:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
        print(err.output, file=sys.stderr)
        raise err

    # decode wav and check wav properties
    wav_data, fs = soundfile.read(tmp_wav)
    os.remove(tmp_wav)
    assert len(wav_data.shape) == 2, 'Input media should be stereo. Easy to fix'
    assert len(wav_data) > 1  # media should not be empty
    assert fs == outsr
    return wav_data


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

def energy_idx(winlen):
    """
    return energy indices corresponding to 1000Hz for a signal sampled at 4KHz
    """
    i1000 = (1000. * winlen) / 4000.
    return tuple([int(e) for e in (i1000-1, i1000, i1000+1)])

def contiguous_regions(booltab):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is duration"""

    # Find the indicies of changes in "condition"
    d = np.diff(booltab)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if booltab[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if booltab[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, booltab.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx[:, 0], idx[:, 1]-idx[:, 0]


def wavdata2bip(wavdata):
    """
    return a list of temporal indices corresponding to the detected bips
    wav signal is assumed to be sampled at 4Kz
    """

    # compute spectrogram with 32 ms windows, and 8 ms step
    win_sec = 0.032
    step_sec = 0.008
    winlen = int(win_sec * 4000)
    step = int(step_sec * 4000)
    nfft = winlen
    spec = my_specgram(wavdata, winlen, step, nfft)

    # bip detection at the frame level
    # candidates should have an energy ratio > 0.5 around the 1000Hz frequency

    energy_1000hz = np.sum(spec[:, energy_idx(winlen)], axis=1)
    energy_all = np.sum(spec, axis=1)
    energy_1000hz[energy_all == 0] = 0
    energy_all[energy_all == 0] = 1
    energy_ratio = energy_1000hz / energy_all

    # get bip candidates
    idx, dur = contiguous_regions(energy_ratio > 0.5)

    # keep candidates having duration between 0.08 and 0.16 seconds
    dur_sec = (dur -1) * step_sec + win_sec
    valid_durs = np.logical_and(dur_sec > 0.08, dur_sec < 0.16)
    idx = idx[valid_durs]
    dur = dur[valid_durs]
    dur_sec = dur_sec[valid_durs]

    if len(idx) == 0:
        return []

    # keep candidates associated to an energy above 20% of the max energy found in candidates
    candidate_energy = np.array([np.mean(energy_all[i:(i+d)]) for i, d in zip(idx, dur)])
    valid_energy = candidate_energy > (0.2 * np.max(candidate_energy))

    idx = idx[valid_energy]
    dur = dur[valid_energy]
    dur_sec = dur_sec[valid_energy]

    return idx * step_sec


def is_bip_pattern(bip_list, dur):
    """
    Tell if a bip list seems to be a speaking clock pattern
    """

    if len(bip_list) == 0: # no bip found
        return False

    # get time interval between bips
    dbip = np.int32(np.round(np.diff(bip_list)))
    # count the amount of valid and invalid time intervals
    nb1 = np.sum(dbip == 1)
    nb10 = np.sum(dbip == 10)
    nb17 = np.sum(dbip == 17)
    nbother = len(dbip) - nb1 - nb10 - nb17

    # in ideal case, there should be 8 bips per minute, this is not systematic
    est_bips = dur / 60. * 8
    # condition for valid bip pattern:
    # * amount of valid bips > 4* amount of invalid bips
    # * 0.8 ideal number of bips < nb bip founds < 1.2 ideal number of bips
    if dur < 60:
        # more tolerance for small durations
        return ((nb1 + nb10 + nb17) > (4 * nbother)) and len(dbip) >= (est_bips * 0.3) and len(dbip) <= (est_bips * 1.5)
    return ((nb1 + nb10 + nb17) > 4 * nbother) and len(dbip) > (est_bips * 0.8) and len(dbip) < (est_bips * 1.2)


def speaking_clock_detection(infname, tmpdir, ffmpeg):
    """
    Returns the number of the channel corresponding to speaking clock
    -1 if speaking clock has not been found
    -2 if speaking clock has been found in more than 1 channel
    """
    # decode media to a 4kHz wav and store it in a numpy array
    wav_data = decode_media(infname, tmpdir, ffmpeg, 4000)
    ret = []

    for i in range(wav_data.shape[1]):
        bips = wavdata2bip(wav_data[:, i])
        if is_bip_pattern(bips, wav_data.shape[0] / 4000.):
            ret.append(i)

    if len(ret) == 0:
        return -1
    elif len(ret) == 1:
        return ret[0]
    if len(ret) > 1:
        return -2

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='''Speaking Clock detection.
    Prints the number of the channel corresponding to the speaking clock
    (0 ... N) preceded by SPEAKING_CLOCK_TRACK.
    If no speaking clock has been found, prints SPEAKING_CLOCK_NONE.
    If speaking clock has been found on several channels, this is likely to
    be an error and the program will print SPEAKING_CLOCK_MULTIPLE.''')

    parser.add_argument('-m', '--media', required=True,
                        help='full path to media to analyze')

    parser.add_argument('-t', '--tmpdir', default='/dev/shm/',
                        help=''' Temporary directory used to store intermediate files. Should
        be a fast access directory such as Ram Disk or SSD hard drive. Default
        value: /dev/shm (linux ram disk)''')

    parser.add_argument('-o', '--output', default=sys.stdout, type=argparse.FileType('w'),
                        help='output file for the result. Default value: /dev/stdout.')

    parser.add_argument('-f', '--ffmpeg', default='ffmpeg',
                        help='''Full path to ffmpeg binary. If not provided, this will used
        default binary installed on the system. This program has been tested
        with ffmpeg version 2.8.8-0ubuntu0.16.04.1''')

    args = parser.parse_args()
    ret = speaking_clock_detection(args.media, args.tmpdir, args.ffmpeg)


    if ret >= 0:
        print('SPEAKING_CLOCK_TRACK', ret, file=args.output)
    elif ret == -1:
        print('SPEAKING_CLOCK_NONE', file=args.output)
    else:
        assert ret == -2
        print('SPEAKING_CLOCK_MULTIPLE', file=args.output)
