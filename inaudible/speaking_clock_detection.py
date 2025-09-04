#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
""" Speaking Clock Detection - version 1.3 2025-09-02
Author: David Doukhan <ddoukhan@ina.fr>

The detection of Speaking clock is based on the detection of bips
Bips are 1kHz impulses, of duration variying between 80 and 160 ms
The pattern corresponding to bips is 0 10 20 30 40 57 58 59
Which leads to diff bip pattern of 17, 3*1; 4*10
"""

import numpy as np
import soundfile
import subprocess
from tempfile import TemporaryFile#, NamedTemporaryFile
from .scikits_talkbox import my_specgram
import warnings


class TmpWavDecoder:
    def __init__(self, ffmpeg='ffmpeg', outsr=4000, start_sec = None, end_sec=None):
        self.ffmpeg = ffmpeg
        self.outsr = outsr
        self.start_sec = start_sec
        self.end_sec = end_sec

    def __call__(self, infname):
        #infname can be a path or a url
        cmd = [self.ffmpeg, '-i', infname, '-f', 'wav', '-acodec', 'pcm_s16le']
        if self.outsr is not None:
            cmd += ['-ar', str(self.outsr)]
        if self.start_sec is not None:
            cmd += ['-ss', '%f' % self.start_sec]
        if self.end_sec is not None:
            cmd += ['-to', '%f' % self.end_sec]
        cmd += ['pipe:1']
        with TemporaryFile() as out, TemporaryFile() as err:
            ret = subprocess.run(cmd, stdout=out, stderr=err)
            if ret.returncode != 0:
                err.seek(0)
                msg = err.read()
                raise Exception(msg)
            out.seek(0)
            wav_data, fs = soundfile.read(out)
        assert(fs == self.outsr)
        if len(wav_data.shape) == 1:
            wav_data = np.expand_dims(wav_data, axis=1)
        return wav_data


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

    # get bip candidates
    idx, dur = contiguous_regions(energy_ratio > 0.5)

    # keep candidates having duration between 0.08 and 0.16 seconds
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


def speaking_clock_detection(infname, ffmpeg, end_sec=None):
    """
    Returns the number of the channel corresponding to speaking clock
    -1 if speaking clock has not been found
    -2 if speaking clock has been found in more than 1 channel
    """
    # decode media to a 4kHz wav and store it in a numpy array
    twd = TmpWavDecoder(ffmpeg=ffmpeg, outsr=4000, end_sec=end_sec)
    wav_data = twd(infname) 
    
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



def phase_inversion_detection(infname, ffmpeg='ffmpeg', start_sec=None, end_sec=120):
    twd = TmpWavDecoder(ffmpeg=ffmpeg, start_sec=start_sec, end_sec=end_sec)
    wav_data = twd(infname)
    _, nbchan = wav_data.shape
    if nbchan == 1:
        return False
    if nbchan > 2:
        # TODO : implement something intelligent
        warnings.warn('file %s has more than 2 tracks' % infname)
        return False
    l = wav_data[:, 0]
    r = wav_data[:, 1]
    lrms = np.sqrt(np.mean(l * l))
    rrms = np.sqrt(np.mean(r * r))
    db = 10 * np.log10(lrms / rrms)
    if abs(db) > 10:
        # a track is stronger than the other one, do not compute phase inversion
        warnings.warn('empty track in file ' + infname)
        return False    
    
    correlation = np.sum(l * r)
    
    if correlation < 0:
        return True
    return False


class WavExtractor:
    def __init__(self, detect_clock=False, final_sr=16000, detect_clock_dur=None, detect_phase_dur=120):
        self.detect_clock = detect_clock
        if detect_clock_dur is not None:
            assert(detect_clock == True)
        self.detect_clock_dur = detect_clock_dur
        self.final_sr = final_sr
        self.detect_phase_dur = detect_phase_dur
    def __call__(self, src, dst):
        cmd = ['ffmpeg', '-i', src, '-ar', str(self.final_sr), '-f', 'wav', '-acodec', 'pcm_s16le']
        scd = -1
        dret = {}
        if self.detect_clock:
            scd = speaking_clock_detection(src, 'ffmpeg', self.detect_clock_dur)
            dret['speaking_clock'] = scd
        if scd == 0:
            cmd += ['-filter_complex',  '[0:a]channelsplit=channel_layout=stereo:channels=FR[right]', '-map', '[right]']
        elif scd == 1:
            cmd += ['-filter_complex',  '[0:a]channelsplit=channel_layout=stereo:channels=FL[left]', '-map', '[left]']
        else:
            dret['phase_inversion'] = phase_inversion_detection(src, end_sec=self.detect_phase_dur)
            if dret['phase_inversion']:
                cmd += ['-filter_complex', '[0:a]channelsplit=channel_layout=stereo[left][right]; [left]volume=1[left]; [right]volume=-1[right]; [left][right]amix=inputs=2']
            cmd += ['-ac', '1']
        cmd += [dst]
        ret = subprocess.run(cmd, capture_output=True)
        dret['return_code'] = ret.returncode
        if ret.returncode:
            dret['errmsg'] = ret.stderr.decode('utf-8')
        return dret


