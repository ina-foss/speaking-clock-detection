#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2025 Ina (David Doukhan - http://www.ina.fr/)

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


import os
import multiprocessing as mp
import unittest
from inaudible.speaking_clock_detection import speaking_clock_detection, phase_inversion_detection

testpath = '/rex/store2a/home/ddoukhan/2018_09_16_lucrate_titan/ddoukhan/corpus_horlange_parlante/'
NCPU = mp.cpu_count() // 2

negativefname = ['CAC95021493_pivot_HP_False.mp4', 'CAC95021493_pivot_HP_True.mp4', 'CAC96046817_pivot_false.mp4', 'CAC96046817_pivot_true.mp4', 'CPF86640151_pivot_false.mp4', 'CPF86640151_pivot_true.mp4', 'I16237730_pivot_HP_False.mp4', 'I16237730_pivot_HP_True.mp4', 'I16265123_pivot_HP_False.mp4', 'I16265123_pivot_HP_True.mp4', 'KPCAA761216_VIS_04_NOHP.MP4', 'KPCAA770330_VIS_02_NOHP.MP4', 'KPCAA810423_VIS_06_NOHP.MP4', 'KPCAA820516_VIS_05_NOHP.MP4', 'KPCAB760501_VIS_01_NOHP.MP4', 'KPCAB830611_VIS_02_NOHP.MP4', 'KPCAB840806_VIS_01_NOHP.MP4', 'KPCAB840806_VIS_07_NOHP.MP4', 'KPCAB840807_VIS_02_NOHP.MP4', 'KPCAB840811_VIS_02_NOHP.MP4', 'KPCAB850708_VIS_07_NOHP.MP4', 'KPCAB880508_VIS_07_NOHP.MP4', 'KPCAB890319_VIS_02_NOHP.MP4', 'KPCAB910520_VIS_02_NOHP.MP4', 'KPCAB930527_VIS_04_NOHP.MP4', 'KPCAB940715_VIS_02_NOHP.MP4', 'LXC01049800_pivot_HP_False.mp4', 'LXC01049800_pivot_HP_True.mp4', 'LXC11002321_pivot_HP_False.mp4', 'LXC11002321_pivot_HP_True.mp4', 'MGCAB0011943--AR_VIS_05_NOHP.MP4', '_NOHP_FPVDC08120705_VIS_01.MP4', '_NOHP_FPVDC08120705_VIS_01_NOHP.MP4', '_NOHP_KPRXC-LM900406_VIS_01.MP4', '_NOHP_KPRXC-LM900406_VIS_01_NOHP.MP4']
negativeoutput = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
postivefname = ['CAA8301928101_pivot.mp4', 'CAB8200138901_pivot.mp4', 'CAB8302090001_pivot.mp4', 'CAB85101909_pivot.mp4', 'CAB89029306_pivot.mp4', 'CAC02001913_pivot.mp4', 'I04305437_pivot.mp4', 'I16230638_pivot.mp4', 'I16271187_pivot.mp4', 'I16299493_pivot.mp4', 'KMCPC94063001_VIS_01.MP4', 'KPCAA761216_VIS_04.MP4', 'KPCAA770330_VIS_02.MP4', 'KPCAA770330_VIS_04.MP4', 'KPCAA770330_VIS_04_NOHP.MP4', 'KPCAA810423_VIS_06.MP4', 'KPCAA820516_VIS_05.MP4', 'KPCAB760501_VIS_01.MP4', 'KPCAB830611_VIS_02.MP4', 'KPCAB840806_VIS_01.MP4', 'KPCAB840806_VIS_07.MP4', 'KPCAB840807_VIS_02.MP4', 'KPCAB840811_VIS_02.MP4', 'KPCAB850708_VIS_07.MP4', 'KPCAB860221_EXP_03.MP2', 'KPCAB860603_EXP_03.MP2_HP_MixeCanaux.mp2', 'KPCAB880508_VIS_07.MP4', 'KPCAB890319_VIS_02.MP4', 'KPCAB910520_VIS_02.MP4', 'KPCAB930527_VIS_04.MP4', 'KPCAB940715_VIS_02.MP4', 'KPCPA760608_VIS_01.MP4', 'KPCPA760608_VIS_01_NOHP.MP4', 'MGCAB0011943--AR_VIS_05.MP4', 'MGCPA0019768_VIS_01.MP4', 'MGCPA0019768_VIS_01_NOHP.MP4']
postiveoutput = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1]


def myspeakingclock(x):
    return speaking_clock_detection(x, 'ffmpeg')


def myspeakingclock_short(x):
    # test on 10 mins max
    return speaking_clock_detection(x, 'ffmpeg', end_sec=60*10)


class TestSpeakingClock(unittest.TestCase):
    def test_positive(self):
        p = testpath + 'positive/'
        fl = sorted(os.listdir(p))
        self.assertEqual(fl, postivefname)
        fl = [p + e for e in fl]
        with mp.Pool(NCPU) as p:
            lret = p.map(myspeakingclock, fl)
        self.assertEqual(lret, postiveoutput)

    def test_negative(self):
        p = testpath + 'negative/'
        fl = sorted(os.listdir(p))
        self.assertEqual(fl, negativefname)
        fl = [p + e for e in fl]
        with mp.Pool(NCPU) as p:
            lret = p.map(myspeakingclock, fl)
        self.assertEqual(lret, negativeoutput)

    def test_positive_short(self):
        p = testpath + 'positive/'
        fl = sorted(os.listdir(p))
        self.assertEqual(fl, postivefname)
        fl = [p + e for e in fl]
        with mp.Pool(NCPU) as p:
            lret = p.map(myspeakingclock_short, fl)
        self.assertEqual(lret, postiveoutput)

    def test_negative_short(self):
        p = testpath + 'negative/'
        fl = sorted(os.listdir(p))
        self.assertEqual(fl, negativefname)
        fl = [p + e for e in fl]
        with mp.Pool(NCPU) as p:
            lret = p.map(myspeakingclock_short, fl)
        self.assertEqual(lret, negativeoutput)
        
    def test_phase_inversion(self):
        f = '/rex/store2a/home/sdevauchelle/corpus/diachronique_1980/raw_ts/tv/MGCPB0042023.01.ts'
        self.assertEqual(phase_inversion_detection(f), True)

if __name__ == '__main__':
    unittest.main()
