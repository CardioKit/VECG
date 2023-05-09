"""
Copyright © 2023 Maximilian Kapsecker

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import sklearn.preprocessing as prepr
import numpy as np
import neurokit2 as nk


class Preprocess:
    def __init__(self, onset: int, offset: int, final_length: int = None, peak='R'):
        self.onset = onset
        self.offset = offset
        self.window_length = self.onset + self.offset
        if final_length:
            self.final_length = final_length
        else:
            self.final_length = self.window_length
        self.peak = peak

    def clean(self, ecg_signal, sampling_rate):
        try:
            clean = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
            return clean
        except:
            return []

    def quality(self, ecg_signal, sampling_rate):
        try:
            quality = nk.ecg_quality(ecg_signal, sampling_rate=sampling_rate, method="zhao2018")
            return quality
        except:
            return []

    def pqrst_peaks(self, ecg_signal, sampling_rate):
        accessor = 'ECG_' + self.peak + '_Peaks'
        try:
            _, waves_peak = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
            if self.peak != 'R':
                _, waves_peak = nk.ecg_delineate(
                    ecg_signal, waves_peak,
                    sampling_rate=sampling_rate,
                    method="peak",
                )
            return {accessor: waves_peak[accessor], 'sampling_rate': 500}
        except Exception as e:
            return {accessor: np.empty(0), 'sampling_rate': 500}

    def preprocess(self, data, return_clean=True):
        X = data.X
        Y = data.Y
        sampling_rate_source = data.SAMPLING_RATE
        ecg_clean = [self.clean(x, sampling_rate_source) for x in X]
        rpeaks = [self.pqrst_peaks(x, sampling_rate_source) for x in ecg_clean]
        # ecg_quality = [self.quality(x, sampling_rate_source) for x in ecg_clean]
        result = []
        labelling = []
        union = []
        qual = []
        for l in range(0, len(ecg_clean)):
            if return_clean:
                series = ecg_clean[l]
            else:
                series = X[l]
            temp = rpeaks[l]['ECG_' + self.peak + '_Peaks'] - self.onset
            temp = temp[temp >= 0]
            temp = temp[(temp + self.window_length) < len(series)]
            for k in temp:
                temp1 = nk.signal.signal_resample(
                    series[k:(k + self.window_length)],
                    desired_length=self.final_length,
                    sampling_rate=sampling_rate_source,
                )
                result.append(temp1)
                union.append(l)
                # qual.append(self.quality(temp1, sampling_rate_source))
            labelling.append(np.repeat(Y[l].reshape(1, Y.shape[1]), len(temp), axis=0))
        result = np.array(result)
        result = prepr.minmax_scale(result, axis=1)
        result = result.reshape(len(result), self.final_length, 1)
        labelling = np.concatenate(labelling)

        return result, labelling, union, qual
