import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymavlog import MavLog
from scipy import signal
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

class ArduPilotFilterReviewer:
    # constants
    MAX_NUM_HARMONICS = 16

    def __init__(self, mavlog, notch_freq, notch_bandwith, notch_att, filter_version:int=2, tune:bool=False, autotune:bool=False):
        self.mavlog         = mavlog
        self.notch_freq     = notch_freq
        self.notch_bandwith = notch_bandwith 
        self.notch_att      = notch_att
        self.filter_version = filter_version
        self.tune           = tune
        self.autotune       = autotune

        # build params dict from log
        params           = pd.DataFrame(self.mavlog.get('PARM').fields)
        self.params_dict = dict(zip(params['Name'].values, params['Value'].values))

    def plot_filter_review(self, target_instance: int = 0):
        gyro_data   = self._parse_gyro(target_instance)
        window_size = int(self.params_dict['INS_LOG_BAT_CNT'])
        rate        = gyro_data['sample_rate']

        bins, fft_result, time = self._run_batch_fft(gyro_data, window_size)

        if self.autotune:
            H_                              = self._calculate_transfer_function(bins, time, rate)
            pre_x_, pre_y_, pre_z_, _, _, _ = self._estimate_pre_post(fft_result, H_, window_size, rate)
            self.motor_freq                 = self._detect_motor_frequency(bins, pre_x_, pre_y_, pre_z_)
            if self.motor_freq is not None:
                self.notch_freq = self.motor_freq
                self.tune       = True  # so _load_filters applies the override
            else:
                self.motor_freq = None

        H = self._calculate_transfer_function(bins, time, rate)

        pre_x, pre_y, pre_z, post_x, post_y, post_z = self._estimate_pre_post(fft_result, H, window_size, rate)

        self._plot(bins, pre_x, pre_y, pre_z, post_x, post_y, post_z)

    def _parse_gyro(self, target_instance: int):
        isbh     = pd.DataFrame(self.mavlog.get('ISBH').fields)
        isbd_raw = self.mavlog.get('ISBD').fields

        isbd_list = []
        for i in range(len(isbd_raw['N'])):
            isbd_list.append({
                'N':     isbd_raw['N'][i],
                'seqno': isbd_raw['seqno'][i],
                'x':     isbd_raw['x'][i],
                'y':     isbd_raw['y'][i],
                'z':     isbd_raw['z'][i],
            })
        isbd_df = pd.DataFrame(isbd_list)

        isbh_gyro = isbh[(isbh['type'] == 1) & (isbh['instance'] == target_instance)]

        x_all, y_all, z_all = [], [], []
        sample_time = None
        sample_rate = None
        mul         = None

        for _, header in isbh_gyro.iterrows():
            seq         = header['N']
            mul         = 1.0 / float(header['mul'])
            sample_rate = float(header['smp_rate'])

            batch = isbd_df[isbd_df['N'] == seq]
            for _, bd in batch.iterrows():
                x_all.extend(bd['x'])
                y_all.extend(bd['y'])
                z_all.extend(bd['z'])

            if sample_time is None:
                sample_time = float(header['SampleUS']) * 1e-6

        x_all = np.array(x_all) * mul
        y_all = np.array(y_all) * mul
        z_all = np.array(z_all) * mul

        return {
            'x': x_all,
            'y': y_all,
            'z': z_all,
            'sample_time': sample_time,
            'sample_rate': sample_rate,
        }

    def _run_batch_fft(self, gyro_data, window_size):
        rate = gyro_data['sample_rate']

        # hard code 50% overlap
        window_overlap = 0.5

        window_spacing     = round(window_size * (1 - window_overlap))
        windowing_function = self._hanning(window_size)

        # average sample time
        sample_time = 1.0 / rate        

        # get bins
        bins = self._rfft_freq(window_size, sample_time)

        # run fft on gyro data to get pre-filter amplitude spectrum per window
        fft_result = self._run_fft(gyro_data, ['x', 'y', 'z'], window_size, window_spacing, windowing_function)

        # time array — one timestamp per fft window
        time = np.array(fft_result['center']) * sample_time + gyro_data['sample_time']

        return bins, fft_result, time

    def _hanning(self, length):
        w = np.zeros(length)
        scale = (2 * np.pi) / (length - 1)
        for i in range(length):
            w[i] = 0.5 - 0.5 * np.cos(scale * i)
        return w

    def _window_correction_factors(self, w):
        return {
            'linear': 1.0 / np.mean(w),
            'energy': 1.0 / np.sqrt(np.mean(w * w))
        }

    def _real_length(self, length):
        return int(np.floor(length / 2) + 1)

    def _rfft_freq(self, length, d):
        real_len = self._real_length(length)
        freq     = np.zeros(real_len)
        for i in range(real_len):
            freq[i] = i / (length * d)
        return freq

    def _run_fft(self, data, keys, window_size, window_spacing, windowing_function):
        # compute mean pre-filter amplitude across all windows
        num_windows = int(np.floor((len(data[keys[0]]) - window_size) / window_spacing) + 1)
        real_len    = self._real_length(window_size)

        # allocate for each window
        ret = {'center': [0.0] * num_windows}
        for key in keys:
            ret[key] = [None] * num_windows

        # pre-allocate scale array.
        # double positive spectrum to account for discarded energy in negative spectrum
        # note that we don't scale the DC or Nyquist limit
        # normalize all points by the window size
        end_scale           = 1 / window_size
        mid_scale           = 2 / window_size
        scale               = [mid_scale] * real_len
        scale[0]            = end_scale
        scale[real_len - 1] = end_scale

        scale              = np.array(scale) # for fast multiplication 
        windowing_function = np.asarray(windowing_function)

        for i in range(num_windows):
            # calculate the start of each window
            window_start = i * window_spacing
            window_end = window_start + window_size
            ret['center'][i] = window_start + window_size * 0.5

            # take average time for window
            for key in keys:
                if key not in data:
                    continue

                # get data and apply windowing function
                windowed = data[key][window_start:window_end] * windowing_function

                # run fft and allocate for result
                fft_out = np.fft.rfft(windowed)

                # apply scale and convert to convert complex format
                ret[key][i] = [
                    (np.real(fft_out) * scale).tolist(),
                    (np.imag(fft_out) * scale).tolist(),
                ]

        return ret

    def _get_hnotch_param_names(self):
        prefixes = ['INS_HNTCH_', 'INS_HNTC2_']
        ret = []
        for prefix in prefixes:
            ret.append({
                'enable':      prefix + 'ENABLE',
                'mode':        prefix + 'MODE',
                'freq':        prefix + 'FREQ',
                'bandwidth':   prefix + 'BW',
                'attenuation': prefix + 'ATT',
                'ref':         prefix + 'REF',
                'min_ratio':   prefix + 'FM_RAT',
                'harmonics':   prefix + 'HMNCS',
                'options':     prefix + 'OPTS',
            })
        return ret

    def _load_filters(self):
        hnotch_params = self._get_hnotch_param_names()

         # load static
        filters_static = self._digital_biquad_filter(self.params_dict.get('INS_GYRO_FILTER'))

        defaults = {
            'enable'     : 0, 
            'mode'       : 0, 
            'freq'       : 80.0, 
            'bandwidth'  : 80.0 / 2,
            'attenuation': 40.0, 
            'ref'        : 1.0, 
            'min_ratio'  : 1.0,
            'harmonics'  : 3, 
            'options'    : 0,
        }

        # load harmonics notches
        filters_notch = []
        for hnotch in hnotch_params:
            params = {}
            for key, param_name in hnotch.items():
                value = self.params_dict.get(param_name)
                params[key] = value if value is not None else defaults[key]

            # ensure int types for bitmask fields
            params['harmonics'] = int(params['harmonics'])
            params['options']   = int(params['options'])

            if self.tune:
                params['freq']        = self.notch_freq     if self.notch_freq     is not None else params['freq']
                params['bandwidth']   = self.notch_bandwith if self.notch_bandwith is not None else params['bandwidth']
                params['attenuation'] = self.notch_att      if self.notch_att      is not None else params['attenuation']

            filters_notch.append(self._harmonic_notch_filter(params))

        return {'static': filters_static, 'notch': filters_notch}

    def _digital_biquad_filter(self, freq):
        target_freq = freq

        if target_freq <= 0:
            def transfer(Hn, Hd, sample_freq, Z1, Z2):
                return
            return transfer

        # build transfer function and apply to H division done at final step
        def transfer(Hn, Hd, sample_freq, Z1, Z2):
            fr = sample_freq / target_freq
            ohm = np.tan(np.pi / fr)
            c = 1.0 + 2.0 * np.cos(np.pi / 4.0) * ohm + ohm * ohm

            b0 = ohm * ohm / c
            b1 = 2.0 * b0
            b2 = b0
            a1 = 2.0 * (ohm * ohm - 1.0) / c
            a2 = (1.0 - 2.0 * np.cos(np.pi / 4.0) * ohm + ohm * ohm) / c

            length = len(Z1[0])
            for i in range(length):
                # H(z) = (b0 + b1*z^-1 + b2*z^-2)/(a0 + a1*z^-1 + a2*z^-2)
                numerator_r = b0 + b1 * Z1[0][i] + b2 * Z2[0][i]
                numerator_i =      b1 * Z1[1][i] + b2 * Z2[1][i]
                denominator_r = 1 + a1 * Z1[0][i] + a2 * Z2[0][i]
                denominator_i =     a1 * Z1[1][i] + a2 * Z2[1][i]

                # this is just two instances of complex multiplication
                # reimplementing it inline here saves memory and is faster
                num_ac = Hn[0][i] * numerator_r;   num_bd = Hn[1][i] * numerator_i
                num_ad = Hn[0][i] * numerator_i;   num_bc = Hn[1][i] * numerator_r
                Hn[0][i] = num_ac - num_bd;         Hn[1][i] = num_ad + num_bc

                den_ac = Hd[0][i] * denominator_r; den_bd = Hd[1][i] * denominator_i
                den_ad = Hd[0][i] * denominator_i; den_bc = Hd[1][i] * denominator_r
                Hd[0][i] = den_ac - den_bd;         Hd[1][i] = den_ad + den_bc

        return transfer

    def _notch_filter(self, attenuation_dB, bandwidth_hz, harmonic_mul, min_freq_fun, spread_mul):
        A = 10.0 ** (-attenuation_dB / 40.0)

        def transfer(Hn, Hd, center, sample_freq, Z1, Z2):
            center_freq_hz = center * harmonic_mul

            # check center frequency is in the allowable range
            if (center_freq_hz <= 0.5 * bandwidth_hz) or (center_freq_hz >= 0.5 * sample_freq):
                return

            min_freq = min_freq_fun(harmonic_mul)
            A_ = A
            if center_freq_hz < min_freq:
                disable_freq = min_freq * 0.25
                if center_freq_hz < disable_freq:
                    # disable
                    return
                
                # reduce attenuation (A of 1.0 is no attenuation)
                ratio = (center_freq_hz - disable_freq) / (min_freq - disable_freq)
                A_ = 1.0 + (A - 1.0) * ratio

            center_freq_hz = max(center_freq_hz, min_freq) * spread_mul

            octaves = np.log2(center_freq_hz / (center_freq_hz - bandwidth_hz / 2.0)) * 2.0
            Q = (2.0 ** octaves) ** 0.5 / ((2.0 ** octaves) - 1.0)
            Asq = A_ ** 2

            omega = 2.0 * np.pi * center_freq_hz / sample_freq
            alpha = np.sin(omega) / (2 * Q)

            b0 =  1.0 + alpha * Asq
            b1 = -2.0 * np.cos(omega)
            b2 =  1.0 - alpha * Asq
            a0 =  1.0 + alpha
            a1 =  b1
            a2 =  1.0 - alpha

            # build transfer function and apply to H division done at final step
            len_ = len(Z1[0])
            for i in range(len_):
                # H(z) = (b0 + b1*z^-1 + b2*z^-2)/(a0 + a1*z^-1 + a2*z^-2)
                numerator_r = b0 + b1 * Z1[0][i] + b2 * Z2[0][i]
                numerator_i =      b1 * Z1[1][i] + b2 * Z2[1][i]
                denominator_r = a0 + a1 * Z1[0][i] + a2 * Z2[0][i]
                denominator_i =      a1 * Z1[1][i] + a2 * Z2[1][i]

                # this is just two instances of complex multiplication
                # reimplementing it inline here saves memory and is faster
                num_ac = Hn[0][i] * numerator_r;   num_bd = Hn[1][i] * numerator_i
                num_ad = Hn[0][i] * numerator_i;   num_bc = Hn[1][i] * numerator_r
                Hn[0][i] = num_ac - num_bd;         Hn[1][i] = num_ad + num_bc

                den_ac = Hd[0][i] * denominator_r; den_bd = Hd[1][i] * denominator_i
                den_ad = Hd[0][i] * denominator_i; den_bc = Hd[1][i] * denominator_r
                Hd[0][i] = den_ac - den_bd;         Hd[1][i] = den_ad + den_bc

        return transfer

    def _multi_notch(self, attenuation_dB, bandwidth_hz, harmonic, min_freq_fun, num, center):
        # calculate spread required to achieve an equivalent single notch using two notches with bandwidth/2
        notch_spread = bandwidth_hz / (32.0 * center)
        bw_scaled = (bandwidth_hz * harmonic) / num

        notches = []
        notches.append(self._notch_filter(attenuation_dB, bw_scaled, harmonic, min_freq_fun, 1.0 - notch_spread))
        notches.append(self._notch_filter(attenuation_dB, bw_scaled, harmonic, min_freq_fun, 1.0 + notch_spread))
        if num == 3:
            notches.append(self._notch_filter(attenuation_dB, bw_scaled, harmonic, min_freq_fun, 1.0))

        def transfer(Hn, Hd, center, sample_freq, Z1, Z2):
            for notch in notches:
                notch(Hn, Hd, center, sample_freq, Z1, Z2)

        return transfer

    def _harmonic_notch_filter(self, params):
        enabled = params['enable'] > 0

        def static():
            return params['mode'] == 0

        def harmonics():
            return params['harmonics']

        if not enabled:
            # disable
            def transfer(Hn, Hd, instance, index, sample_freq, Z1, Z2):
                return
            return {'enabled': lambda: False, 'static': static, 'harmonics': harmonics, 'transfer': transfer}

        triple = (params['options'] & 16) != 0
        double = (params['options'] & 1)  != 0
        single = not double and not triple

        filter_V1 = self.filter_version == 1
        treat_low_freq_as_min = (params['options'] & 32) != 0

        def get_min_freq(harmonic):
            if filter_V1:
                return 0.0
            min_freq = params['freq'] * params['min_ratio']
            if treat_low_freq_as_min:
                return min_freq * harmonic
            return min_freq

        notches = []
        for n in range(self.MAX_NUM_HARMONICS):
            if params['harmonics'] & (1 << n):
                harmonic = n + 1
                if single:
                    notches.append(self._notch_filter(params['attenuation'], params['bandwidth'] * harmonic, harmonic, get_min_freq, 1.0))
                else:
                    notches.append(self._multi_notch(params['attenuation'], params['bandwidth'], harmonic, get_min_freq, 2 if double else 3, params['freq']))

        def transfer(Hn, Hd, instance, index, sample_freq, Z1, Z2):
            # get target frequencies from target
            if params['mode'] == 0:
                freq = [params['freq']]
            else:
                # placeholder - dynamic tracking not yet implemented
                freq = [params['freq']]

            if freq is not None:
                for i in range(len(notches)):
                    # cycle over each notch
                    for j in range(len(freq)):
                        # run each notch multiple times for multi frequency motor/esc/fft tracking
                        notches[i](Hn, Hd, freq[j], sample_freq, Z1, Z2)

        return {'enabled': lambda: True, 'static': static, 'harmonics': harmonics, 'transfer': transfer}

    def _calculate_transfer_function(self, bins, time, rate):
        filters = self._load_filters()

        # Z = e^jw, Z1 = Z^-1, Z2 = Z^-2
        omega = 2.0 * np.pi * bins / rate
        Z1 = [np.cos(-omega), np.sin(-omega)]
        Z2 = [np.cos(-2.0 * omega), np.sin(-2.0 * omega)]

        def calc(index, time, rate, Z1, Z2):
            Z_len = len(Z1[0])
            Hn_static = [np.ones(Z_len), np.zeros(Z_len)]
            Hd_static = [np.ones(Z_len), np.zeros(Z_len)]

            # low pass does not change frequency in flight
            filters['static'](Hn_static, Hd_static, rate, Z1, Z2)

            # evaluate any static notch
            for k in range(len(filters['notch'])):
                if filters['notch'][k]['enabled']() and filters['notch'][k]['static']():
                    filters['notch'][k]['transfer'](Hn_static, Hd_static, index, None, rate, Z1, Z2)

            len_ = len(time)
            ret_H = [None] * len(time)
            for j in range(len_):
                Hn = [Hn_static[0].copy(), Hn_static[1].copy()]
                Hd = [Hd_static[0].copy(), Hd_static[1].copy()]

                for k in range(len(filters['notch'])):
                    if filters['notch'][k]['enabled']() and not filters['notch'][k]['static']():
                        filters['notch'][k]['transfer'](Hn, Hd, index, j, rate, Z1, Z2)

                denom = Hd[0] ** 2 + Hd[1] ** 2
                H_r = (Hn[0] * Hd[0] + Hn[1] * Hd[1]) / denom
                H_i = (Hn[1] * Hd[0] - Hn[0] * Hd[1]) / denom
                ret_H[j] = [H_r, H_i]

            return ret_H

        return calc(0, time, rate, Z1, Z2)

    def _estimate_pre_post(self, fft_result, H, window_size, rate):
        # white noise noise model
        # https://en.wikipedia.org/wiki/Quantization_(signal_processing)#Quantization_noise_model
        # see also analog devices:
        # "Taking the Mystery out of the Infamous Formula, "SNR = 6.02N + 1.76dB," and Why You Should Care"
        # the 16 here is the number of bits in the batch log
        quantization_noise = 1.0 / (np.sqrt(3) * 2 ** (16 - 0.5))

        windowing_function = self._hanning(window_size)

        # get windowing correction factors for use later when plotting
        window_correction = self._window_correction_factors(windowing_function)

        # windowing amplitude correction depends on spectrum of interest and resolution
        wc = window_correction['linear']

        # scale quantization by the window correction factor so correction can be applied later
        quantization_correction = quantization_noise * (1.0 / wc)

        # compute mean pre-filter amplitude across all windows
        num_windows = len(fft_result['x'])
        real_len = self._real_length(window_size)

        fft_mean_pre_x  = np.zeros(real_len)
        fft_mean_pre_y  = np.zeros(real_len)
        fft_mean_pre_z  = np.zeros(real_len)
        fft_mean_post_x = np.zeros(real_len)
        fft_mean_post_y = np.zeros(real_len)
        fft_mean_post_z = np.zeros(real_len)

        # compute mean post-filter estimate across all windows
        for j in range(num_windows):
            amp_x = np.sqrt(np.array(fft_result['x'][j][0])**2 + np.array(fft_result['x'][j][1])**2)
            amp_y = np.sqrt(np.array(fft_result['y'][j][0])**2 + np.array(fft_result['y'][j][1])**2)
            amp_z = np.sqrt(np.array(fft_result['z'][j][0])**2 + np.array(fft_result['z'][j][1])**2)

            fft_mean_pre_x += amp_x
            fft_mean_pre_y += amp_y
            fft_mean_pre_z += amp_z

            attenuation = np.sqrt(H[j][0]**2 + H[j][1]**2)
            fft_mean_post_x += (amp_x - quantization_correction) * attenuation + quantization_correction
            fft_mean_post_y += (amp_y - quantization_correction) * attenuation + quantization_correction
            fft_mean_post_z += (amp_z - quantization_correction) * attenuation + quantization_correction

        # convert to dB (20 * log10(amplitude))
        pre_x  = 20 * np.log10(np.maximum(fft_mean_pre_x  * wc / num_windows, 1e-20))
        pre_y  = 20 * np.log10(np.maximum(fft_mean_pre_y  * wc / num_windows, 1e-20))
        pre_z  = 20 * np.log10(np.maximum(fft_mean_pre_z  * wc / num_windows, 1e-20))
        post_x = 20 * np.log10(np.maximum(fft_mean_post_x * wc / num_windows, 1e-20))
        post_y = 20 * np.log10(np.maximum(fft_mean_post_y * wc / num_windows, 1e-20))
        post_z = 20 * np.log10(np.maximum(fft_mean_post_z * wc / num_windows, 1e-20))

        return pre_x, pre_y, pre_z, post_x, post_y, post_z

    def _plot(self, bins, pre_x, pre_y, pre_z, post_x, post_y, post_z):        
        plt.figure(figsize=(28, 8))

        hnotch_params_list = self._get_hnotch_param_names()
        for notch_idx, hnotch in enumerate(hnotch_params_list):
            enable = self.params_dict.get(hnotch['enable'], 0)
            if enable <= 0:
                continue

            notch_freq_ = float(self.params_dict.get(hnotch['freq']))
            bw = float(self.params_dict.get(hnotch['bandwidth']))
            harmonics = int(self.params_dict.get(hnotch['harmonics']))

            if self.autotune and self.motor_freq is not None:
                notch_freq_ = self.motor_freq
            elif self.tune:
                notch_freq_ = self.notch_freq     if self.notch_freq     is not None else notch_freq_
                bw          = self.notch_bandwith if self.notch_bandwith is not None else bw

            for n in range(self.MAX_NUM_HARMONICS):
                if harmonics & (1 << n):
                    harmonic = n + 1
                    center_freq = notch_freq_ * harmonic

                    lower = center_freq - bw / 2.0
                    upper = center_freq + bw / 2.0

                    plt.axvspan(lower, upper, color='gray', alpha=0.3)
                    plt.axvline(x=center_freq, color='red', linestyle='--')

        plt.plot(bins, pre_x,  label='X Pre-filter')
        plt.plot(bins, pre_y,  label='Y Pre-filter')
        plt.plot(bins, pre_z,  label='Z Pre-filter')
        plt.plot(bins, post_x, label='X Post-filter')
        plt.plot(bins, post_y, label='Y Post-filter')
        plt.plot(bins, post_z, label='Z Post-filter')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        plt.grid(True)

        custom_handles = [
            Line2D([0], [0], color='red', linestyle='--', label='Harmonic Notch Frequency'),
            Patch(facecolor='gray', alpha=0.3, label='Harmonic Notch Bandwidth'),
        ]
        
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles=handles + custom_handles, labels=labels + [h.get_label() for h in custom_handles])

    def _detect_motor_frequency(self, bins, pre_x, pre_y, pre_z, min_freq=50, max_freq=400):
        # average across all three axes
        pre_avg = (pre_x + pre_y + pre_z) / 3

        # restrict search to frequency range of interest
        freq_mask   = (bins >= min_freq) & (bins <= max_freq)
        bins_masked = bins[freq_mask]
        pre_masked  = pre_avg[freq_mask]

        # find all peaks
        peak_indices, _ = signal.find_peaks(pre_masked)

        if len(peak_indices) == 0:
            print('No motor frequency peak detected.')
            return None

        # pick the peak with the highest amplitude
        best_peak_idx = peak_indices[np.argmax(pre_masked[peak_indices])]
        motor_freq    = bins_masked[best_peak_idx]

        print(f'Detected Motor Frequency: {motor_freq:.1f} Hz')

        return motor_freq