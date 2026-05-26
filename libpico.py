# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Yoghourt (Yoghourt33 on github)
"""
Library above picosdk for Picoscope 2204A

Purpose:
- Connect to the instrument
- Manage picoscope signal generator (sine wave)
- Auto-range Y scale
- Choose a fit timebase w.r.t. specific sine frequency capture
- Acquire A and B scope data
- In Fourier space, extract a specific freq ray contribution (amplitude **and phase** ) with sub-bin precision
- compute related uncertainties
"""

import ctypes
import time
import math
import numpy as np
from numpy.fft import rfft, rfftfreq

from picosdk.ps2000 import ps2000 as ps
from picosdk.functions import assert_pico2000_ok, adc2mV, mV2adc


class PicoScope2204A:
    """High-level helper for PicoScope 2204A acquisition and FFT-based peak estimation."""
    # maximum ADC count value
    MAX_ADC = ctypes.c_int16(32767)
    VOLTAGE_RANGE_INDEX2LABEL = {v: k for k, v in ps.PS2000_VOLTAGE_RANGE.items()}
    RANGE_NAME = {}
    def __init__(self):

        """Initialize default state, channel identifiers, and result placeholders."""
        self.RANGE_NAME = {
            self.VOLTAGE_RANGE_INDEX2LABEL[idx]: f"{int(v * 1000) if v < 1 else int(v)} {'mV' if v < 1 else 'V'}"
            for idx, v in ps.PICO_VOLTAGE_RANGE.items()
        }

        self.handle = ctypes.c_int16(-1)

        self.timebase = None
        self.time_unit = None
        self.dt_ns = None

        self.fs_hz = None
        self.ch = {
            'A' : {
                'psid' : "PS2000_CHANNEL_A",
                'active' : 1,
                'rng' : "PS2000_2V",
                'coupling' : 'AC',
            },
            'B': {
                'psid': "PS2000_CHANNEL_B",
                'active': 1,
                'rng': "PS2000_2V",
                'coupling': 'AC',
            }
        }

    def open(self):
        """Open the first available PicoScope 2204A device and apply any power-source fixups."""

        status = ps.ps2000_open_unit()
        if status in (282, 286):
            assert False
            st = ps.ps2000aChangePowerSource(self.handle, st)
        assert_pico2000_ok(status)
        self.handle = ctypes.c_int16(status)

    def close(self):
        """Stop capture and close the device safely."""

        try:
            ps.ps2000_stop(self.handle)
        finally:
            ps.ps2000_close_unit(self.handle)

    def choose_capture_length(self, fs_hz, target_freq_hz, cycles=32, min_samples=1024, max_samples=2048):
        """Choose an FFT-friendly capture length that spans a target number of cycles."""

        n = int(np.ceil(cycles * fs_hz / target_freq_hz))
        n = max(min_samples, n)
        n = min(max_samples, n)
        return 1 << (n - 1).bit_length()

    def set_channel(self, channel='A', rng='PS2000_20V', active = 1, coupling='AC'):
        """Enable analog channels A and B with AC coupling and the selected input range."""
        self.ch[channel]['rng'] = rng
        self.ch[channel]['active'] = active
        self.ch[channel]['coupling'] = coupling

        status = ps.ps2000_set_channel(self.handle,
                                       ps.PS2000_CHANNEL[self.ch[channel]['psid']],
                                       self.ch[channel]['active'],
                                       ps.PICO_COUPLING[self.ch[channel]['coupling']],
                                       ps.PS2000_VOLTAGE_RANGE[self.ch[channel]['rng']])
        assert_pico2000_ok(status)

    def set_simple_trigger(self, channel, threshold_mv=20, falling_edge : bool=False, delay_pct : int = 0,
                           auto_trigger_after_ms : int = 1000):
        """Configure a simple rising-edge trigger on channel A using an approximate mV threshold."""
        ch_id = self.ch[channel]['psid']
        rng = self.ch[channel]['rng']
        threshold_adc = mV2adc(threshold_mv, ps.PS2000_VOLTAGE_RANGE[rng], self.MAX_ADC)
        direction = int(falling_edge)  # direction : 0=rising edge
        # delay_pct = 0  # location of trigger in window, from -100% to +100%
        # auto_trigger_after_ms = 1000
        status = ps.ps2000_set_trigger(self.handle, ps.PS2000_CHANNEL[ch_id], threshold_adc, direction,
                                       delay_pct, auto_trigger_after_ms)
        assert_pico2000_ok(status)

    def set_siggen(self, freq_hz, pk_to_pk_uv=2000000):
        """Configure the built-in sine generator for a fixed output frequency."""

        pk_to_pk_uv = int(pk_to_pk_uv)
        assert_pico2000_ok(ps.ps2000_set_sig_gen_built_in(
            self.handle,
            0,  # offset voltage
            pk_to_pk_uv,
            ps.PS2000_WAVE_TYPE['PS2000_SINE'],
            float(freq_hz),     # start frequency
            float(freq_hz),     # stop frequency
            0.0,                # increment
            1.0,                # dwell time
            ps.PS2000_SWEEP_TYPE['PS2000_UP'],       # sweep type
            0                   # sweeps
        ))

    def choose_timebase(self, n_samples, target_freq_hz, samples_per_cycle):
        """Select the fastest valid timebase that still meets the requested samples-per-cycle target."""

        desired_dt_ns = (1.0 / (target_freq_hz * samples_per_cycle)) * 1e9

        best = None
        for tb in range(1, 32):
            interval_ns = ctypes.c_int32()
            time_units = ctypes.c_int32()
            oversample = ctypes.c_int16(1)
            max_samples = ctypes.c_int32()

            st = ps.ps2000_get_timebase(self.handle, tb, n_samples, ctypes.byref(interval_ns), ctypes.byref(time_units),
                                        oversample, ctypes.byref(max_samples))
            if st == 0:
                # call failed, stop here
                break
            # Call succeeded. Check if still meeting the requested min sample rate = max time interval between samples
            if interval_ns.value <= desired_dt_ns:
                best = (tb, interval_ns.value, time_units.value, max_samples.value)
            else:
                # got too far, keep previous setting
                break

        if best is None:
            raise RuntimeError("No valid timebase found")

        self.timebase, self.dt_ns, self.time_unit, _ = best
        self.fs_hz = 1e9 / self.dt_ns
        return best

    def acquire_block(self, total_samples, segment_index=0):
        """Acquire one block and return channel A and B as voltage arrays in volts."""

        ti = ctypes.c_int32()
        assert_pico2000_ok(ps.ps2000_run_block(self.handle, total_samples, self.timebase, 1, ctypes.byref(ti)))

        ready = 0
        while ready == 0:
            time.sleep(0.001)
            ready = ps.ps2000_ready(self.handle)

        buf_a = (ctypes.c_int16 * total_samples)()
        buf_b = (ctypes.c_int16 * total_samples)()
        overflow = ctypes.c_int16()
        status = ps.ps2000_get_values(self.handle, ctypes.byref(buf_a), ctypes.byref(buf_b), None, None,
                                      ctypes.byref(overflow), total_samples)
        assert_pico2000_ok(status)

        a_mv = adc2mV(np.array(buf_a, dtype=np.int16), ps.PS2000_VOLTAGE_RANGE[self.ch['A']['rng']], self.MAX_ADC)
        b_mv = adc2mV(np.array(buf_b, dtype=np.int16), ps.PS2000_VOLTAGE_RANGE[self.ch['B']['rng']], self.MAX_ADC)
        return np.asarray(a_mv, dtype=np.float64) / 1000.0, np.asarray(b_mv, dtype=np.float64) / 1000.0

    def auto_select_range(self, freq_hz, coupling, samples_per_cycle, desired_vpk_estimate=None):
        """Choose the smallest safe input range using a preview capture and peak-fraction rules."""

        if desired_vpk_estimate is None:
            a_idx = ps.PS2000_VOLTAGE_RANGE[self.ch['A']['rng']]
            b_idx = ps.PS2000_VOLTAGE_RANGE[self.ch['B']['rng']]
        else:
            # find voltage range that best suits desired voltage
            assert desired_vpk_estimate <= max(ps.PICO_VOLTAGE_RANGE.values())

            for idx in reversed(list(ps.PS2000_VOLTAGE_RANGE.values())):
                if idx == min(ps.PS2000_VOLTAGE_RANGE.values())+1:
                    # Can't do better than min range, of course. Min for 2204A is 50mV not 20mV
                    break
                if ps.PICO_VOLTAGE_RANGE[idx-1] < desired_vpk_estimate <= ps.PICO_VOLTAGE_RANGE[idx]:
                    # best fit is current voltage range. The range below would clip
                    break
            a_idx = idx
            b_idx = idx

        # We have a starting point. Test and refine per channel through acquisition
        # Max number of loops is nb of ranges
        for _ in range(len(ps.PS2000_VOLTAGE_RANGE)):
            self.set_channel('A',rng=self.VOLTAGE_RANGE_INDEX2LABEL[a_idx], coupling=coupling)
            self.set_channel('B',rng=self.VOLTAGE_RANGE_INDEX2LABEL[b_idx], coupling=coupling)

            self.choose_timebase(3968, freq_hz, samples_per_cycle=samples_per_cycle)
            a_v, b_v = self.acquire_block(3968)
            a_vpk = float(np.max(np.abs(a_v)))
            b_vpk = float(np.max(np.abs(b_v)))

            cont = False    # If a range has to change, continue, else stop the loop at the end of the range check

            # Increase voltage range if needed
            if a_vpk > ps.PICO_VOLTAGE_RANGE[a_idx] and a_idx < max(ps.PS2000_VOLTAGE_RANGE.values()):
                a_idx = a_idx+1
                cont = True
            if b_vpk > ps.PICO_VOLTAGE_RANGE[b_idx] and b_idx < max(ps.PS2000_VOLTAGE_RANGE.values()):
                b_idx = b_idx+1
                cont = True
            # Decrease voltage range if needed
            if a_idx > min(ps.PS2000_VOLTAGE_RANGE.values())+1 and a_vpk < ps.PICO_VOLTAGE_RANGE[a_idx - 1]:
                a_idx = a_idx - 1
                cont = True
            if b_idx > min(ps.PS2000_VOLTAGE_RANGE.values())+1 and b_vpk < ps.PICO_VOLTAGE_RANGE[b_idx - 1]:
                b_idx = b_idx - 1
                cont = True

            if not cont:
                # No changed occurred, neither on A voltage range nor B voltage range. Exit.
                break
        # Job done, auto-range finished

    def frequency_rms_uncertainty(self, n_samples, fs_hz, std_repeat_hz):
        """Estimate one-sigma frequency uncertainty from repeatability and FFT bin width."""

        df = fs_hz / n_samples
        sigma_fft = df / math.sqrt(12.0)
        return math.sqrt(std_repeat_hz ** 2 + sigma_fft ** 2)

    def amplitude_rms_uncertainty(self, std_repeat_vpk, full_scale_v, max_adc_value):
        """Estimate one-sigma amplitude uncertainty in Vpk from repeatability and quantization."""

        lsb_v = full_scale_v / max_adc_value
        sigma_q_vrms = lsb_v / math.sqrt(12.0)
        sigma_q_vpk = sigma_q_vrms * math.sqrt(2.0)
        return math.sqrt(std_repeat_vpk ** 2 + sigma_q_vpk ** 2)

    # def quinn_1994(self, F,k):
    #     """ Find sub-bin peak frequency
    #      k is the best bin for searched peak.
    #      B. G. Quinn, "Estimating Frequency by Interpolation Using Fourier Coefficients," IEEE
    #     Trans. Signal Processing, Vol. 42, pp. 1264-1268, May 1994."""
    #
    #     a1 = np.real(F[k-1]/F[k])
    #     a2 = np.real(F[k+1]/F[k])
    #
    #     # Quinn2 symetrical estimator
    #     d1 = +a1 / (1-a1)
    #     d2 = -a2 / (1-a2)
    #
    #     # Selection rule
    #     if d1>0 and d2>0:
    #         dk = d2
    #     else:
    #         dk = d1
    #     return dk

    def grandke_freq(self, F, k):
        """Estimate sub-bin peak frequency using Grandke (1983) eq.(10) for Hann window.

        Reference: Thomas Grandke, "Interpolation Algorithms for Discrete Fourier
        Transforms of Weighted Signals," IEEE Trans. Instrumentation and Measurement,
        Vol. IM-32, No. 2, pp. 350-355, June 1983.

        Equation (10): xm = (2*alpha - 1) / (alpha + 1)
        where alpha = |G_H((lm+1)*df)| / |G_H(lm*df)|

        Notes:
        - Derived specifically for Hann window — do NOT use Quinn (1994) with Hann,
          as Quinn assumes rectangular window and gives wrong results (ratio of
          adjacent complex bins is ~-0.5 under Hann, not ~+0.5 as Quinn expects).
        - k_low is determined by comparing the two neighbors of the peak bin,
          NOT by comparing k+1 vs k directly.
        - xm is in [0, 1): fractional offset from k_low toward k_low+1.

        Args:
            F : complex rfft spectrum
            k : bin index of the peak (from argmax on FA only, never FB)
        Returns:
            k_low : lower of the two bins bracketing the true peak
            xm    : fractional offset in [0, 1]
        """
        if np.abs(F[k + 1]) > np.abs(F[k-1]):
            k_low = k
        else:
            k_low = k - 1

        alpha = np.abs(F[k_low + 1]) / np.abs(F[k_low])
        xm = (2 * alpha - 1) / (alpha + 1)
        xm = np.clip(xm, 0.0, 1.0)

        return k_low, xm

    def grandke_1983_amplitude(self, F, k_low, xm):
        """Estimate complex amplitude at sub-bin position using Grandke (1983) eq.(11).

        Reference: Thomas Grandke, "Interpolation Algorithms for Discrete Fourier
        Transforms of Weighted Signals," IEEE Trans. Instrumentation and Measurement,
        Vol. IM-32, No. 2, pp. 350-355, June 1983.

        Equations (11a) and (11b) — use the bin with larger magnitude for accuracy:
            Am = factor * exp(-pi*j*xm) * (1 + xm) * G_H(lm*df)        [eq. 11a]
            Am = factor * exp(-pi*j*xm) * (xm - 2) * G_H((lm+1)*df)    [eq. 11b]
        where factor = 2*pi*xm*(1-xm) / sin(pi*xm)

        IMPORTANT — common implementation mistake:
            The terms (1+xm) and (xm-2) are REAL multiplicative factors applied
            AFTER the complex exponential, NOT exponents inside exp().
            i.e.: factor * exp(-pi*j*xm) * (xm-2) * F[k]    [CORRECT]
            NOT:  factor * exp(-pi*j*xm*(xm-2)) * F[k]       [WRONG]

        Boundary cases (verified empirically):
            xm -> 0 : Am -> 2 * F[k_low]      (peak exactly on k_low)
            xm -> 1 : Am -> 2 * F[k_low+1]    (peak exactly on k_low+1)
            The factor 2 is required for consistency with the normalization
            1/(n * mean(window)) applied after this function.

        Args:
            F     : complex rfft spectrum (FA or FB)
            k_low : lower bin, as returned by grandke_freq() — must be the SAME
                    value for both channels FA and FB to preserve phase coherence
            xm    : fractional offset in [0, 1), as returned by grandke_freq()
        Returns:
            Am : complex amplitude (un-normalized, apply 1/(n*cg) after)
        """
        if xm < 1e-6:
            # Avoid limit case when xm ~ 0
            return 2*F[k_low]
        if xm > 1 - 1e-6:
            # Avoid limit case when xm ~ 1
            return 2*F[k_low + 1]

        factor = 2 * np.pi * xm * (1-xm) / np.sin(np.pi * xm)
        if np.abs(F[k_low+1]) >= np.abs(F[k_low]):
            Am = factor * np.exp(-np.pi * 1j * xm) * (xm - 2) * F[k_low + 1]
        else:
            Am = factor * np.exp(-np.pi * 1j * xm) * (1 + xm) * F[k_low]
        return Am

    def fft_impedance(self, a_v, b_v, fs_hz, target_freq_hz, bw_hz, probe_x1_x10,
                      Rtest):
        """ Compute complex impedance Z from two-channel voltage measurement with generator not necessarily sync'ed
            to scope.

        Measurement principle:
            Generator (sine at target frequency) → Rtest → Z_DUT
            Scope channel A measures Va (upstream of Rtest)
            Scope channel B measures Vb (downstream of Rtest = across Z_DUT)
            Both channels MUST have same ground
            *** Both channels MUST be acquired simultaneously ****

            Current : I = (Va - Vb) / Rtest
            Impedance: Z = Vb / I = Vb / (Va - Vb) * Rtest

        Key implementation choices:
        - Extracting peak frequency from freq domain instead of time domain brings several benefits :
            + Rejects DC offset, harmonic distortion, and uncorrelated noise (both random and deterministic) by
              focusing on a single bin.
            + Tolerates frequency drift (shift & jitter) between generator and scope on a capture-by-capture basis,
              since the peak bin is re-estimated at each acquisition. Averaging at the caller level re-centers any
              frequency shift, while amplitude accuracy is improved compared to evaluating at a fixed target
              frequency — which would systematically underestimate amplitude when the generator drifts off-bin.
        - Grandke (1983) interpolation for both frequency and amplitude,
          derived for Hann window. Quinn (1994) was tested and rejected:
          incompatible with Hann window (assumes rectangular window).
        - k_low and xm computed on channel A ONLY, then applied identically
          to channel B. This preserves phase coherence between channels,
          which is essential for accurate Z computation. Computing k_low/xm
          independently per channel introduces a phase artifact that creates
          a spurious ~40 dB/decade slope in the residual impedance.
        - Subtraction Va_peak - Vb_peak is performed on complex vectors BEFORE
          taking the modulus. Module-only subtraction |Va| - |Vb| introduces
          a systematic phase error that also creates a ~40 to ~60dB/decade artifact.
        - Normalization: 1 / (n * mean(window)) consistent with Grandke eq.(11)
          boundary cases (factor 2 for single-sided rfft already absorbed by Grandke amplitude computation).
        """

        n = len(a_v)
        w = np.hanning(n)
        df = fs_hz / n
        cg = np.sum(w) / n

        # Signal conditioning
        a_v = np.asarray(a_v, dtype=np.float64)* probe_x1_x10
        b_v = np.asarray(b_v, dtype=np.float64)* probe_x1_x10
        a_v_dc = np.mean(a_v)
        b_v_dc = np.mean(b_v)
        if abs(b_v_dc) > 0.1:
            raise ValueError(f"Too high DC component on B ({b_v_dc})")
        a_v -= a_v_dc
        b_v -= b_v_dc

        # Get to Fourier world
        FA = rfft(a_v * w)
        FB = rfft(b_v * w)
        freqs = rfftfreq(n, d=1/fs_hz)


        # Find target bin
        A_mag_k = np.abs(FA)
        band2 = (freqs >= (target_freq_hz - bw_hz/2)) & (freqs <= (target_freq_hz + bw_hz/2))
        if not np.any(band2):
            raise ValueError(f"No FFT bin found in band [{target_freq_hz-bw_hz/2:.1f}, {target_freq_hz+bw_hz/2:.1f}] Hz")
        band2_idx = np.where(band2)[0]
        k_local = np.argmax(A_mag_k[band2_idx])
        k = band2_idx[k_local]

        # Find sub-bin position
        # dk = self.quinn_1994(FA, k)
        k_low, xm = self.grandke_freq(FA, k)  # computed on FA only
        f_pk = freqs[k_low] + xm * df

        # Amplitude interpolation avoiding phase error bias, and normalization w.r.t rfft and windowing
        Va_c = self.grandke_1983_amplitude(FA, k_low, xm)
        Vb_c = self.grandke_1983_amplitude(FB, k_low, xm)
        A_pk = (1 / (n * cg)) * Va_c
        B_pk = (1 / (n * cg)) * Vb_c

        # Complex impedance with minimized phase/amplitude errors
        if A_pk == 0j or B_pk == 0j:
            pass    # for debug purpose
        Z = Rtest * B_pk/(A_pk - B_pk)

        sigma_fft = df / math.sqrt(12.0)

        return f_pk, A_pk, B_pk, Z, sigma_fft, a_v_dc, b_v_dc



if __name__ == "__main__":
    TBD("Lib autotest would be nice to have...")
    print("Finished!")