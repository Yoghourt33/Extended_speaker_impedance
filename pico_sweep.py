# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Yoghourt (Yoghourt33 on github)
"""
Script devoted to measure accurately a complex impedance through non-ideal (budget) bench. Goal is speaker HF impedance.
Hardware :
- Picoscope 2204A
- sine generator > 3MHz 8Vpp that does not "blip" on frequency change (picoscope siggen is limited to 100kHz 2V)
- test fixture : breadboard with BNC in, Rtest=3.3R precision resistor metal film, BNC out, with provisioning of scope
  probe grips (also for probes gnd)
  BNC - female banana adapter
- For calibration, you'll also need
   + a good and shorted possible strap (mine taken from used desoldering braid)
   + a mini-cable for direct connection to speaker (mine 2.5mm² 16cm multi-stranded domestic cable)

Proceeding :
- Connect to instruments
- Run a sweep, frequency per frequency with log spacing
    + extract accurately target frequency results from scope traces, including complex impedance computation
    + compute uncertainties on the fly
- Generate a plot (A and B voltages, deduced Z as module and phase) and save results
"""
import csv
import math
import matplotlib.pyplot as plt
import time
import datetime

from picosdk.ps2000 import ps2000 as ps

from config import *
import libpico
import libsiggen

def t_95(n):
    """Return the two-sided 95% t critical value for n samples."""

    table = {

        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,

        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,

        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,

        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,

        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,

        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042

    }

    return table.get(n, 1.96)

def run_sweep(vpeak_amplitude, Rtest):
    """Run the sweep and return a list of per-frequency result dictionaries."""

    if SIGGEN_VISA_ADDR:
        siggen = libsiggen.SigGenDriver(resource_name=SIGGEN_VISA_ADDR)
    print("Connected to:", siggen.idn())
    siggen.set_impedance('MAX')
    siggen.set_sine(frequency_hz=1e3, amplitude_vpp=2*vpeak_amplitude)
    siggen.enable_output(True)

    scope = libpico.PicoScope2204A()
    scope.open()
    print("Connected to:", scope)

    results = []
    try:
        for f in TARGET_FREQS:
            if SIGGEN_VISA_ADDR:
                siggen.set_freq(f)
            else:
                scope.set_siggen(f, pk_to_pk_uv=2*VPEAK_AMPLITUDE*1e6)
            time.sleep(min(max(1/f*200, 0.1),1))     # Frequency-adaptive settling time between 0.1s and 1s

            scope.set_simple_trigger('A',threshold_mv=0, falling_edge=False)

            scope.auto_select_range(f, coupling=COUPLING, samples_per_cycle=16)

            assert scope.fs_hz
            total_samples = scope.choose_capture_length(scope.fs_hz, f, cycles=CYCLES_TO_CAPTURE)
            scope.choose_timebase(total_samples, f, samples_per_cycle=SAMPLES_PER_CYCLE)

            fmeas_list = []
            A_pk_list = []
            a_dc_list = []
            b_dc_list = []
            B_pk_list = []
            Z_pk_list = []
            sigma_fft_list = []

            bw = max(BW_MIN_HZ, 0.15 * f)
            a_full_scale_v = ps.PICO_VOLTAGE_RANGE[ps.PS2000_VOLTAGE_RANGE[scope.ch['A']['rng']]]
            b_full_scale_v = ps.PICO_VOLTAGE_RANGE[ps.PS2000_VOLTAGE_RANGE[scope.ch['B']['rng']]]

            for _ in range(N_AVG):
                a_v, b_v = scope.acquire_block(total_samples)

                fmeas, A_pk, B_pk, Z_pk, sigma_fft, a_dc, b_dc = scope.fft_impedance(a_v, b_v, scope.fs_hz, f, bw, PROBE_X1_X10, Rtest)

                # f_peak, a_peak_vpk, sigma_fft, _, _= scope.fft_peak_subbin(a_v, scope.fs_hz, f, bw, PROBE_X1_X10)
                # _, b_peak_vpk, _, _, _             = scope.fft_peak_subbin(b_v, scope.fs_hz, f, bw, PROBE_X1_X10)
                # # b_pf ~ a_pf, a_freqs == b_freqs, a_sigma_fft == b_sigma_fft

                # a_vpk = np.abs(A_pk)
                # a_phase = np.angle(A_pk, deg=True)

                fmeas_list.append(fmeas)
                A_pk_list.append(A_pk)
                B_pk_list.append(B_pk)
                Z_pk_list.append(Z_pk)
                sigma_fft_list.append(sigma_fft)
                a_dc_list.append(a_dc)
                b_dc_list.append(b_dc)
                # a_spectra.append(a_mag)
                # b_spectra.append(b_mag)

            def mean_std_complex(x):
                x_arr = np.array(x)
                x_mean = np.mean(x_arr)
                x_mean_mod = np.abs(x_mean)
                x_mean_phase = np.angle(x_mean, deg=True)
                x_mod_std = np.std(np.abs(x_arr))
                return x_mean_mod, x_mean_phase, x_mod_std

            def mean_std_real(x):
                x_arr = np.array(x, dtype=float)
                return float(np.mean(x_arr)), float(np.std(x_arr))

            fmeas, fmeas_std = mean_std_real(fmeas_list)
            a_vpk, a_deg, a_vpk_std = mean_std_complex(A_pk_list)
            b_vpk, b_deg, b_vpk_std = mean_std_complex(B_pk_list)
            z_ohm, z_deg, z_ohm_std = mean_std_complex(Z_pk_list)
            z_deg_std = float(np.std(np.angle(np.array(Z_pk_list), deg=True)))
            max_a_dc = max(a_dc_list)
            max_b_dc = max(b_dc_list)
            sigma_f_total = scope.frequency_rms_uncertainty(total_samples, scope.fs_hz, fmeas_std)
            sigma_a_vpk_total = scope.amplitude_rms_uncertainty(a_vpk_std, a_full_scale_v, scope.MAX_ADC.value)
            sigma_b_vpk_total = scope.amplitude_rms_uncertainty(b_vpk_std, b_full_scale_v, scope.MAX_ADC.value)

            # def sigma_to_ci_95_half_total(sigma):
            #     confidence = t_95(N_AVG - 1) * sigma / math.sqrt(N_AVG) if N_AVG > 1 else np.nan
            #     return float(confidence)

            results.append({
                "ftarget_Hz": float(f),

                "Arange": scope.RANGE_NAME[scope.ch['A']['rng']],
                "Brange": scope.RANGE_NAME[scope.ch['B']['rng']],

                "timebase": int(scope.timebase),
                "dt_ns": float(scope.dt_ns),
                "fs_Hz": float(scope.fs_hz),
                "bw_Hz": float(bw),

                "samples": int(total_samples),

                "fmeas_Hz": fmeas,

                "a_Vpk": a_vpk,
                "a_deg": a_deg,
                "b_Vpk": b_vpk,
                "z_ohm": z_ohm,
                "z_deg": z_deg,
                "b_deg": b_deg,

                "sigma_f_Hz": sigma_f_total,
                "sigma_a_Vpk": sigma_a_vpk_total,
                "sigma_b_Vpk": sigma_b_vpk_total,
                "sigma_z_ohm": z_ohm_std,
                "sigma_z_deg": z_deg_std,
                "a_adc_lsb_v": float(a_full_scale_v / scope.MAX_ADC.value),
                "b_adc_lsb_v": float(b_full_scale_v / scope.MAX_ADC.value),
                "max_a_dc" : max_a_dc,
                "max_b_dc": max_b_dc,
            })

            print(
                f"f={f:6.0f} Hz | tb={scope.timebase} | δfmeas={(fmeas/f-1)*1e6:2.0f} ppm | "
                f"A rng={scope.RANGE_NAME[scope.ch['A']['rng']]:>6s} : {a_vpk*1e3:4.2f} mVpk+{max_a_dc*1e3:.0f}mV|dc|_max | "
                f"B rng={scope.RANGE_NAME[scope.ch['B']['rng']]:>6s} : {b_vpk*1e3:4.2f} mVpk+{max_b_dc*1e3:.0f}mV|dc|_max | "
                f"|Z| {z_ohm:4.3f} Ω | "
            )

    finally:
        scope.close()
        siggen.close()

    return results


def save_results(results, filename="picoscope_2204A_sweep.csv"):
    """Write results to CSV."""

    if not results:
        return

    with open(filename, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(results[0].keys()))

        writer.writeheader()

        writer.writerows(results)


def plot_results(results, filename="picoscope_2204A_sweep.png"):
    """Plot measured peak frequency, amplitude, and dBV versus target frequency."""

    f_target = np.array([r["ftarget_Hz"] for r in results])
    # f_meas = np.array([r["fmeas_Hz"] for r in results])
    a_meas = np.array([r["a_Vpk"] for r in results])
    b_meas = np.array([r["b_Vpk"] for r in results])
    # a_dbv = np.array([20*np.log10(r["a_Vpk"]) for r in results])
    # b_dbv = np.array([20*np.log10(r["b_Vpk"]) for r in results])
    z_ohm = np.array([r["z_ohm"] for r in results])
    z_dbohm = 20 * np.log10(z_ohm)
    z_deg = np.array([r["z_deg"] for r in results])
    z_sigma = np.array([r["sigma_z_ohm"] for r in results])
    z_sigma_deg = np.array([r["sigma_z_deg"] for r in results])
    z_sigma_rel = z_sigma / z_ohm * 100  # as %
    # f_ci = np.array([r["f_ci"] for r in results])
    # a_ci = np.array([r["a_vpk_ci95_half_total"] for r in results])
    # b_ci = np.array([r["b_vpk_ci95_half_total"] for r in results])

    fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    i=0
    ax[i].set_xscale('log')

    # --- Measured frequency ---
    # ax[i].errorbar(f_target, f_meas, yerr=f_ci, fmt=".-", capsize=3, label="F measured ±95% CI")
    # ax[i].set_yscale('log')
    # ax[i].set_ylabel("Peak frequency (Hz)")
    # ax[i].grid(True, which="both", ls="--", alpha=0.5)
    # ax[i].legend()
    # i +=1

    # --- Peak amplitude voltage ---
    ax[i].semilogx(f_target, a_meas, label="A")
    ax[i].semilogx(f_target, b_meas, label="B")
    ax[i].set_ylabel("Peak amplitude (Vpk)")
    ax[i].grid(True, which="both", ls="--", alpha=0.5)
    ax[i].legend()
    i +=1

    # --- Amplitude  in dBV ---
    # ax[i].semilogx(f_target, a_dbv, label="A")
    # ax[i].semilogx(f_target, b_dbv, label="B")
    # ax[i].set_ylabel("Peak amplitude (dBV)")
    # ax[i].grid(True, which="both", ls="--", alpha=0.5)
    # ax[i].legend()
    # i +=1

    # --- Impedance |Z| + phase ---
    color_z   = "tab:blue"
    color_phi = "tab:orange"
    ax[i].semilogx(f_target, z_dbohm, color=color_z, label="|Z| downward test fixture")
    ax[i].set_ylabel("|Z| (dBΩ)", color=color_z)
    ax[i].tick_params(axis='y', labelcolor=color_z)
    ax[i].grid(True, which="both", ls="--", alpha=0.5)
    axb = ax[i].twinx()
    axb.semilogx(f_target, z_deg, color=color_phi, linestyle="--", label="Phase (°)")
    axb.set_ylabel("Phase (°)", color=color_phi)
    axb.tick_params(axis='y', labelcolor=color_phi)
    axb.set_ylim(-90, 90)
    axb.axhline(0, color=color_phi, linewidth=0.5, linestyle=":")
    # Combined legend
    lines_z, labels_z = ax[i].get_legend_handles_labels()
    lines_phi, labels_phi = axb.get_legend_handles_labels()
    ax[i].legend(lines_z + lines_phi, labels_z + labels_phi)
    i +=1

    # --- Sigma |Z| (relative to |Z| ---
    ax[i].semilogx(f_target, z_sigma_rel, color='tab:blue', marker='.', markersize=2)
    ax[i].axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='1%')
    ax[i].axhline(5.0, color='gray', linestyle=':', alpha=0.5, label='5%')
    ax[i].set_ylabel('σ|Z| / |Z| (%)')
    ax[i].set_title(' |Z| relative uncertainty')
    ax[i].grid(True, which='both', ls='--', alpha=0.5)
    ax[i].legend()
    i += 1

    # Sigma Z phase
    ax[i].semilogx(f_target, z_sigma_deg, color='tab:orange', marker='.', markersize=2)
    ax[i].axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='1°')
    ax[i].axhline(5.0, color='gray', linestyle=':', alpha=0.5, label='5°')
    ax[i].set_ylabel('σφ (°)')
    ax[i].set_xlabel('Target frequency (Hz)')
    ax[i].set_title('Z phase uncertainty')
    ax[i].grid(True, which='both', ls='--', alpha=0.5)
    ax[i].legend()
    # i += 1

    ax[i].set_xlabel("Target frequency (Hz)")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()  # must follow savefig() else a blank image is saved!


if __name__ == "__main__":
    tic = time.time()
    print("Starting sweep at ", datetime.datetime.now())
    data = run_sweep(vpeak_amplitude=VPEAK_AMPLITUDE, Rtest=3.3051)
    toc = time.time()
    save_results(data, PATH_RAW/"pico_sweep.csv")

    plot_results(data, PATH_RAW/"pico_sweep.png")
    print(f"results saved to {PATH_RAW}/pico_sweep.*")
    print(f"Finished! {(toc - tic)/60:.2f} min")