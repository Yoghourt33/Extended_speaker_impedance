# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Yoghourt (Yoghourt33 on github)
"""
post_process.py — Impedance measurement post-processing and calibration
=======================================================================

Calibration sequence:
    1.1  A=B probe matching check  → gain mismatch correction between channels A and B, for channels de-embedding
    1.2  Open, no adapter          → noise floor reference, test fixture HF parasitic capa
    1.2b 50R BNC termination       → accuracy check after de-embedding on 50R load
    2.1  Open, with adapter        → open reference (for short-open-load extract w/o mini-cable)
    2.1c Rload_ref, with adapter   → accuracy check after de-embedding on 50R load
    2.2  Shorted, with adapter     → short reference (for short-open-load extract w/o mini-cable)
    3.1  Mini-cable open           → open reference (for short-open-load extract WITH mini-cable)
    3.2  Mini-cable shorted        → short reference (for short-open-load extract WITH mini-cable)
    4.x  DUT measurements          → Speaker or whatever

Observed adapter resonances in 2.2 (test fixture + adapter + strap), also appearing slightly in 3.2:
    F1 ≈ 24kHz     — origin unclear: NOT a simple LC resonance with known passives : frequency gets HIGHER
                     on 3.2 (mini-cable shorted = additional cable inductance/capacitance)
                     Physical LC analysis gives implausible L or C values.
                     Suspected cause: superimposed complex resonances of overall test fixture, or generator noise pick-up.
    F2 ≈ 950 kHz   — Probably cumulated inductance  (~40nH) resonating with overall parasitic capacitance.
                     Stable between 2.2 and 3.2

De-embedding method:
    Gain/phase mismatch correction from 1.1 (A=B, same point)
    This reduces systematic error from ~6-7% to ~1% on resistive loads.
    Residual ~1% error consistent with picoscope ADC absolute accuracy (3% guaranteed, ~1% typical).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from config import *

# =============================================================================
# CSV loader
# =============================================================================
def load_csv(fname : Path) -> pd.DataFrame:
    """
    Load a sweep CSV — first row contains Rtest metadata, second row has column names.
    :param fname: file path
    :return: dataframe
    """
    df = pd.read_csv(fname)
    cols = [
        'ftarget_Hz',
        'a_Vpk','a_deg','b_Vpk','b_deg','z_ohm','z_deg',
        # 'sigma_f_Hz','sigma_a_Vpk','sigma_b_Vpk','sigma_z_ohm','sigma_z_deg',
        # 'a_adc_lsb_v','b_adc_lsb_v','max_a_dc','max_b_dc'
    ]
    assert all(col in df.columns for col in cols)
    assert df['ftarget_Hz'].dtype == 'float64' and not df['ftarget_Hz'].isnull().values.any()
    return df.reset_index(drop=True)


# =============================================================================
# Probes mismatch de-embedding
# =============================================================================
def deembed(df : pd.DataFrame, df_ref : pd.DataFrame, Rtest : float):
    """
    Apply gain mismatch correction and recompute Z.
        (all variables below are complex-valued and frequency dependent)
            extract Va_11 and Vb_11 from df11 dataframe
            cplx_ratio = Vb_11 / Va_11
            Va_corr = Va * cplx_ratio(f)
            Z = Rtest * Vb / (Va_corr - Vb)
    Returns dataframe df with added columns: a_corr_Vpk, a_corr_deg, Z_corr (complex-value)

    :param df:      dataframe to de-embed from probes
    :param df_ref:  dataframe containing probes mismatch data
    :param Rtest:   Test resistance value, to recompute Z
    :return:        updated dataframe
    """
    Va = df['a_Vpk'] * np.exp(1j * np.radians(df['a_deg']))
    Vb = df['b_Vpk'] * np.exp(1j * np.radians(df['b_deg']))
    df['Z'] = Rtest * Vb / (Va - Vb)

    ratio_c = df_ref['b_Vpk']/df_ref['a_Vpk']*np.exp(1j*(np.radians(df_ref['b_deg']-df_ref['a_deg'])))
    if  np.array_equal(df['ftarget_Hz'].values,df_ref['ftarget_Hz'].values):
        pass
    else:
        # frequencies do not match, use linear interpolation on real & imaginary parts
        # NOTE : interpolating separately module and phase is NOT adequate for a spectrum approach
        #   - phase is wrapped -> phase jump would lead to very poor interpolation
        #   - phase and module are not separate variables, especially unstable if ratio is small
        ratio_re = np.interp(df['ftarget_Hz'].values, df_ref['ftarget_Hz'].values, np.real(ratio_c))
        ratio_im = np.interp(df['ftarget_Hz'].values, df_ref['ftarget_Hz'].values, np.imag(ratio_c))
        ratio_c = ratio_re + 1j * ratio_im

    Va_corr = Va*ratio_c
    df['a_corr_Vpk'] = np.abs(Va_corr)
    df['a_corr_deg'] = np.angle(Va_corr, deg=True)
    df['Z_corr'] = Rtest * Vb / (Va_corr - Vb)

    # WARNING : hiZ (like open circuit impedance or parallel resonance with high Q can
    # lead to probe deembedding artefact amplitude(Va_corr) < amplitude(Vb)
    # This is due to measured Va-Vb hitting noise floor of correction ratio Vb/Va.
    # The main consequence is Re(Z_corr) becoming negative, which has no physical sense in our context.
    # WOR-AROUND : force Re(Z) positive while preserving Im(Z).
    # NOTE: this is a heuristic.
    # - For open circuit measures, heuristic choice is guided by capacitive behavior expected
    #   to appear by MHz frequencies, making open circuit measure gradually rise above noise floor
    # - For speaker measure, heuristic choice is guided by uncorrected Z
    re_z = np.real(df['Z_corr'])
    idx = np.real(df['Z_corr']) < 0
    if any(idx):
        im_z = np.imag(df['Z_corr'])
        df.loc[idx, 'Z_corr'] = -re_z[idx] + 1j*im_z[idx]

# =============================================================================
# Plot
# =============================================================================
def plot_z(results_dict: dict[str,tuple], title: str="Impedance sweep", filename: str|None=None,
           bounds: dict[tuple[int,int],tuple] | None=None,
           add_nyquist:bool=False):
    """
    Plot |Z| dBΩ + phase for multiple datasets.
    results_dict: {label: dataframe} with columns z_ohm, z_deg (or z_corr_ohm, z_corr_deg).

    """
    if add_nyquist:
        fig, ax = plt.subplots(3, 2, figsize=(12, 12))
    else:
        fig, ax = plt.subplots(2 , 2, figsize=(11, 8))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    phi_max = 0
    for i, (label, (df, col)) in enumerate(results_dict.items()):
        f   = df['ftarget_Hz'].values
        if col is None:
            col_z   = 'z_corr_ohm' if 'z_corr_ohm' in df.columns else 'z_ohm'
            col_deg = 'z_corr_deg' if 'z_corr_deg' in df.columns else 'z_deg'
            z   = df[col_z].values
            phi = df[col_deg].values
        else:
            z = df[col]
            phi = np.angle(z, deg=True)
        phi_max = max(max(np.abs(phi)), phi_max)
        c   = colors[i % len(colors)]
        ax[0,0].semilogx(f, 20*np.log10(np.abs(z)), color=c, label=label)
        ax[1,0].semilogx(f, phi, color=c, linestyle='--', label=label)
        ax[0,1].semilogx(f, np.real(z), color=c, label=label)
        ax[1,1].semilogx(f, np.imag(z), color=c, linestyle='--', label=label)
        if add_nyquist:
            ax[2,0].plot(np.real(z), -np.imag(z), color=c, label=label)
            ax[2,1].plot(np.real(z), -np.imag(z), color=c, label=label)

    ax[0,0].set_title(title)
    ax[0,0].legend(fontsize=8)
    ax[0,0].set_ylabel('|Z| (dBΩ)')
    ax[0,0].grid(True, which='both', ls='--', alpha=0.5)
    if bounds and (0, 0) in bounds.keys():
        ax[0,0].set_ylim(bounds[0,0][0], bounds[0,0][1])

    ax[1,0].set_xlabel('Frequency (Hz)')
    ax[1,0].set_ylabel('Phase (°)')
    ax[1,0].axhline(0, color='gray', lw=0.5, ls=':')
    ax[1,0].grid(True, which='both', ls='--', alpha=0.5)
    if bounds and (1, 0) in bounds.keys():
        ax[1,0].set_ylim(bounds[1,0][0], bounds[1,0][1])
    elif phi_max < 90:
        ax[1,0].set_ylim(-90, 90)
        ax[1,0].set_yticks(np.arange(-90, 90.1, 15))
    elif phi_max < 180:
        ax[1,0].set_ylim(-180, 180)
        ax[1,0].set_yticks(np.arange(-180, 180.1, 30))

    ax[0,1].set_ylabel('Resistance Re(Z) (Ω)')
    # ax[0,1].set_yscale('symlog')
    ax[0,1].axhline(0, color='gray', lw=0.5, ls=':')
    ax[0,1].grid(True, which='both', ls='--', alpha=0.5)
    if bounds and (0, 1) in bounds.keys():
        ax[0,1].set_ylim(bounds[0,1][0], bounds[0,1][1])

    ax[1,1].set_xlabel('Frequency (Hz)')
    ax[1,1].set_ylabel('Reactance Im(Z) (jΩ)')
    # ax[1,1].set_yscale('symlog')
    ax[1,1].axhline(0, color='gray', lw=0.5, ls=':')
    ax[1,1].grid(True, which='both', ls='--', alpha=0.5)
    if bounds and (1, 1) in bounds.keys():
        ax[1,1].set_ylim(bounds[1,1][0], bounds[1,1][1])

    if add_nyquist:
        ax[2,0].set_xlabel('Re(Z)')
        ax[2,0].set_ylabel('-Im(Z)')
        ax[2,0].axhline(0, color='gray', lw=0.5, ls=':')
        ax[2,0].axvline(0, color='gray', lw=0.5, ls=':')
        ax[2,0].axis('equal')
        ax[2,0].grid(True, which='both', ls='--', alpha=0.5)
        ax[2,0].set_title("Nyquist plane, full ->")

        ax[2,1].set_xlabel('Re(Z)')
        ax[2,1].set_ylabel('-Im(Z)')
        ax[2,1].axhline(0, color='gray', lw=0.5, ls=':')
        ax[2,1].axvline(0, color='gray', lw=0.5, ls=':')
        ax[2,1].axis('equal')
        ax[2,1].grid(True, which='both', ls='--', alpha=0.5)
        xylim = np.concatenate([ax[2,1].get_xlim(), ax[2,1].get_ylim()])
        ax[2,1].set_title(f"-> zoomed")
        ax[2,1].set_xlim(bounds[2,1][0], bounds[2,1][1])
        ax[2,1].set_ylim(bounds[2,1][2], bounds[2,1][3])

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
    plt.show()


# =============================================================================
# Main — calibration summary
# =============================================================================
if __name__ == '__main__':
    # --- Load calibration files ---
    df_11  = load_csv(PATH_RAW / r'1.1 with banana adapter, open, A=B (probe matching check).csv')
    df_12  = load_csv(PATH_RAW / r'1.2 no banana adapter, open, siggen=8Vp.csv')
    df_12b = load_csv(PATH_RAW / r'1.2b no adapter, bnc termination=50.05742R.csv')
    df_21  = load_csv(PATH_RAW / r'2.1 with banana adapter, open.csv')
    df_21c = load_csv(PATH_RAW / r'2.1c with banana adapter, Rload=50.8614, siggen=8Vp.csv')
    df_22  = load_csv(PATH_RAW / r'2.2 with banana adapter, shorted, siggen=8Vp.csv')
    df_31  = load_csv(PATH_RAW / r'3.1 mini-cable, open, siggen=8Vp.csv')
    df_32  = load_csv(PATH_RAW / r'3.2 mini-cable, shorted, siggen=8Vp.csv')
    df_41  = load_csv(PATH_RAW / r'4.1 mini-cable + yawl speaker.csv')

    # --- De-embed probes from measures ---
    deembed(df_12,  df_11, Rtest=RTEST)
    deembed(df_12b, df_11, Rtest=RTEST)
    deembed(df_21,  df_11, Rtest=RTEST)
    deembed(df_21c, df_11, Rtest=RTEST)
    deembed(df_22,  df_11, Rtest=RTEST)
    deembed(df_31,  df_11, Rtest=RTEST)
    deembed(df_32,  df_11, Rtest=RTEST)
    deembed(df_41,  df_11, Rtest=RTEST)

    # --- Accuracy check on known loads ---
    for ref_load_label, ref_load_df, Rref in [
        ('1.2b 50.057Ω (no adapter)', df_12b, RBNC_REF),
        ('2.1c 50.861Ω (with adapter)', df_21c, RLOAD_REF),
    ]:
        mask = (ref_load_df['ftarget_Hz'] > 5000) & (ref_load_df['ftarget_Hz'] < 100000)
        z_mean = ref_load_df['Z_corr'][mask].abs().mean()
        err  = (z_mean - Rref) / Rref * 100
        print(f'{ref_load_label}: Z_corr={z_mean:.4f}Ω, err={err:+.2f}%')
    print()

    if True:
        df_draw = pd.DataFrame()
        df_draw['ftarget_Hz'] = df_11[df_11['ftarget_Hz'] >= 0.5e6]['ftarget_Hz']
        jw = 1j*2*np.pi*df_draw['ftarget_Hz']
        df_draw['XC_TF'] = 1/(10**(-65.2/20) + jw * C_TF)
        df_draw['XC_TF_ADAPTER'] = 1 / (10 ** (-68.5 / 20) + jw * C_TF_ADAPTER)
        df_draw['XC_TF_MINICABLE'] = 1 / (10 ** (-65.2 / 20) + jw * C_TF_MINICABLE)

        # --- Plot calibration overview ---
        plot_z({
            # '1.2 open no adapter'   : (df_12, 'Z'),
            # '2.1 open with adapter' : (df_21, 'Z'),
            # '3.1 open mini-cable'   : (df_31, 'Z'),
            '1.2 open no adapter (corr)'   : (df_12, 'Z_corr'),
            '2.1 open with adapter (corr)' : (df_21, 'Z_corr'),
            '3.1 open mini-cable (corr)'   : (df_31, 'Z_corr'),
            f'capped {C_TF*1e12:.0f}pF': (df_draw, 'XC_TF'),
            f'capped {C_TF_ADAPTER*1e12:.0f}pF': (df_draw, 'XC_TF_ADAPTER'),
            f'capped {C_TF_MINICABLE*1e12:.0f}pF': (df_draw, 'XC_TF_MINICABLE'),
        },  title='Open, calibration sweeps — probes de-embedded',
            filename=PATH_POST_PROC/'open_cal_overview')
        plot_z({
            '1.2b 50Ω no adapter (corr)'   : (df_12b, 'Z_corr'),
            '2.1c 50Ω with adapter (corr)' : (df_21c, 'Z_corr'),
        },  title='Ref load, calibration sweeps — probes de-embedded',
            filename=PATH_POST_PROC/'ref_load_cal_overview')
        plot_z({
            '2.2 shorted adapter (corr)'   : (df_22, 'Z_corr'),
            '3.2 shorted mini-cable (corr)': (df_32, 'Z_corr'),
        }, title='Shorted, calibration sweeps — probes de-embedded',
           filename=PATH_POST_PROC/'shorted_cal_overview')

    # --- Short-load extraction, short-open-load extraction ---

    # Model is Zmeas = Zshort + Zdut//Zopen <=> Zdut = (Zmeas-Zshort) * (1 - (Zmeas-Zshort)/Zopen)
    # In the case Zopen measure is not usable or cannot be replaced by a model, then consider short-load extract only
    # SL extract is equivalent to Zopen == inf.

    # Perform extract for (4.1) : Zdut = speaker, and bench to deembed is test-fixture + adapter + mini-cable
    df_41['Zsl'] = df_41['Z_corr'] - df_32['Z_corr']
    df_41['Zsol'] = df_41['Zsl'] / (1 - df_41['Zsl'] / df_31['Z_corr'])
    plot_z({
        'Speaker raw measure (4.1)'  : (df_41, 'Z'),
        'Bench Zopen, probes deembedded (3.1 wrt 1.1)' : (df_31, 'Z_corr'),
        'Bench Zshort, probes deembedded (3.2 wrt 1.1)'  : (df_32, 'Z_corr'),
    },  title='Speaker raw measure w.r.t. mini-cable bench',
        filename=PATH_POST_PROC/'bench+speaker_overview',
        bounds={(0, 1): (0, 2000),
                (1, 1): (-2000, 2000)})
    plot_z(
        results_dict = {
            'Raw measure (4.1)'  : (df_41, 'Z'),
            'Probes deembedding (w.r.t. 1.1)'  : (df_41, 'Z_corr'),
            # 'Short-load extract (=4.1 w/o 3.2)'  : (df_41_c, 'Zsl'),
            'Bench deembedding (w.r.t. 1.1, 3.1, 3.2)' : (df_41, 'Zsol'),
        },
        title='Zspeaker impedance, full de-embedding',
        filename=PATH_POST_PROC/'speaker_overview',
        bounds= {(0, 1): (0, 50),
                 (1,1): (-25,50),
                 (2,1): (-20,30,-50,15), },
        add_nyquist=True)
    df_41.to_csv(PATH_POST_PROC/'4.1c yawl speaker stand-alone.csv')
    print(f"results saved to {PATH_POST_PROC}")
    print("Finished!")