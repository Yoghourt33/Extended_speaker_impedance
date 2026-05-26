# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Yoghourt (Yoghourt33 on github)

from pathlib import Path
import numpy as np

debug = False   # for quick'n'dirty execution, added verbosity...

# Datapaths, equipment path
# =========================
PATH_RAW = Path("data/Raw_measures")
PATH_POST_PROC = Path("data/Post-processed")
SIGGEN_ADDR="USB0::0x0957::0x5707::xxxxxxxx::0::INSTR"

# Sweeps configuration
# ====================
FSTART = 1e3

# Picoscope 2204A max sampling rate with 2 channels enabled = 50MSa/s => fstop_max < 50M/N with N = nb of samples/cycle
# In practice, 8Sa/cycle generate discrepancies (from insufficient sampling accuray, FFT, sub-bin interp, whatever...)
# N=16 --> max fstop = 3.125MHz
# N=32 --> max fstop ~1.56MHz
# N=64 --> max fstop ~780kHz
SAMPLES_PER_CYCLE = 16
FSTOP = 50e6 / SAMPLES_PER_CYCLE
NB_FSTEPS = round((np.log10(FSTOP)-np.log10(FSTART))*(200 if not debug else 10)) + 1
TARGET_FREQS = np.clip(np.logspace(np.log10(FSTART), np.log10(FSTOP), NB_FSTEPS), FSTART, FSTOP)

# Measure averaging (number of scope traces that will be acquired)
N_AVG = 100 if not debug else 9

# Scope timebase rough control, equivalent to desired number of cycles to acquire in each trace
# Picoscope 2204A max memory is 8k / 2 channels - 128 = 3968 samples. Compromise is amplitude precision (Sa/cycle++)
# vs. frequency precision (nb_cycles++)
# Effective nb of cycles will depend on effective sampling frequency, and fft-friendly target of 2^k samples
CYCLES_TO_CAPTURE = int(3968/SAMPLES_PER_CYCLE)
BW_MIN_HZ = 5.0

# Probes mode. It is HIGHLY recommended to use x10 mode, and perform compensation check before measuring anything.
# Rationale :
# - At scope end, probed voltage will appear coming from a 1M/(1+9MR) resistor bridge even at HF
# - At test fixture end, Zprobe=10MR//(15pF/10) -> |Zprobe|~10kR@10MHz => impact not measurable
PROBE_X1_X10 = 10   # probe factor will be applied during post-processing to recover probe attenuation

COUPLING = 'DC'  # fft extract will raise an error it detects DC>50mV

IS_SPEAKER_NOT_ELECTRICAL_ONLY = True
if IS_SPEAKER_NOT_ELECTRICAL_ONLY:
    VPEAK_AMPLITUDE = 2
else:
    VPEAK_AMPLITUDE = 8
    COUPLING = 'DC'

# Post-processing constants
# =========================
"""
Known physical constants :
    -- measured 4-wire (kelvin connection)
    Rtest        =  3.3051  Ω ± 0.0043 Ω (measure from A to B across Rtest)
    Rbnc_ref     = 50.05742 Ω ± 0.008  Ω (measure from B to gnd)
    Rload_ref    = 50.8614  Ω ± 0.008  Ω (measure from B to gnd)
    R_strap      = 11.62 mΩ              (strap across BNC-banana adapter, measure from B point to gnd)
    R_mini_cable = 16.87 mΩ              (16cm 2.5mm² mini-cable DC resistance, B point to gnd with mini-cable shorted)
                                         -> 16.87-11.62=5.25mR for stand-alone mini-cable
                                         => higher than theoretical 2.2mR = 17.4e-9*2*0.16/2.5e-6, probably from contact
                                         resistances
    -- measured 2-wire
	Meter accuracy@30pF ~0.5pF. Observed variability is rather 5 to 15pF! Below measured stabilized by averaging.
	Ctf            = 16pF     (measured from B to GND, test fixture is stand-alone)
	Ctf_adapter    = 21pF     (from B to gnd, test fixture + bnc-banana adapter at B side)
	Ctf_2adapters  = 28pF     (from B to gnd, test fixture + bnc-banana adapters at both sides)
	                          -> standalone bnc-banana adapter effective capacitance 6pF
	Cttf_minicable = 36pF     (from B to gnd, test fixture + adapter + mini-cable left open)
	                          -> standalone mini-cable effective capacitance 14pF
	                          => ok with theoretical 8-18pF = pi*Eo*Er*0.16/acosh(d/2r)
	                          with E0=8.854e-12 vacuum permittivity
	                               Er=2.2 to 4.8 relative permittivity of PVC + some air
	                               d = conductors centre distance, estimated at 4*r
	                               r = conductor radius = sqrt(2.5e-6/pi)
"""

# =============================================================================
# Constants
# =============================================================================
RTEST       = 3.3051    # Ω, 4-wire measurement of test resistor (metal film type), in situ point A to point B
RBNC_REF    = 50.05742  # Ω, 4-wire measurement of BNC stub, from B to gnd
RLOAD_REF   = 50.8614   # Ω, 4-wire delta measurement (54.1665 - Rtest), from A to gnd
C_TF           = 16e-12 # F, 2-wire measure with calibrated meter in capa mode, for correlation with calibration results
C_TF_ADAPTER   = 22e-12 # F, 2-wire measure with calibrated meter in capa mode, for correlation with calibration results
C_TF_MINICABLE = 36e-12 # F, 2-wire measure with calibrated meter in capa mode, for correlation with calibration results
