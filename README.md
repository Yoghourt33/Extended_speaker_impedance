# Extended_speaker_impedance
### Speaker extended impedance (audio to HF), measurement and post-processing pipeline

This project contains:
- a PicoScope 2204A acquisition library & FFT sub-bin extraction,
- a signal generator wrapper,
- a measurement script (frequency sweep),
- a post-processing pipeline to de-embed probe mismatch, test fixtures, and extract DUT impedance

## Hardware
- PicoScope 2204A
- Sine generator with blip-free frequency change (mine >10MHz 8Vpk max)
- Test fixture with Rtest about 3.3R (precision resistor or kelvin-measured, metal film)
- BNC / banana adapter
- Mini-cable for direct connection to DUT (mine 2.5mm² 16cm multi-stranded non-audio cable)
- DUT: loudspeaker

## Workflow
1. Run frequency sweep acquisitions (`pico_sweep.py`) and check results
   1. stand-alone test fixture
      1. 1.1=both probes at B point and gnd of test fixture
      2. (optional) 1.2=probes at normal place, open. 1.2b=same as 1.2 with 50R BNC termination 
   2. (optional) test fixture + banana adapter only (open, 50R load, shorted)
   3. Test fixture + bnc-banana adapter + mini-cable
      1. mini-cable end open
      2. mini-cable end shorted
   4. Connect DUT (loudspeaker) to mini-cable
2. Post-process previous measures to extract DUT impedance and generate plots (`post_process.py`)

## Requirements
National instruments visa libraries or Keysight IO library, for signal generator control

See also `requirements.txt`

## Notes
- complex impedance correction.  Module-only subtraction |Va| - |Vb| would introduce a systematic phase error and ~40 to ~60dB/decade magnitude artifacts.
- pico_sweep.py et al.
  - sub-bin peak frequency and associated complex amplitude using Grandke (1983) with Hann window. Several other methods
  including Quinn (1994) were tested and rejected due to consistency/stability issues.
  - synchronized extract from A and B to preserve phase coherence
- Work on complex amplitudes for accurate impedance computation.
- post_process.py
  - produces Nyquist / Bode plots
  - The post-processing pipeline uses probe mismatch de-embedding based on the A=B reference measurement 1.1
  - Re(Z)<0 appearing after probe mismatch correction are treated as correction artifacts and documented in the code.
  - The final DUT extraction was validated by repeatability check.

This project is licensed under the GNU GPL v3.0 or later.
See the LICENSE file for details.

## Usage
```bash
python pico_sweep.py
python post_process.py
