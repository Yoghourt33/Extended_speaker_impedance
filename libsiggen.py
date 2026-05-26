# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Yoghourt (Yoghourt33 on github)
"""
Simple PyVISA driver for a ###### signal generator.

Purpose:
- Connect to the instrument
- Configure impedance setting
- Generate a sine wave with a given frequency and amplitude
- Change sine wave frequency w/o blip
- Disable output on closure of connection

Notes:
- This driver assumes channel 1 by default.
- The amplitude is interpreted as peak-to-peak voltage.
- The effective output load of generator is fixed 50R. set_impedance() effect is on signal generator display
"""

import pyvisa


class SigGenDriver:
    """Minimal driver for a #### signal generator."""

    def __init__(self, resource_name: str, channel: int = 1, timeout_ms: int = 5000):
        """
        Create a driver instance and open the VISA resource.

        Args:
            resource_name: VISA resource string, for example "USB0::0x0957::0x2C07::MYXXXXXXXX::INSTR"
            channel: Output channel to control, typically 1 or 2
            timeout_ms: VISA timeout in milliseconds
        """

        self.resource_name = resource_name
        self.channel = channel
        self.timeout_ms = timeout_ms
        self.rm = pyvisa.ResourceManager()
        self.inst = self.rm.open_resource(self.resource_name)
        self.inst.timeout = self.timeout_ms
        self.inst.write_termination = "\n"
        self.inst.read_termination = "\n"

        # Put the output into a known impedance configuration.
        self.set_impedance(50)

    def close(self) -> None:

        """Close the VISA session cleanly."""
        self.enable_output(False)
        if self.inst is not None:
            self.inst.close()

        if self.rm is not None:
            self.rm.close()

    def idn(self) -> str:

        """Return the instrument identification string."""

        return self.inst.query("*IDN?").strip()

    def set_impedance(self, load=50) -> None:
        """Configure the display according to specified load."""

        self.inst.write(f"OUTP{self.channel}:LOAD {load}")

    def set_freq(self, frequency_hz: float) -> None:
        self.inst.write(f"SOUR{self.channel}:FREQ {frequency_hz}")

    def set_sine(self, frequency_hz: float, amplitude_vpp: float, offset_v: float = 0.0) -> None:
        """
        Configure the selected channel to output a sine wave.

        Args:
            frequency_hz: Sine frequency in Hz
            amplitude_vpp: Sine amplitude in volts peak-to-peak
            offset_v: DC offset in volts
        """

        # Configure waveform type and basic parameters.
        self.inst.write(f"SOUR{self.channel}:FUNC SIN")
        self.set_freq(frequency_hz)
        self.inst.write(f"SOUR{self.channel}:VOLT {amplitude_vpp}")
        self.inst.write(f"SOUR{self.channel}:VOLT:OFFS {offset_v}")

    def enable_output(self, enable=True) -> None:
        """Enable the selected channel output."""

        self.inst.write(f"OUTP{self.channel} {'ON' if enable else 'OFF'}")



if __name__ == "__main__":

    # Example usage:

    # Replace the VISA resource string with your actual instrument resource.

    DRIVER_RESOURCE = "USB0::0x0957::0x5707::MY59000328::0::INSTR"

    siggen = SigGenDriver(DRIVER_RESOURCE, channel=1)

    try:
        print("Connected to:", siggen.idn())
        siggen.set_impedance(50)
        siggen.set_sine(frequency_hz=1e6, amplitude_vpp=2.0)
        siggen.enable_output(True)
        print("Sine output enabled.")

    finally:
        # Always close the session when done.
        siggen.close()
