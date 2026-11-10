# -*- coding: utf-8 -*-
"""
National Instruments Analog Output Sine Wave Generator

This program generates a continuous sine wave on an analog output channel
using a National Instruments device. The program waits for user input
to stop the sine wave output.

Author: Stefan Mucha_admin
"""

import nidaqmx
from nidaqmx import constants
import numpy as np

# Detect available National Instruments DAQ devices
system = nidaqmx.system.System.local()
if not system.devices:
    raise RuntimeError("No NI DAQ device detected. Exiting...")

device_name = system.devices[0].name  # Select the first available device

# Configuration parameters
output_channels = 1         # Number of output channels
output_channel_name = 'ao0' # Channel name on the NI device
output_fs = 100000          # Sampling frequency (Hz)
f = 1000                    # Frequency of the signal (Hz)
n_vals = output_fs // f     # Number of values per period
amp_factor = 1.8            # Amplitude factor for the sine wave
offset = 1.8                # Offset for the sine wave

# If amp factor > offset, stop with a warning
if amp_factor > offset:
    raise RuntimeError("Amplitude factor cannot be greater than offset. Exiting...")

# Generate sine wave output data
x = np.arange(n_vals)  # X-axis points for plotting
out_data = amp_factor * np.sin(2 * np.pi * f * (x / output_fs)) + offset  # Sine wave values

# Setup write task
write_task = nidaqmx.Task(new_task_name="out")
write_task.ao_channels.add_ao_voltage_chan(
    physical_channel=f"{device_name}/{output_channel_name}",
    min_val=-10,
    max_val=10,
    units=constants.VoltageUnits.VOLTS
)

write_task.timing.cfg_samp_clk_timing(
    rate=output_fs,
    sample_mode=constants.AcquisitionType.CONTINUOUS,
    samps_per_chan=n_vals
)

# Start the write task
write_task.write(out_data, auto_start=True)
print("Sine wave output started. Press Enter to stop...")

# Wait for user input to stop the output
input()

# Stop and close the write task
write_task.stop()
# Set output voltage to offset value
write_task.write(np.repeat(offset, n_vals), auto_start=True)
write_task.stop()
write_task.close()
print("Sine wave output stopped.")
