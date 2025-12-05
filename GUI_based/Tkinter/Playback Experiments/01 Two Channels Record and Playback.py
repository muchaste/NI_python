# -*- coding: utf-8 -*-
"""
Simple two channel recording program with spectrogram option
Created on Thu Jul 13 16:43:45 2023

@author: Stefan Mucha
"""


import nidaqmx
from nidaqmx import constants
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import collections
import pandas as pd
import time
import gc
import pyarrow.feather as feather
# import pickle

# Get list of DAQ device names
daqSys = nidaqmx.system.System()
daqList = daqSys.devices.device_names

if len(daqList) == 0:
    raise ValueError('No DAQ detected, check connection.')
    
    
# Define additional helper functions for FFT and synthetic signal generation

def compute_dominant_frequency(data, sample_rate, min_freq, max_freq):
    """Compute the dominant frequency of a quasi-sinusoidal signal within specified bounds."""
    # Compute FFT and frequency bins
    fft_result = np.fft.rfft(data)
    frequencies = np.fft.rfftfreq(len(data), d=1/sample_rate)
    # Apply frequency bounds
    valid_indices = np.where((frequencies >= min_freq) & (frequencies <= max_freq))
    filtered_frequencies = frequencies[valid_indices]
    filtered_fft_result = np.abs(fft_result[valid_indices])
    # Find the frequency with the maximum FFT amplitude within the bounded range
    dominant_freq = filtered_frequencies[np.argmax(filtered_fft_result)]
    return dominant_freq


def generate_synthetic_signal(
    dominant_freq, 
    offset, 
    pre_stim_duration, 
    stimulus_duration, 
    post_stim_duration, 
    sample_rate, 
    amp_factor
):
    """Generate a synthetic signal with specified durations and amplitude modulation factor."""
    
    # Calculate the total number of samples for each phase
    pre_stim_samples = int(pre_stim_duration * sample_rate)
    stimulus_samples = int(stimulus_duration * sample_rate)
    post_stim_samples = int(post_stim_duration * sample_rate)
    
    # Create the pre-stimulus zeros
    pre_stim_signal = [0] * pre_stim_samples
    
    # Create the stimulus sine wave with amplitude modulation
    time_array = np.arange(stimulus_samples) / sample_rate
    frequency_with_offset = dominant_freq + offset
    amp_factor_array = np.repeat(float(amp_factor), stimulus_samples)
    gradient = np.arange(sample_rate)/sample_rate
    amp_factor_array[0:sample_rate] = amp_factor_array[0:sample_rate]*gradient
    amp_factor_array[len(amp_factor_array)-sample_rate:] = amp_factor_array[len(amp_factor_array)-sample_rate:]*gradient[::-1]
    stimulus_signal = amp_factor_array * np.sin(2 * np.pi * frequency_with_offset * time_array)
    
    # Create the post-stimulus zeros
    post_stim_signal = [0] * post_stim_samples
    
    # Combine all phases into the final signal
    synthetic_signal = np.concatenate((pre_stim_signal, stimulus_signal, post_stim_signal))
    
    # return synthetic_signal.tolist()
    return synthetic_signal

def generate_sine_wave(
    freq, 
    sample_rate, 
    amp_factor
):
    """Generate a synthetic signal with specified durations and amplitude modulation factor."""
    
    frequency = float(freq)
    amp = float(amp_factor)
    time_array = np.arange(sample_rate) / sample_rate
    sine_wave = amp * np.sin(2 * np.pi * frequency * time_array)
    
    return sine_wave


class VoltageContinuousInput(tk.Frame):
    
    def __init__(self, master):
        super().__init__(master)

        # Configure root tk class
        self.master = master
        self.master.title("eFish Playback Machine")
        self.master.geometry("1600x900")

        self.create_widgets()
        self.pack()
        self.run = False
        self.playback_active = False

    def create_widgets(self):
        # The main frame is made up of four subframes
        self.DAQSettingsFrame = DAQSettings(self, title="DAQ Settings")
        self.DAQSettingsFrame.grid(row=0, column=0, sticky="ew", pady=(20,0), padx=(20,20), ipady=10)
        
        self.plotSettingsFrame = PlotSettings(self, title="Plot Settings")
        self.plotSettingsFrame.grid(row=1, column=0, sticky="ew", pady=(20,0), padx=(20,20), ipady=10)
        
        self.ExperimentSettingsFrame = ExperimentSettings(self, title="Experiment Settings")
        self.ExperimentSettingsFrame.grid(row=2, column=0, sticky="ew", pady=(20,0), padx=(20,20), ipady=10)

        self.graphDataFrame = GraphData(self)
        self.graphDataFrame.grid(row=0, rowspan=3, column=1, pady=(20,0), ipady=10, sticky="ew")

    def start_acquisition(self):
        # Prevent the user from starting the task a second time
        self.ExperimentSettingsFrame.connect_button['state'] = 'disabled'
        self.ExperimentSettingsFrame.reset_button['state'] = 'disabled'
        self.ExperimentSettingsFrame.close_button['state'] = 'disabled'
        self.ExperimentSettingsFrame.playback_button['state'] = 'enabled'
        self.ExperimentSettingsFrame.signal_start_button['state'] = 'enabled'

        # Shared flag to alert task if it should stop
        self.continue_acquisition = True

        # Get task settings from the user
        self.input_channel1 = f"{self.DAQSettingsFrame.chosen_daq.get()}/{self.DAQSettingsFrame.input_channel1_entry.get()}"
        self.input_channel2 = f"{self.DAQSettingsFrame.chosen_daq.get()}/{self.DAQSettingsFrame.input_channel2_entry.get()}"
        self.max_voltage = 10
        self.min_voltage = -10
        self.daq_string = f"{self.DAQSettingsFrame.chosen_daq.get()}"

        self.sample_rate = int(self.DAQSettingsFrame.sample_rate_entry.get())
        self.refresh_rate = int(self.plotSettingsFrame.refresh_rate_entry.get())
        self.samples_to_plot = int(float(self.plotSettingsFrame.plot_duration_entry.get()) * self.sample_rate)
        self.spec_min_y = int(self.plotSettingsFrame.spec_min_freq_entry.get())
        self.spec_max_y = int(self.plotSettingsFrame.spec_max_freq_entry.get())
        self.plot_spec = int(self.plotSettingsFrame.plot_spectrogram.get())
        
        self.plot_buffer1 = collections.deque(maxlen=self.samples_to_plot)   # New, second buffer only for plotting (uses more memory but no if/else necessary later)
        self.plot_buffer2 = collections.deque(maxlen=self.samples_to_plot)   # New, second buffer only for plotting (uses more memory but no if/else necessary later)
        
        # Create and start tasks
        self.read_task = nidaqmx.Task()
        self.read_task.ai_channels.add_ai_voltage_chan(self.input_channel1, min_val=self.min_voltage, max_val=self.max_voltage, units=constants.VoltageUnits.VOLTS)
        self.read_task.ai_channels.add_ai_voltage_chan(self.input_channel2, min_val=self.min_voltage, max_val=self.max_voltage, units=constants.VoltageUnits.VOLTS)
        self.read_task.timing.cfg_samp_clk_timing(self.sample_rate, sample_mode=constants.AcquisitionType.CONTINUOUS)
        
        # Start the read task
        self.read_task.start()

        # Spin off call to check
        self.master.after(10, self.run_task)
        
    def run_task(self):
        # Check if the task needs to update the data queue and plot
        samples_available = self.read_task._in_stream.avail_samp_per_chan

        if samples_available >= int(self.sample_rate / self.refresh_rate):
            temp_data = self.read_task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)
            self.plot_buffer1.extend(temp_data[0])
            self.plot_buffer2.extend(temp_data[1])
            if self.playback_active:
                self.val_buffer1.extend(temp_data[0])
                self.val_buffer2.extend(temp_data[1])
                if time.time() - self.t_start > self.total_duration:
                    self.playback_active = False
                    self.read_task.stop()
                    self.write_task.stop()
                    self.write_task.close()
                    self.ExperimentSettingsFrame.playback_button['state'] = 'enabled'
                    self.ExperimentSettingsFrame.close_button['state'] = 'enabled'
                    # Save data to a file
                    time_vec = np.arange(0, len(self.val_buffer1) / (self.sample_rate / 1000), 1000 / self.sample_rate)
                    data_output = np.vstack((time_vec, self.val_buffer1, self.val_buffer2)).T
                    data_output = pd.DataFrame(data_output, columns=["Time [ms]", "ch 0", "ch 1"])
                    feather.write_feather(data_output, self.filepath)
                    self.save_log_file()
                    
                    # LATEST MODS; CHANGE IF NOT GOOD
                    del data_output, self.val_buffer1, self.val_buffer2
                    gc.collect()
                    messagebox.showinfo("Finished", "Recording complete")
                    self.read_task.start()

            self.graphDataFrame.ax1.cla()
            self.graphDataFrame.ax1.plot(self.plot_buffer1)
            self.graphDataFrame.ax1.set_title("Channel 1")
            self.graphDataFrame.ax1.set_ylabel("Voltage")
            self.graphDataFrame.ax2.cla()
            self.graphDataFrame.ax2.plot(self.plot_buffer2, 'r')
            self.graphDataFrame.ax2.set_title("Channel 2")
            self.graphDataFrame.ax2.set_ylabel("Voltage")
            self.graphDataFrame.ax2.set_xlabel("Sample")
            
            # Plot spectrogram only if checkbox is checked
            if self.plot_spec == 1:
                self.spec1_dom_f = compute_dominant_frequency(self.plot_buffer1, self.sample_rate, self.spec_min_y, self.spec_max_y)
                self.spec2_dom_f = compute_dominant_frequency(self.plot_buffer2, self.sample_rate, self.spec_min_y, self.spec_max_y)
                self.graphDataFrame.ax3.cla()
                self.graphDataFrame.ax3.specgram(self.plot_buffer1, Fs=self.sample_rate, NFFT=2**12)
                self.graphDataFrame.ax3.set_ylabel("Frequency")
                self.graphDataFrame.ax3.set_ylim(self.spec_min_y, self.spec_max_y)
                self.graphDataFrame.ax3.set_title(f"Channel 1 - Dominant Freq: {self.spec1_dom_f:.2f} Hz")
                self.graphDataFrame.ax4.cla()
                self.graphDataFrame.ax4.specgram(self.plot_buffer2, Fs=self.sample_rate, NFFT=2**12)
                self.graphDataFrame.ax4.set_ylabel("Frequency")
                self.graphDataFrame.ax4.set_xlabel("Sample")
                self.graphDataFrame.ax4.set_ylim(self.spec_min_y, self.spec_max_y)
                self.graphDataFrame.ax4.set_title(f"Channel 2 - Dominant Freq: {self.spec2_dom_f:.2f} Hz")
                
            self.graphDataFrame.graph.draw()

        # Check if the task should sleep or stop
        if self.continue_acquisition:
            self.master.after(10, self.run_task)
        else:
            self.read_task.stop()
            self.read_task.close()
            self.ExperimentSettingsFrame.connect_button['state'] = 'enabled'
            self.ExperimentSettingsFrame.close_button['state'] = 'enabled'
            
    def playback_record(self):
        
        # Prevent the user from starting the task a second time
        self.ExperimentSettingsFrame.playback_button['state'] = 'disabled'
        
        # Set filepath for saving
        self.filepath = filedialog.asksaveasfilename(defaultextension=".feather", filetypes=[("Feather files", "*.feather")])
        
        if not self.filepath:
            self.ExperimentSettingsFrame.playback_button['state'] = 'enabled'
            return
        else:
            # Shared flag to alert task if it should stop
            self.playback_active = True
            
            # Get task settings from the user
            self.output_channel1 = f"{self.DAQSettingsFrame.chosen_daq.get()}/{self.DAQSettingsFrame.output_channel1_entry.get()}"
            # Parse stimulus parameters
            self.pre_stimulus_duration = int(self.ExperimentSettingsFrame.pre_stimulus_entry.get())
            self.stimulus_duration = int(self.ExperimentSettingsFrame.stimulus_entry.get())
            self.post_stimulus_duration = int(self.ExperimentSettingsFrame.post_stimulus_entry.get())
            self.offset_frequency = float(self.ExperimentSettingsFrame.offset_var.get())
            self.amp_factor = float(self.ExperimentSettingsFrame.amp_factor_entry.get())        
            self.total_duration = self.pre_stimulus_duration + self.stimulus_duration + self.post_stimulus_duration
            self.number_of_samples = self.total_duration * self.sample_rate
            self.val_buffer1 = collections.deque(maxlen=self.number_of_samples)
            self.val_buffer2 = collections.deque(maxlen=self.number_of_samples)
            # self.val_buffer3 = collections.deque(maxlen=self.number_of_samples)
    
            # Playback generation
            # FFT analysis on the latest segment
            self.dominant_freq = compute_dominant_frequency(self.plot_buffer1, self.sample_rate, self.spec_min_y, self.spec_max_y)
            self.playback_signal = generate_synthetic_signal(self.dominant_freq, self.offset_frequency, self.pre_stimulus_duration, 
                                                             self.stimulus_duration, self.post_stimulus_duration, self.sample_rate,
                                                             self.amp_factor)
    
            self.write_task = nidaqmx.Task()
            self.write_task.ao_channels.add_ao_voltage_chan(self.output_channel1, min_val=self.min_voltage, max_val=self.max_voltage, units=constants.VoltageUnits.VOLTS)
            self.write_task.timing.cfg_samp_clk_timing(self.sample_rate, source="/" + self.daq_string + "/ai/SampleClock",
                                                       sample_mode=constants.AcquisitionType.CONTINUOUS, samps_per_chan=self.number_of_samples)
    
            # Start write task first, waiting for the analog input sample clock
            self.write_task.write(self.playback_signal, auto_start=False)
            self.write_task.start()
            self.t_start = time.time()
        
    def start_signal_output(self):
        
        # Prevent the user from starting the task a second time
        self.ExperimentSettingsFrame.signal_start_button['state'] = 'disabled'
        self.ExperimentSettingsFrame.signal_stop_button['state'] = 'enabled'
        self.ExperimentSettingsFrame.playback_button['state'] = 'disabled'
        
        # Get task settings from the user
        self.amp_factor = float(self.ExperimentSettingsFrame.amp_factor_entry.get())        
        self.output_channel1 = f"{self.DAQSettingsFrame.chosen_daq.get()}/{self.DAQSettingsFrame.output_channel1_entry.get()}"

        test_signal = generate_sine_wave(1000, self.sample_rate, self.amp_factor)
        
        self.calib_task = nidaqmx.Task()
        self.calib_task.ao_channels.add_ao_voltage_chan(self.output_channel1, min_val=self.min_voltage, max_val=self.max_voltage, units=constants.VoltageUnits.VOLTS)
        self.calib_task.timing.cfg_samp_clk_timing(self.sample_rate, source="/" + self.daq_string + "/ai/SampleClock",
                                                   sample_mode=constants.AcquisitionType.CONTINUOUS)
        
        # Start write task first, waiting for the analog input sample clock
        self.calib_task.write(test_signal, auto_start=True)
        # self.calib_task.start()
        
    def stop_signal_output(self):
        # Prevent the user from starting the task a second time
        self.ExperimentSettingsFrame.signal_start_button['state'] = 'enabled'
        self.ExperimentSettingsFrame.signal_stop_button['state'] = 'disabled'
        self.ExperimentSettingsFrame.playback_button['state'] = 'enabled'
        
        self.calib_task.stop()

        zero_signal = np.repeat(0, self.sample_rate*0.1)
        self.calib_task.write(zero_signal, auto_start=True)
        # self.calib_task.start()
        self.calib_task.stop()
        self.calib_task.close()


    def stop_acquisition(self):
        # Callback for the "stop task" button
        self.continue_acquisition = False
        self.ExperimentSettingsFrame.reset_button['state'] = 'enabled'
        self.ExperimentSettingsFrame.signal_start_button['state'] = 'disabled'
        
        if self.playback_active:
            self.playback_active = False
            zero_signal = np.repeat(0, self.sample_rate*0.1)
            self.write_task.stop()
            self.write_task.write(zero_signal, auto_start=True)
            # self.calib_task.start()
            self.write_task.stop()
            self.write_task.close()
        
    def save_log_file(self):
        log_filename = f"log_{self.filepath.split('/')[-1].split('.')[0]}.txt"
        log_filepath = "/".join(self.filepath.split('/')[:-1]) + "/" + log_filename
        
        log_data = {
            "Number of Input Channels": 2,  # Based on the channels defined in the script
            "Number of Output Channels": 1,  # Based on the output channels defined in the script
            "Sample Rate": self.sample_rate,
            "Fish ID": self.ExperimentSettingsFrame.fish_entry.get(),
            "Temperature": self.ExperimentSettingsFrame.temp_entry.get(),
            "Conductivity": self.ExperimentSettingsFrame.cond_entry.get(),
            "Dominant frequency": self.dominant_freq,
            "Frequency Offset": self.offset_frequency,
            "Min Frequency": self.spec_min_y,
            "Max Frequency": self.spec_max_y,
            "Stimulus Duration": self.stimulus_duration,
            "Pre Stimulus Duration": self.pre_stimulus_duration,
            "Post Stimulus Duration": self.post_stimulus_duration,
            "Amplitude Factor": self.amp_factor,
            "Path to Datafile": self.filepath
        }
        
        with open(log_filepath, 'w') as log_file:
            for key, value in log_data.items():
                log_file.write(f"{key}: {value}\n")

    def reset_device(self):
        daq = nidaqmx.system.Device(self.DAQSettingsFrame.chosen_daq.get())
        daq.reset_device()


class DAQSettings(tk.LabelFrame):

    def __init__(self, parent, title):
        super().__init__(parent, text=title, labelanchor='n')
        self.parent = parent
        self.x_padding = (10, 10)
        self.create_widgets()

    def create_widgets(self):
        self.chosen_daq = tk.StringVar()
        self.chosen_daq.set(daqList[0])

        self.daq_selection_label = ttk.Label(self, text="Select DAQ")
        self.daq_selection_label.grid(row=0, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.daq_selection_menu = ttk.OptionMenu(self, self.chosen_daq, daqList[0], *daqList)
        s = ttk.Style()
        s.configure("TMenubutton", background="white")
        self.daq_selection_menu.grid(row=0, column=1, sticky='w', padx=self.x_padding)

        self.input_channel_label = ttk.Label(self, text="Input Channels")
        self.input_channel_label.grid(row=1, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.input_channel1_entry = ttk.Entry(self, width=10)
        self.input_channel1_entry.insert(0, "ai5")
        self.input_channel1_entry.grid(row=1, column=1, sticky='w', padx=self.x_padding)
        
        self.input_channel2_entry = ttk.Entry(self, width=10)
        self.input_channel2_entry.insert(0, "ai2")
        self.input_channel2_entry.grid(row=2, column=1, sticky='w', padx=self.x_padding)
        
        self.output_channel_label = ttk.Label(self, text="Output Channel")
        self.output_channel_label.grid(row=3, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.output_channel1_entry = ttk.Entry(self, width=10)
        self.output_channel1_entry.insert(0, "ao0")
        self.output_channel1_entry.grid(row=3, column=1, columnspan=2, sticky='w', padx=self.x_padding)
        
        self.sample_rate_label = ttk.Label(self, text="Sample Rate (Hz)")
        self.sample_rate_label.grid(row=4, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.sample_rate_entry = ttk.Entry(self, width=10)
        self.sample_rate_entry.insert(0, "40000")
        self.sample_rate_entry.grid(row=4, column=1, sticky='w', padx=self.x_padding)


class PlotSettings(tk.LabelFrame):

    def __init__(self, parent, title):
        super().__init__(parent, text=title, labelanchor='n')
        self.parent = parent
        self.x_padding = (10, 10)
        self.create_widgets()

    def create_widgets(self):
        self.plot_spectrogram = tk.IntVar()
        self.plot_spectrogram.set(1)
        
        self.plot_duration_label = ttk.Label(self, text="Plot duration")
        self.plot_duration_label.grid(row=0, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.plot_duration_entry = ttk.Entry(self, width=10)
        self.plot_duration_entry.insert(0, "1")
        self.plot_duration_entry.grid(row=0, column=1, sticky='w', padx=self.x_padding)

        self.refresh_rate_label = ttk.Label(self, text="Refresh Rate")
        self.refresh_rate_label.grid(row=1, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.refresh_rate_entry = ttk.Entry(self, width=10)
        self.refresh_rate_entry.insert(0, "10")
        self.refresh_rate_entry.grid(row=1, column=1, sticky='w', padx=self.x_padding)
        
        self.spec_min_freq_label = ttk.Label(self, text="Min Freq (Hz)")
        self.spec_min_freq_label.grid(row=2, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.spec_min_freq_entry = ttk.Entry(self, width=10)
        self.spec_min_freq_entry.insert(0, "100")
        self.spec_min_freq_entry.grid(row=2, column=1, sticky='w', padx=self.x_padding)

        self.spec_max_freq_label = ttk.Label(self, text="Max Freq (Hz)")
        self.spec_max_freq_label.grid(row=3, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.spec_max_freq_entry = ttk.Entry(self, width=10)
        self.spec_max_freq_entry.insert(0, "1400")
        self.spec_max_freq_entry.grid(row=3, column=1, sticky='w', padx=self.x_padding)
        
        self.spec_plot_check = ttk.Checkbutton(self, text='Plot Spectrogram', variable=self.plot_spectrogram)
        self.spec_plot_check.grid(row=4, column=0, columnspan=2, sticky='w', padx=self.x_padding, pady=(10,0))
        # self.spec_plot_check.select()
        

# Playback Settings Frame

class ExperimentSettings(tk.LabelFrame):
    """Frame to set playback parameters: frequency offset, stimulus duration, and pre/post durations."""
    
    def __init__(self, parent, title):
        super().__init__(parent, text=title, labelanchor='n')
        self.parent = parent
        self.x_padding = (10, 10)
        self.create_widgets()
        
    def create_widgets(self):
        # Fish ID
        self.fish_label = tk.Label(self, text="Fish ID")
        self.fish_label.grid(row=0, column=0, padx=self.x_padding, pady=5, sticky="w")
        self.fish_entry = tk.Entry(self, width=10)
        self.fish_entry.grid(row=0, column=1, padx=self.x_padding, pady=5, sticky="w")
        
        # Temperature
        self.temp_label = tk.Label(self, text="Temp. (C)")
        self.temp_label.grid(row=1, column=0, padx=self.x_padding, pady=5, sticky="w")
        self.temp_entry = tk.Entry(self, width=10)
        self.temp_entry.grid(row=1, column=1, padx=self.x_padding, pady=5, sticky="w")

        # Conductivity
        self.cond_label = tk.Label(self, text="Cond. (uS)")
        self.cond_label.grid(row=2, column=0, padx=self.x_padding, pady=5, sticky="w")
        self.cond_entry = tk.Entry(self, width=10)
        self.cond_entry.grid(row=2, column=1, padx=self.x_padding, pady=5, sticky="w")

        # Frequency offset selection
        self.offset_label = tk.Label(self, text="Frequency Offset (Hz)")
        self.offset_label.grid(row=3, column=0, padx=self.x_padding, pady=5, sticky="w")
        self.offset_var = tk.StringVar(value="0")
        self.offset_menu = ttk.Combobox(self, textvariable=self.offset_var, values=["-200", "-100", "-20", "-10", "-5", "-2.5", "0", "2.5", "5", "10", "20", "100", "200"], width=7)
        self.offset_menu.grid(row=3, column=1, padx=self.x_padding, pady=5, sticky="w")
        
        # Pre-stimulus duration
        self.pre_stimulus_label = tk.Label(self, text="Pre-Stim Duration (s)")
        self.pre_stimulus_label.grid(row=4, column=0, padx=self.x_padding, pady=5, sticky="w")
        self.pre_stimulus_entry = tk.Entry(self, width=10)
        self.pre_stimulus_entry.insert(0, "60")
        self.pre_stimulus_entry.grid(row=4, column=1, padx=self.x_padding, pady=5, sticky="w")

        # Stimulus duration input
        self.stimulus_label = tk.Label(self, text="Stimulus Duration (s)")
        self.stimulus_label.grid(row=5, column=0, padx=self.x_padding, pady=5, sticky="w")
        self.stimulus_entry = tk.Entry(self, width=10)
        self.stimulus_entry.insert(0, "60")
        self.stimulus_entry.grid(row=5, column=1, padx=self.x_padding, pady=5, sticky="w")

        # Post-stimulus duration
        self.post_stimulus_label = tk.Label(self, text="Post-Stim Duration (s)")
        self.post_stimulus_label.grid(row=6, column=0, padx=self.x_padding, pady=5, sticky="w")
        self.post_stimulus_entry = tk.Entry(self, width=10)
        self.post_stimulus_entry.insert(0, "60")
        self.post_stimulus_entry.grid(row=6, column=1, padx=self.x_padding, pady=5, sticky="w")
        
        # Amplitude modulation factor
        self.amp_factor_label = tk.Label(self, text="Amplitude Factor")
        self.amp_factor_label.grid(row=7, column=0, padx=self.x_padding, pady=5, sticky="w")
        self.amp_factor_entry = tk.Entry(self, width=10)
        self.amp_factor_entry.insert(0, "0.1")
        self.amp_factor_entry.grid(row=7, column=1, padx=self.x_padding, pady=5, sticky="w")
        
        # Connect button
        self.connect_button = ttk.Button(self, text="Start", command=self.parent.start_acquisition)
        self.connect_button.grid(row=8, column=0, sticky='ew', padx=self.x_padding, pady=(10, 0))
        
        # Disconnect button
        self.disconnect_button = ttk.Button(self, text="Stop", command=self.parent.stop_acquisition)
        self.disconnect_button.grid(row=8, column=1, sticky='ew', padx=self.x_padding, pady=(10, 0))
        
        # Test signal start button
        self.signal_start_button = ttk.Button(self, text="Test Signal On", command=self.parent.start_signal_output)
        self.signal_start_button.grid(row=9, column=0, sticky='ew', padx=self.x_padding, pady=(10, 0))
        self.signal_start_button['state'] = 'disabled'
        
        # Test signal stop button
        self.signal_stop_button = ttk.Button(self, text="Test Signal Off", command=self.parent.stop_signal_output)
        self.signal_stop_button.grid(row=9, column=1, sticky='ew', padx=self.x_padding, pady=(10, 0))
        self.signal_stop_button['state'] = 'disabled'
                
        # Playback button
        self.playback_button = ttk.Button(self, text="Start Playback", command=self.parent.playback_record)
        self.playback_button.grid(row=10, column=0, columnspan=2, sticky='ew', padx=self.x_padding, pady=(10, 0))
        self.playback_button['state'] = 'disabled'
        
        # Reset button
        self.reset_button = ttk.Button(self, text="Reset DAQ", command=self.parent.reset_device)
        self.reset_button.grid(row=11, column=0, sticky='ew', padx=self.x_padding, pady=(10, 0))
        
        # Close button
        self.close_button = ttk.Button(self, text="Close", command=root.destroy)
        self.close_button.grid(row=11, column=1, sticky='ew', padx=self.x_padding, pady=(10, 0))


class GraphData(tk.Frame):

    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()

    def create_widgets(self):
        self.graph_title = ttk.Label(self, text="Voltage Input")
        self.fig = Figure(figsize=(13, 8), dpi=100)
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax1.set_title("Channel 1")
        self.ax1.set_ylabel("Voltage")
        self.ax2 = self.fig.add_subplot(2, 2, 3)
        self.ax2.set_title("Channel 2")
        self.ax2.set_ylabel("Voltage")
        self.ax2.set_xlabel("Sample")
        self.ax3 = self.fig.add_subplot(2, 2, 2)
        self.ax3.set_title("Channel 1")
        self.ax3.set_ylabel("Frequency")
        self.ax4 = self.fig.add_subplot(2, 2, 4)
        self.ax4.set_title("Channel 2")
        self.ax4.set_ylabel("Frequency")
        self.ax4.set_xlabel("Sample")
        self.graph = FigureCanvasTkAgg(self.fig, self)
        self.graph.draw()
        self.graph.get_tk_widget().pack()


# Create the Tkinter class and primary application "VoltageContinuousInput"
root = tk.Tk()
app = VoltageContinuousInput(root)

# Start the application
app.mainloop()
