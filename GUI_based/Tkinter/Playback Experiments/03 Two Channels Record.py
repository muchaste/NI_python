# -*- coding: utf-8 -*-
"""
Simple two channel recording program with spectrogram option
Created on Mon Feb 17 16:21:29 2025

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

# Get list of DAQ device names
daqSys = nidaqmx.system.System()
daqList = daqSys.devices.device_names

if len(daqList) == 0:
    raise ValueError('No DAQ detected, check connection.')
    
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
    
class VoltageContinuousInput(tk.Frame):
    
    def __init__(self, master):
        super().__init__(master)

        # Configure root tk class
        self.master = master
        self.master.title("eFish Recorder")
        self.master.geometry("1600x900")

        self.create_widgets()
        self.pack()
        self.run = False
        self.recording_active = False

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
        self.ExperimentSettingsFrame.record_button['state'] = 'enabled'
        
        # Prevent modification in entry fields
        self.DAQSettingsFrame.daq_selection_menu.state(['disabled'])
        self.DAQSettingsFrame.input_channel1_entry.state(['disabled'])
        self.DAQSettingsFrame.input_channel2_entry.state(['disabled'])
        self.DAQSettingsFrame.sample_rate_entry.state(['disabled'])
        self.plotSettingsFrame.plot_duration_entry.state(['disabled'])
        self.plotSettingsFrame.refresh_rate_entry.state(['disabled'])
        self.plotSettingsFrame.spec_min_freq_entry.state(['disabled'])
        self.plotSettingsFrame.spec_max_freq_entry.state(['disabled'])
        self.plotSettingsFrame.spec_plot_check.state(['disabled'])

        # Shared flag to alert task if it should stop
        self.continue_acquisition = True

        # Get task settings from the user
        self.input_channel1 = f"{self.DAQSettingsFrame.chosen_daq.get()}/{self.DAQSettingsFrame.input_channel1_entry.get()}"
        self.input_channel2 = f"{self.DAQSettingsFrame.chosen_daq.get()}/{self.DAQSettingsFrame.input_channel2_entry.get()}"
        self.output_channel1 = f"{self.DAQSettingsFrame.chosen_daq.get()}/{self.DAQSettingsFrame.output_channel1_entry.get()}"
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
        self.write_task = nidaqmx.Task()
        self.write_task.do_channels.add_do_chan(self.output_channel1)
        
        
        # Start the tasks
        self.read_task.start()
        self.write_task.start()

        # Spin off call to check
        self.master.after(10, self.run_task)
        
    def run_task(self):
        # Check if the task needs to update the data queue and plot
        samples_available = self.read_task._in_stream.avail_samp_per_chan

        if samples_available >= int(self.sample_rate / self.refresh_rate):
            temp_data = self.read_task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)
            self.plot_buffer1.extend(temp_data[0])
            self.plot_buffer2.extend(temp_data[1])
            if self.recording_active:
                self.val_buffer1.extend(temp_data[0])
                self.val_buffer2.extend(temp_data[1])
                if time.time() - self.t_start > self.rec_dur:
                    self.recording_active = False
                    self.read_task.stop()
                    self.write_task.write(False)
                    self.ExperimentSettingsFrame.record_button['state'] = 'enabled'
                    self.ExperimentSettingsFrame.close_button['state'] = 'enabled'
                    # Save data to a file
                    time_vec = np.arange(0, len(self.val_buffer1) / (self.sample_rate / 1000), 1000 / self.sample_rate)
                    data_output = np.vstack((time_vec, self.val_buffer1, self.val_buffer2)).T
                    data_output = pd.DataFrame(data_output, columns=["time_ms", "ch1", "ch2"])
                    # data_output.to_csv(self.filepath+"csv", index=None, sep=';', decimal=",", mode='w')
                    feather.write_feather(data_output, self.filepath)
                    self.save_log_file()
                    
                    # Show the end of measurement in a popup
                    #*** THIS IS NEW; CHANGE IF NOT GOOD***
                    messagebox.showinfo("Finished")
                    # LATEST MODS; CHANGE IF NOT GOOD
                    del data_output, self.val_buffer1, self.val_buffer2
                    gc.collect()
                    self.read_task.start()

            self.graphDataFrame.ax1.cla()
            self.graphDataFrame.ax1.plot(self.plot_buffer1)
            if self.recording_active:
                self.graphDataFrame.ax1.scatter(self.samples_to_plot, 0, s=200, color='green', clip_on=False)
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
            
    def start_record(self):
        
        # Prevent the user from starting the task a second time
        self.ExperimentSettingsFrame.record_button['state'] = 'disabled'
        
        # Set filepath for saving
        self.filepath = filedialog.asksaveasfilename(defaultextension=".feather", filetypes=[("Feather files", "*.feather")])
        
        if not self.filepath:
            self.ExperimentSettingsFrame.record_button['state'] = 'enabled'
            return
        else:
            # Shared flag to alert task if it should stop
            self.recording_active = True
            self.write_task.write(True)
            self.rec_dur = float(self.ExperimentSettingsFrame.rec_dur_entry.get())
            self.number_of_samples = int(self.rec_dur * self.sample_rate)
            self.val_buffer1 = collections.deque(maxlen=self.number_of_samples)
            self.val_buffer2 = collections.deque(maxlen=self.number_of_samples)
    
            self.t_start = time.time()

    def stop_acquisition(self):
        # Callback for the "stop task" button
        self.continue_acquisition = False
        self.ExperimentSettingsFrame.reset_button['state'] = 'enabled'
        self.ExperimentSettingsFrame.record_button['state'] = 'disabled'
        
        # Enable modification in entry fields
        self.DAQSettingsFrame.daq_selection_menu.state(['!disabled'])
        self.DAQSettingsFrame.input_channel1_entry.state(['!disabled'])
        self.DAQSettingsFrame.input_channel2_entry.state(['!disabled'])
        self.DAQSettingsFrame.sample_rate_entry.state(['!disabled'])
        self.plotSettingsFrame.plot_duration_entry.state(['!disabled'])
        self.plotSettingsFrame.refresh_rate_entry.state(['!disabled'])
        self.plotSettingsFrame.spec_min_freq_entry.state(['!disabled'])
        self.plotSettingsFrame.spec_max_freq_entry.state(['!disabled'])
        self.plotSettingsFrame.spec_plot_check.state(['!disabled'])

        
    def save_log_file(self):
        log_filename = f"log_{self.filepath.split('/')[-1].split('.')[0]}.txt"
        log_filepath = "/".join(self.filepath.split('/')[:-1]) + "/" + log_filename
        
        log_data = {
            "Number of Input Channels": 2,  # Based on the channels defined in the script
            "Sample Rate": self.sample_rate,
            "Recording ID": self.ExperimentSettingsFrame.id_entry.get(),
            "Recording Duration": self.rec_dur,
	    "Start Time": self.t_start
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
        self.input_channel2_entry.insert(0, "ai4")
        self.input_channel2_entry.grid(row=2, column=1, sticky='w', padx=self.x_padding)
        
        self.output_channel_label = ttk.Label(self, text="Output Channel")
        self.output_channel_label.grid(row=3, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.output_channel1_entry = ttk.Entry(self, width=10)
        self.output_channel1_entry.insert(0, "port1/line0")
        self.output_channel1_entry.grid(row=3, column=1, columnspan=2, sticky='w', padx=self.x_padding)
       
        self.sample_rate_label = ttk.Label(self, text="Sample Rate (Hz)")
        self.sample_rate_label.grid(row=4, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.sample_rate_entry = ttk.Entry(self, width=10)
        self.sample_rate_entry.insert(0, "20000")
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
        

# Settings Frame
class ExperimentSettings(tk.LabelFrame):
    """Frame to set experiment parameters"""
    
    def __init__(self, parent, title):
        super().__init__(parent, text=title, labelanchor='n')
        self.parent = parent
        self.x_padding = (10, 10)
        self.create_widgets()
        
    def create_widgets(self):
        # Stimulus duration input
        self.id_label = tk.Label(self, text="Recording ID")
        self.id_label.grid(row=0, column=0, padx=self.x_padding, pady=5, sticky="w")
        self.id_entry = ttk.Entry(self, width=10)
        # self.stimulus_entry.insert(0, "1")
        self.id_entry.grid(row=0, column=1, padx=self.x_padding, pady=5, sticky="w")
        
        # Recording duration
        self.rec_dur_label = ttk.Label(self, text="Record Duration (s)")
        self.rec_dur_label.grid(row=1, column=0, padx=self.x_padding, pady=5, sticky="w")
        self.rec_dur_entry = ttk.Entry(self, width=10)
        self.rec_dur_entry.insert(0, "60")
        self.rec_dur_entry.grid(row=1, column=1, padx=self.x_padding, pady=5, sticky="w")
        
        # Connect button
        self.connect_button = ttk.Button(self, text="Start", command=self.parent.start_acquisition)
        self.connect_button.grid(row=2, column=0, sticky='ew', padx=self.x_padding, pady=(10, 0))
        
        # Disconnect button
        self.disconnect_button = ttk.Button(self, text="Stop", command=self.parent.stop_acquisition)
        self.disconnect_button.grid(row=2, column=1, sticky='ew', padx=self.x_padding, pady=(10, 0))
                
        # record button
        self.record_button = ttk.Button(self, text="Start Recording", command=self.parent.start_record)
        self.record_button.grid(row=3, column=0, columnspan=2, sticky='ew', padx=self.x_padding, pady=(10, 0))
        self.record_button['state'] = 'disabled'
        
        # Reset button
        self.reset_button = ttk.Button(self, text="Reset DAQ", command=self.parent.reset_device)
        self.reset_button.grid(row=4, column=0, sticky='ew', padx=self.x_padding, pady=(10, 0))
        
        # Close button
        self.close_button = ttk.Button(self, text="Close", command=root.destroy)
        self.close_button.grid(row=4, column=1, sticky='ew', padx=self.x_padding, pady=(10, 0))


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