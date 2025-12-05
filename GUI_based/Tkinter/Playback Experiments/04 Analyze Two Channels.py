

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.gridspec as gridspec
from tkinter import Tk, filedialog, Button, IntVar, DoubleVar, Entry, Label, Frame, Checkbutton
import os
from scipy.signal import detrend, butter, filtfilt

class AnalysisGUI:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Data Analysis Tool")
        
        # GUI variables
        self.threshold = DoubleVar(value=0.00)
        self.min_y = DoubleVar(value=400)
        self.max_y = DoubleVar(value=1000)
        # self.bandpass_filter_flag = IntVar(value=1)
        # self.bandpass_width = IntVar(value=300)
        # self.lp_cutoff = IntVar(value=400)
        # self.hp_cutoff = IntVar(value=1200)
        self.nfft = IntVar(value=13)  # Default NFFT exponent
        self.noverlap = IntVar(value=10)  # Default noverlap exponent
        
        self.log_data = None
        self.data = None
        self.mean_pulse_form = None  # To store computed mean pulse form

        # Create plot area
        self.plot_frame = Frame(self.root)
        self.plot_frame.pack(side="top", fill="both", expand=True)

        # Create control area
        self.control_frame = Frame(self.root)
        self.control_frame.pack(side="bottom", fill="x")

        # Add controls
        Button(self.control_frame, text="Load File", command=self.load_file).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="Spec min freq:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.min_y, width=10).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="Spec max freq:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.max_y, width=10).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="NFFT exponent:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.nfft, width=10).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="noverlap exponent:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.noverlap, width=10).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="Zerocross threshold:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.threshold, width=10).pack(side="left", padx=5, pady=5)
        # Checkbutton(self.control_frame, text="Bandpass Filter", variable=self.bandpass_filter_flag).pack(side="left", padx=5, pady=5)
        # Label(self.control_frame, text="Bandpass width (Hz):").pack(side="left", padx=5, pady=5)
        # Entry(self.control_frame, textvariable=self.bandpass_width, width=10).pack(side="left", padx=5, pady=5)
        Button(self.control_frame, text="Refresh Plot", command=self.refresh_plot).pack(side="left", padx=5, pady=5)

        # Initialize plot (5 subplots)
        self.fig = plt.figure(constrained_layout=True, figsize=(8, 6))
        self.gs = gridspec.GridSpec(nrows=3, ncols=3, figure=self.fig)
        # self.fig, self.axes = plt.subplots(3, 1, figsize=(15, 8))
        self.canvas = None
        self.toolbar = None

    def load_file(self):
        """Load the log file and data."""
        filepath = filedialog.askopenfilename(title="Select Log File", filetypes=[("Text files", "*.txt")])
        if filepath:
            self.log_data, self.data = self.load_data(filepath)
            self.refresh_plot()

    def load_data(self, log_filepath):
        """Load log file and associated data file."""
        with open(log_filepath, 'r') as file:
            log_data = {line.split(": ")[0]: line.split(": ")[1].strip() for line in file.readlines()}
    
        base_filepath = log_filepath.split('log_')[0] + log_filepath.split('log_')[-1].split('.')[0]
        feather_filepath = base_filepath + '.feather'
        parquet_filepath = base_filepath + '.parquet'
        
        if os.path.exists(feather_filepath):
            data = pd.read_feather(feather_filepath)
        elif os.path.exists(parquet_filepath):
            data = pd.read_parquet(parquet_filepath)
        else:
            raise FileNotFoundError(
                f"Data file not found. Expected either {feather_filepath} or {parquet_filepath}"
            )
        
        return log_data, data
    
    def bandpass_filter(self, data, fs, lowcut, highcut, order=4):
        """
        Returns a bandpass-filtered version of data, with passband = [lowcut, highcut].
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    def inst_freq(self, y, fs, zerocross=0):
        """
        Computes instantaneous frequency of input signal `y` based on sampling rate `fs`
        using the threshold value `zerocross`.
        """
        y1 = y[:-1]
        y2 = y[1:]
        zerocross_idx = np.where((y1 <= zerocross) & (y2 > zerocross))[0]
        amp_step = y[zerocross_idx + 1] - y[zerocross_idx]
        amp_frac = (zerocross - y[zerocross_idx]) / amp_step
        y_frac = zerocross_idx + amp_frac
        inst_f = 1.0 / (np.diff(y_frac) / fs)
        tinst_f = np.cumsum(np.diff(y_frac) / fs) + y_frac[0] / fs
        return inst_f, tinst_f

    def compute_dominant_frequency(self, data, sample_rate, min_freq, max_freq):
        """Compute the dominant frequency of a quasi-sinusoidal signal within specified bounds."""
        fft_result = np.fft.rfft(data)
        frequencies = np.fft.rfftfreq(len(data), d=1/sample_rate)
        valid_indices = np.where((frequencies >= min_freq) & (frequencies <= max_freq))
        filtered_frequencies = frequencies[valid_indices]
        filtered_fft_result = np.abs(fft_result[valid_indices])
        dominant_freq = filtered_frequencies[np.argmax(filtered_fft_result)]
        return dominant_freq
    
    def refresh_plot(self):
        """Refresh the plots based on current settings and analysis type."""
        if self.log_data is None or self.data is None:
            print("No data loaded. Please load a file first.")
            return
        
        if hasattr(self, 'toolbar') and self.toolbar is not None:
            self.toolbar.mode = ''
            self.toolbar.update()
            
        # for ax in self.axes:
        #     ax.clear()
        
        self.fig.clear()

        sample_rate = int(self.log_data["Sample Rate"])
        rec_id = self.log_data["Recording ID"]  
        min_freq = self.min_y.get()
        max_freq = self.max_y.get()
        total_samples = len(self.data)
        time_axis = np.arange(total_samples) / sample_rate
        
        # Detrend channels
        channel_1 = detrend(self.data["ch1"])
        channel_2 = detrend(self.data["ch2"])
        max_y_val = max(np.max(channel_1), np.max(channel_2))
        
        # Wave-type analysis (existing behavior)
        self.ax1 = self.fig.add_subplot(self.gs[0, :])
        self.ax1.plot(time_axis, channel_1, label="Ch 1")
        self.ax1.plot(time_axis, channel_2 + 2*max_y_val, label="Ch 2")
        self.ax1.set_title(f"Raw Data - Recording ID: {rec_id}")
        self.ax1.set_ylabel("Amplitude")
        self.ax1.legend(loc="lower left")
        
        
        # self.axes[0].plot(time_axis, channel_1, label="Ch 1")
        # self.axes[0].plot(time_axis, channel_2 + 2*max_y_val, label="Ch 2")
        # self.axes[0].set_title(f"Raw Data - Recording ID: {rec_id}")
        # self.axes[0].set_ylabel("Amplitude")
        # self.axes[0].legend(loc="lower left")
        
        # # Instantaneous frequency plots
        # inst_freq1, inst_time1 = self.inst_freq(channel_1, sample_rate, self.threshold.get())
        # self.axes[1].plot(inst_time1, inst_freq1, '.')
        # self.axes[1].set_title("Channel 1 Instantaneous Frequency")
        # self.axes[1].set_ylabel("Frequency (Hz)")
        # self.axes[1].set_xlabel("Time (s)")
        # self.axes[1].set_ylim(min_freq, max_freq)
        
        # inst_freq2, inst_time2 = self.inst_freq(channel_2, sample_rate, self.threshold.get())
        # self.axes[2].plot(inst_time2, inst_freq2, '.')
        # self.axes[2].set_title("Channel 2 Instantaneous Frequency")
        # self.axes[2].set_ylabel("Frequency (Hz)")
        # self.axes[2].set_xlabel("Time (s)")
        # self.axes[2].set_ylim(min_freq, max_freq)
        
        # Spectrograms
        nfft_value = 2**self.nfft.get()
        noverlap_value = 2**self.noverlap.get()
        hanning_window = np.hanning(nfft_value)
        # self.axes[1].specgram(channel_1, Fs=sample_rate, NFFT=nfft_value, noverlap=noverlap_value, window=hanning_window)
        # self.axes[1].set_title("Channel 1 Spectrogram")
        # self.axes[1].set_ylabel("Frequency (Hz)")
        # self.axes[1].set_ylim(min_freq, max_freq)
        
        # self.axes[2].specgram(channel_2, Fs=sample_rate, NFFT=nfft_value, noverlap=noverlap_value, window=hanning_window)
        # self.axes[2].set_title("Channel 2 Spectrogram")
        # self.axes[2].set_ylabel("Frequency (Hz)")
        # self.axes[2].set_ylim(min_freq, max_freq)
        
        # Cumulated data
        correlation = np.corrcoef(channel_1, channel_2)[0, 1]
        cumulated_data = channel_1 + channel_2 if correlation > 0 else channel_1 - channel_2
        
        self.ax2 = self.fig.add_subplot(self.gs[1, :])
        self.ax2.specgram(cumulated_data, Fs=sample_rate, NFFT=nfft_value, noverlap=noverlap_value, window=hanning_window)
        self.ax2.set_title("Cumulated Data Spectrogram")
        self.ax2.set_ylabel("Frequency (Hz)")
        self.ax2.set_ylim(min_freq, max_freq)     
        
        
        # # Plot instantaneous frequency
        # threshold = self.threshold.get()
        
        # min_y = float(self.min_y.get())
        # max_y = float(self.max_y.get())
        # dom_freq = self.compute_dominant_frequency(cumulated_data[:60*sample_rate], sample_rate, min_y, max_y)
        

        # # 1) Optional bandpass filter
        # if self.bandpass_filter_flag.get():
        #     low_freq = dom_freq - self.bandpass_width.get()/2
        #     high_freq = dom_freq + self.bandpass_width.get()/2
        #     cumulated_cleaned_data_filtered = self.bandpass_filter(cumulated_data, sample_rate, low_freq, high_freq)
        #     instant_freq, instant_time = self.inst_freq(cumulated_cleaned_data_filtered, sample_rate, threshold)
        # else:
        #     # zero_crossings = np.where(np.diff(np.sign(cumulated_data)))[0]
        #     instant_freq, instant_time = self.inst_freq(cumulated_data, sample_rate, threshold)
        # # zero_crossings = self.calculate_zero_crossings(cumulated_data, threshold)
        # # instant_freq = sample_rate / np.diff(zero_crossings)
        # # instant_time = zero_crossings[:-1] / sample_rate
        # self.axes[2].plot(instant_time, instant_freq,'.')
        # self.axes[2].set_title("Instantaneous Frequency")
        # self.axes[2].set_ylabel("Frequency (Hz)")
        # self.axes[2].set_xlabel("Time (s)")
        # self.axes[2].set_ylim(min_freq, max_freq)
        
        self.ax3 = self.fig.add_subplot(self.gs[2, 0])
        self.ax4 = self.fig.add_subplot(self.gs[2, 1])
        self.ax5 = self.fig.add_subplot(self.gs[2, 2])
        self.ax3.psd(cumulated_data[0:300*sample_rate], nfft_value, sample_rate)
        self.ax3.set_title("Power Spectral Density first 5 min.")
        self.ax3.set_xlim(min_freq, max_freq)
        self.ax4.psd(cumulated_data[(8*60*sample_rate):(9*60*sample_rate)], nfft_value, sample_rate)
        self.ax4.set_title("Power Spectral Density 8th min.")
        self.ax4.set_xlim(min_freq, max_freq)
        self.ax5.psd(cumulated_data[len(cumulated_data)-(300*sample_rate):], nfft_value, sample_rate)
        self.ax5.set_title("Power Spectral Density last 5 min.")
        self.ax5.set_xlim(min_freq, max_freq)
        
        self.fig.tight_layout()
        
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        if hasattr(self, 'toolbar') and self.toolbar is not None:
            self.toolbar.destroy()
    
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
    
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        

if __name__ == "__main__":
    root = Tk()
    app = AnalysisGUI(root)
    root.mainloop()
