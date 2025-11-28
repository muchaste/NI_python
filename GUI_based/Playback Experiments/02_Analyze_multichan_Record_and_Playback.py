import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import Tk, filedialog, Button, Checkbutton, IntVar, DoubleVar, Entry, Label, Frame, messagebox
import os
from scipy.signal import detrend, hilbert, butter, filtfilt

class AnalysisGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Data Analysis Tool")

        # GUI variables
        self.subtract_stimulus_flag = IntVar(value=1)
        self.threshold = DoubleVar(value=0.001)
        # self.stim_threshold = DoubleVar(value=0.001)
        self.nfft = IntVar(value=14)  # Default NFFT
        self.noverlap = IntVar(value=9)  # Default noverlap
        # self.env_lp = IntVar(value=5)   # Default lowpass cutoff frequency
        self.bandpass_filter_flag = IntVar(value=1)
        self.bandpass_width = IntVar(value=300)
        self.lp_cutoff = IntVar(value=400)
        self.hp_cutoff = IntVar(value=1200)
        self.freq_band = IntVar(value=50)
        self.lag = DoubleVar(value=2.0)
        self.log_data = None
        self.data = None

        # Define fonts
        self.label_font = ("TkDefaultFont", 11)
        self.button_font = ("TkDefaultFont", 11)
        self.entry_font = ("TkDefaultFont", 10)

        # Create control area (left side)
        self.control_frame = Frame(self.root)
        self.control_frame.pack(side="left", fill="y", padx=10, pady=10)

        # Add controls vertically
        Button(self.control_frame, text="Load File", command=self.load_file, font=self.button_font).pack(anchor="w", padx=5, pady=5)
        
        Checkbutton(self.control_frame, text="Subtract Stimulus", variable=self.subtract_stimulus_flag, font=self.label_font).pack(anchor="w", padx=5, pady=5)
        
        Label(self.control_frame, text="Threshold:", font=self.label_font).pack(anchor="w", padx=5, pady=(10, 0))
        Entry(self.control_frame, textvariable=self.threshold, width=15, font=self.entry_font).pack(anchor="w", padx=5, pady=(0, 5))
        
        Checkbutton(self.control_frame, text="Bandpass Filter", variable=self.bandpass_filter_flag, font=self.label_font).pack(anchor="w", padx=5, pady=5)
        
        Label(self.control_frame, text="Bandpass width (Hz):", font=self.label_font).pack(anchor="w", padx=5, pady=(10, 0))
        Entry(self.control_frame, textvariable=self.bandpass_width, width=15, font=self.entry_font).pack(anchor="w", padx=5, pady=(0, 5))
        
        Label(self.control_frame, text="NFFT exponent:", font=self.label_font).pack(anchor="w", padx=5, pady=(10, 0))
        Entry(self.control_frame, textvariable=self.nfft, width=15, font=self.entry_font).pack(anchor="w", padx=5, pady=(0, 5))
        
        Label(self.control_frame, text="noverlap exponent:", font=self.label_font).pack(anchor="w", padx=5, pady=(10, 0))
        Entry(self.control_frame, textvariable=self.noverlap, width=15, font=self.entry_font).pack(anchor="w", padx=5, pady=(0, 5))
        
        Label(self.control_frame, text="Frequency Band (Hz):", font=self.label_font).pack(anchor="w", padx=5, pady=(10, 0))
        Entry(self.control_frame, textvariable=self.freq_band, width=15, font=self.entry_font).pack(anchor="w", padx=5, pady=(0, 5))
        
        Label(self.control_frame, text="Lag (s):", font=self.label_font).pack(anchor="w", padx=5, pady=(10, 0))
        Entry(self.control_frame, textvariable=self.lag, width=15, font=self.entry_font).pack(anchor="w", padx=5, pady=(0, 5))
        
        Button(self.control_frame, text="Refresh Plot", command=self.refresh_plot, font=self.button_font).pack(anchor="w", padx=5, pady=15)

        # Create plot area (right side)
        self.plot_frame = Frame(self.root)
        self.plot_frame.pack(side="right", fill="both", expand=True)

        # Initialize plot
        self.fig, self.axes = plt.subplots(4, 1, figsize=(15, 8))
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

        # Construct data filepath from log filepath
        data_filepath = log_filepath.replace('log_', '').replace('.txt', '.bin')

        if not os.path.exists(data_filepath):
            raise FileNotFoundError(f"Data file not found at: {data_filepath}")

        # Load binary data
        n_channels = int(log_data['N_Input_Channels'])
        sample_rate = int(log_data['Sample_Rate'])
        
        # Binary file format: interleaved columns [time_ms, ch0, ch1, ...]
        # Read as float64 (default for numpy tofile)
        raw_data = np.fromfile(data_filepath, dtype=np.float64)
        
        # Reshape into columns: time, ch0, ch1, ...
        n_cols = n_channels + 1  # time + channels
        n_rows = len(raw_data) // n_cols
        data_array = raw_data[:n_rows * n_cols].reshape((n_rows, n_cols))
        
        # Create DataFrame with proper column names
        column_names = ["Time [ms]"] + [f"ch {i}" for i in range(n_channels)]
        data = pd.DataFrame(data_array, columns=column_names)
        
        return log_data, data

    def bandpass_filter(self, data, fs, lowcut, highcut, order=4):
        """
        Returns a bandpass-filtered version of data, with passband = [lowcut, highcut].
        If lowcut is too low (< 10 Hz), applies lowpass filter instead.
        Returns (filtered_data, was_bandpass_applied)
        """
        nyquist = 0.5 * fs
        
        # Check if lowcut is too low for bandpass filtering
        if lowcut < 10:
            # Use lowpass filter instead
            high = highcut / nyquist
            b, a = butter(order, high, btype='low')
            filtered_data = filtfilt(b, a, data)
            return filtered_data, False
        else:
            # Use bandpass filter
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band')
            filtered_data = filtfilt(b, a, data)
            return filtered_data, True

    def inst_freq(self, y, fs,
                     zero_threshold=0.0,
                     min_interval=0.0):
        """
        A robust instantaneous frequency function that:
          1) Optionally filters the data around expected fundamental freq (bandpass).
          2) Detects zero crossings above 'zero_threshold' amplitude step.
          3) Skips suspicious intervals below 'min_interval'.
        """

        # 2) Standard zero crossing detection
        y1 = y[:-1]
        y2 = y[1:]
        zerocross_idx = np.where((y1 <= 0) & (y2 > 0))[0]

        # Filter out suspicious amplitude steps
        amp_step = y[zerocross_idx + 1] - y[zerocross_idx]
        keep = np.abs(amp_step) > zero_threshold  # e.g., zero_threshold=0.001
        zerocross_idx = zerocross_idx[keep]
        amp_step = amp_step[keep]

        # Check if we have any zero crossings left
        if len(zerocross_idx) == 0:
            return np.array([]), np.array([])

        amp_frac = (0 - y[zerocross_idx]) / amp_step
        y_frac = zerocross_idx + amp_frac

        # 3) Convert crossing intervals to frequency, optionally skip too-small intervals
        crossing_intervals = np.diff(y_frac) / fs  # in seconds
        if min_interval > 0:
            # e.g., skip intervals that are less than half of your expected fundamental period
            crossing_intervals = crossing_intervals[crossing_intervals >= min_interval]

        # Check if we have any intervals left
        if len(crossing_intervals) == 0:
            return np.array([]), np.array([])

        inst_f = 1.0 / crossing_intervals
        tinst_f = np.cumsum(crossing_intervals) + (y_frac[0] / fs)
        return inst_f, tinst_f


    # def inst_freq(self, y, fs, zerocross=0):
    #     """
    #     Computes instantaneous frequency of input signal `y` based on sampling rate `fs`
    #     using the threshold value `zerocross` (usually 0 V).

    #     Parameters:
    #     - y: np.ndarray
    #         The input signal.
    #     - fs: float
    #         The sampling rate in Hz.
    #     - zerocross: float, optional
    #         The threshold value for zero-crossing (default is 0).

    #     Returns:
    #     - inst_f: np.ndarray
    #         Instantaneous frequency.
    #     - tinst_f: np.ndarray
    #         Time points for plotting instantaneous frequency.
    #     """
    #     y1 = y[:-1]
    #     y2 = y[1:]

    #     # Find zero-crossing indices
    #     zerocross_idx = np.where((y1 <= zerocross) & (y2 > zerocross))[0]

    #     # Compute fractional zero-crossing positions
    #     amp_step = y[zerocross_idx + 1] - y[zerocross_idx]  # Amplitude step
    #     amp_frac = (zerocross - y[zerocross_idx]) / amp_step  # Fraction of step below zero
    #     y_frac = zerocross_idx + amp_frac  # Adjust zero-crossing indices with fraction

    #     # Compute instantaneous frequency
    #     inst_f = 1.0 / (np.diff(y_frac) / fs)  # Instantaneous frequency
    #     tinst_f = np.cumsum(np.diff(y_frac) / fs) + y_frac[0] / fs  # Time points for plotting

    #     return inst_f, tinst_f

    def subtract_stim(self, y, stim, stim_on, stim_off, Fs, lagrange, ampfactorrange):
        """
        Finds the best lag and amplitude factor to minimize contamination from a stimulus.

        Parameters:
        - y: np.ndarray
            The signal to be cleaned.
        - stim: np.ndarray
            The stimulus signal.
        - stim_on: int
            Time of stimulus onset (s)
        - stim_off: int
            Time of stimulus offset (s)
        - Fs: int
            Sampling frequency.
        - lagrange: range or list
            The range of lags to test.
        - ampfactorrange: range or list
            The range of amplitude factors to test.

        Returns:
        - y: np.ndarray
            The cleaned signal.
        - bestlag: int
            The best lag found.
        - bestampfactor: float
            The best amplitude factor found.
        """


        # Calculate amplitudes
        EODamp = np.sqrt(np.mean(y[:int(stim_on * Fs)] ** 2))
        beatamp = np.sqrt(np.mean(y[int(stim_on * Fs):int(stim_off * Fs)] ** 2))
        stimamp = beatamp - EODamp

        # Find the best lag
        table = []
        for i in lagrange:
            stimnormshift = np.concatenate((np.zeros(i), stim[:-i] / np.max(stim) * stimamp * 4.5))
            yclean = y - stimnormshift
            cleanamp = np.sqrt(np.mean(yclean[int(stim_on * Fs):int(stim_off * Fs)] ** 2))
            table.append(cleanamp - EODamp)

        mincontamination = min(table)
        bestlag = lagrange[table.index(mincontamination)]

        # Find the best amplitude factor
        table = []
        for j in ampfactorrange:
            stimnormshift = np.concatenate((np.zeros(bestlag), stim[:-bestlag] / np.max(stim) * stimamp * 0.1 * j))
            yclean = y - stimnormshift
            cleanamp = np.sqrt(np.mean(yclean[int(stim_on * Fs):int(stim_off * Fs)] ** 2))
            table.append(cleanamp - EODamp)

        mincontamination = min(table)
        bestampfactor = ampfactorrange[table.index(mincontamination)] * 0.1

        # Apply the best lag and amplitude factor
        stimnormshift = np.concatenate((np.zeros(bestlag), stim[:-bestlag] / np.max(stim) * stimamp * bestampfactor))
        y = y - stimnormshift

        return y

    def compute_dominant_frequency(self, data, sample_rate, min_freq, max_freq):
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

    def calculate_envelope(self, data, sample_rate, lowpass_cutoff=5):
        # Hilbert Transform to compute the envelope
        analytic_signal = hilbert(data)
        envelope = np.abs(analytic_signal)

        # Low-pass filter the envelope to retain only low-frequency fluctuations
        b, a = butter(4, lowpass_cutoff / (sample_rate / 2), btype='low')
        filtered_envelope = filtfilt(b, a, envelope)

        return filtered_envelope

    def refresh_plot(self):
        """Refresh the plots based on current settings."""
        if self.log_data is None or self.data is None:
            print("No data loaded. Please load a file first.")
            return

        # Reset zoom or pan mode before refreshing the plot
        if hasattr(self, 'toolbar') and self.toolbar is not None:
            self.toolbar.mode = ''  # Reset active toolbar mode (zoom/pan)
            self.toolbar.update()  # Update the toolbar to reflect the change

        # Clear previous plots
        for ax in self.axes:
            ax.clear()

        # Extract key parameters
        sample_rate = int(self.log_data["Sample_Rate"])
        pre_stim_duration = int(self.log_data["Pre_Stimulus_Duration"])
        stim_duration = int(self.log_data["Stimulus_Duration"])
        # post_stim_duration = int(self.log_data["Post_Stimulus_Duration"])
        fish_id = self.log_data["Fish_ID"]
        fish_freq = float(self.log_data["Dominant_Frequency"])
        frequency_offset = self.log_data["Frequency_Offset"]
        try:
            temperature = float(self.log_data["Temperature"])
        except KeyError:
            temperature = "NA"
        try:
            conductivity = int(self.log_data["Conductivity"])
        except KeyError:
            conductivity = "NA"


        total_samples = len(self.data)
        time_axis = np.arange(total_samples) / sample_rate

        # Detrend raw data
        channel_1 = detrend(self.data["ch 0"])
        stimulus = detrend(self.data["ch 1"])
        # channel_3 = detrend(self.data["ch 2"])  # Stimulus

        # Cumulated data
        # correlation = np.corrcoef(channel_1[:(pre_stim_duration*sample_rate)], channel_2[:(pre_stim_duration*sample_rate)])[0, 1]
        # cumulated_data = channel_1 + channel_2 if correlation > 0 else channel_1 - channel_2

        # Subtract stimulus if checkbox is checked
        amp_facts = np.arange(1,61)
        lag_range = np.arange(1,51)
        if self.subtract_stimulus_flag.get():
            cumulated_cleaned_data = self.subtract_stim(channel_1, stimulus, pre_stim_duration, pre_stim_duration+stim_duration, sample_rate, lag_range, amp_facts)
            # cleaned_2 = self.subtract_stim(channel_2, channel_3, pre_stim_duration, pre_stim_duration+stim_duration, sample_rate, lag_range, amp_facts)
        else:
            cumulated_cleaned_data = channel_1

        # Plot raw data
        max_y = max(np.max(channel_1), np.max(cumulated_cleaned_data))
        # env_lp_cutoff = self.env_lp.get()
        # envelope_cum = self.calculate_envelope(cumulated_cleaned_data, sample_rate, env_lp_cutoff)
        self.axes[0].plot(time_axis, channel_1 + 2 * max_y, label="Raw Data")
        self.axes[0].plot(time_axis, cumulated_cleaned_data, label="Cleaned Data")
        # self.axes[0].plot(time_axis, envelope_cum, label="Envelope")
        self.axes[0].set_title(f"Raw Data - Fish: {fish_id}, Freq. Offset: {frequency_offset} Hz, Temp.: {temperature} °C, Cond.: {conductivity} uS")
        self.axes[0].set_ylabel("Amplitude")
        self.axes[0].legend(loc="lower left")

        # Plot spectrogram
        min_freq = float(self.log_data["Min_Frequency"])
        max_freq = float(self.log_data["Max_Frequency"])
        nfft_exp = self.nfft.get()
        nfft_value = 2**nfft_exp
        noverlap_exp = self.noverlap.get()
        noverlap_value = 2**noverlap_exp
        # Create a Hanning window
        hanning_window = np.hanning(nfft_value)

        self.axes[1].specgram(
            cumulated_cleaned_data, Fs=sample_rate, NFFT=nfft_value, noverlap=noverlap_value, window=hanning_window
        )
        self.axes[1].set_title("Cleaned Data Spectrogram")
        self.axes[1].set_ylabel("Frequency (Hz)")
        self.axes[1].set_ylim(min_freq, max_freq)

        # Plot instantaneous frequency
        threshold = self.threshold.get()

        # 1) Optional bandpass filter
        if self.bandpass_filter_flag.get():
            low_freq = round(float(self.log_data['Dominant_Frequency'])) - self.bandpass_width.get()/2
            high_freq = round(float(self.log_data['Dominant_Frequency'])) + self.bandpass_width.get()/2
            low_freq_stim = round(float(self.log_data['Dominant_Frequency'])) + round(float(self.log_data['Frequency_Offset'])) - self.bandpass_width.get()/2
            high_freq_stim = round(float(self.log_data['Dominant_Frequency'])) + round(float(self.log_data['Frequency_Offset'])) + self.bandpass_width.get()/2
            
            # Filter fish signal
            cumulated_cleaned_data_filtered, fish_bandpass_applied = self.bandpass_filter(cumulated_cleaned_data, sample_rate, low_freq, high_freq)
            cumulated_cleaned_data_filtered = detrend(cumulated_cleaned_data_filtered)
            
            # Filter stimulus signal
            stim_data_filtered, stim_bandpass_applied = self.bandpass_filter(stimulus, sample_rate, low_freq_stim, high_freq_stim)
            stim_data_filtered = detrend(stim_data_filtered)
            
            # Show popup if bandpass couldn't be applied
            if not fish_bandpass_applied or not stim_bandpass_applied:
                filter_info = []
                if not fish_bandpass_applied:
                    filter_info.append(f"Fish signal (low freq: {low_freq:.1f} Hz < 10 Hz)")
                if not stim_bandpass_applied:
                    filter_info.append(f"Stimulus signal (low freq: {low_freq_stim:.1f} Hz < 10 Hz)")
                messagebox.showwarning(
                    "Filtering Notice",
                    f"Bandpass filtering was replaced with lowpass filtering for:\n" + "\n".join(filter_info) +
                    f"\n\nLowpass cutoff frequencies: Fish={high_freq} Hz, Stimulus={high_freq_stim} Hz"
                )
            
            instant_freq, instant_time = self.inst_freq(cumulated_cleaned_data_filtered, sample_rate, threshold)
            instant_freq_stim, instant_time_stim = self.inst_freq(stim_data_filtered, sample_rate, threshold)
        else:
            # zero_crossings = np.where(np.diff(np.sign(cumulated_data)))[0]
            instant_freq, instant_time = self.inst_freq(cumulated_cleaned_data, sample_rate, threshold)
            instant_freq_stim, instant_time_stim = self.inst_freq(stimulus, sample_rate, threshold)


        # Filter stimulus instantaneous frequency to stimulus period only
        if len(instant_time_stim) > 0:
            stim_mask = np.where((instant_time_stim > pre_stim_duration) & (instant_time_stim < pre_stim_duration+stim_duration))[0]
            instant_freq_stim = instant_freq_stim[stim_mask]
            instant_time_stim = instant_time_stim[stim_mask]

        # zero_crossings = self.calculate_zero_crossings(cumulated_data, threshold)
        # instant_freq = sample_rate / np.diff(zero_crossings)
        # instant_time = zero_crossings[:-1] / sample_rate
        
        # Plot instantaneous frequency (only if data exists)
        if len(instant_time) > 0:
            self.axes[2].plot(instant_time, instant_freq,'.')
        if len(instant_time_stim) > 0:
            self.axes[2].plot(instant_time_stim, instant_freq_stim,'.')
        
        # Show warning if no data could be plotted
        if len(instant_time) == 0 and len(instant_time_stim) == 0:
            self.axes[2].text(0.5, 0.5, 'No zero crossings detected\nTry lowering the threshold or adjusting filter settings',
                            ha='center', va='center', transform=self.axes[2].transAxes,
                            fontsize=12, color='red')
        self.axes[2].set_title("Instantaneous Frequency")
        self.axes[2].set_ylabel("Frequency (Hz)")
        self.axes[2].set_xlabel("Time (s)")
        self.axes[2].set_ylim(min_freq, max_freq)

        # Plot statistics
        frequency_band = self.freq_band.get()
        lag = self.lag.get()
        pre_stim_samples = pre_stim_duration * sample_rate
        stim_samples = stim_duration * sample_rate
        window_size = sample_rate  # 1 second windows
        # dom_freq_fish = self.compute_dominant_frequency(cumulated_cleaned_data[0:pre_stim_samples], sample_rate, min_freq, max_freq)

        periods = {
            "Pre-Stimulus": (0, pre_stim_samples),
            "10s-Stimulus": (pre_stim_samples+int(lag*sample_rate), pre_stim_samples + int((lag+10)*sample_rate)), # skip the lag seconds
            "Complete Stimulus": (pre_stim_samples+int(lag*sample_rate), pre_stim_samples + stim_samples - int(lag*sample_rate)), # skip first and last second
            "Post-Stimulus": (pre_stim_samples + stim_samples, total_samples)
        }

        stats = {}
        for period, (start, end) in periods.items():
            # Calculate median dominant frequency from spectrogram over time windows of 1s
            period_length = end - start
            
            # Handle short periods (< 1s) by using the entire period as one window
            if period_length < int(sample_rate*window_size):
                # If period is too short for meaningful FFT (< 100ms or < 2 cycles), use NaN
                min_samples = max(int(0.1 * sample_rate), int(2 * sample_rate / fish_freq))
                if period_length < min_samples:
                    dom_freq = np.nan
                else:
                    dom_freq = self.compute_dominant_frequency(
                        cumulated_cleaned_data[start:end], 
                        sample_rate, 
                        max(1, fish_freq-frequency_band), 
                        fish_freq+frequency_band
                    )
            else:
                # For longer periods, use 1s sliding windows
                time_windows = np.arange(start, end, int(sample_rate*window_size))  # 1s windows
                dom_freqs = []
                
                for time_window in time_windows:
                    window_start = int(time_window)
                    window_end = int(min(time_window + int(sample_rate*window_size), end))
                    window_length = window_end - window_start
                    
                    # Only compute if window has sufficient samples
                    min_samples = max(int(0.1 * sample_rate), int(2 * sample_rate / fish_freq))
                    if window_length >= min_samples:
                        dom_freq_window = self.compute_dominant_frequency(
                            cumulated_cleaned_data[window_start:window_end], 
                            sample_rate, 
                            max(1, fish_freq-frequency_band), 
                            fish_freq+frequency_band
                        )
                        dom_freqs.append(dom_freq_window)
                
                # Compute median only if we have valid windows
                dom_freq = np.median(dom_freqs) if len(dom_freqs) > 0 else np.nan
            
            # Calculate median instantaneous frequency if data exists
            if len(instant_time) > 0:
                period_mask = (instant_time >= start / sample_rate) & (instant_time < end / sample_rate)
                period_freqs = instant_freq[period_mask]
                freq_median = np.median(period_freqs) if len(period_freqs) > 0 else np.nan
            else:
                freq_median = np.nan
            
            # temp = temperature
            # cond = conductivity
            # amp_cov_cum = np.abs(np.std(cumulated_cleaned_data[start:end]) / np.mean(cumulated_cleaned_data[start:end]))
            # envelope_cum_period = detrend(envelope_cum[start:end])+1
            # envelope_cov = np.std(envelope_cum_period) #/np.mean(envelope_cum))
            stats[period] = (dom_freq, freq_median)#, amp_cov_ch1, amp_cov_ch2, envelope_cov

        stats_df = pd.DataFrame(stats, index=["Median Dom. Freq. (spectro)", "Median Inst. Freq."]) #"Amp. CoV (envelope)","Amp. CoV (Ch 1)", "Amp. CoV (Ch 2)",
        stats_df = stats_df.round(2)
        self.axes[3].axis("off")
        table = self.axes[3].table(
            cellText=stats_df.values,
            rowLabels=stats_df.index,
            colLabels=stats_df.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Adjust layout
        self.fig.tight_layout()


        # Update canvas and toolbar
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        if hasattr(self, 'toolbar') and self.toolbar is not None:
            self.toolbar.destroy()

        # Create a new canvas using FigureCanvasTkAgg
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # Add the navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")

if __name__ == "__main__":
    root = Tk()
    app = AnalysisGUI(root)
    root.mainloop()
