import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import Tk, filedialog, Button, Checkbutton, IntVar, DoubleVar, Entry, Label, Frame, messagebox
import os
from scipy.signal import detrend, hilbert, butter, filtfilt
from scipy.ndimage import label as ndimage_label

class AnalysisGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Data Analysis Tool")

        # GUI variables
        self.subtract_stimulus_flag = IntVar(value=1)
        self.combine_channels_flag = IntVar(value=1)
        self.n_recording_channels = 1
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
        self.log_data = None
        self.data = None
        self.fm_threshold = DoubleVar(value=10.0)
        self.fm_baseline_window = DoubleVar(value=1.0)
        self.fm_min_duration = DoubleVar(value=5.0)
        self.fm_events = None
        self.instant_freq = np.array([])
        self.instant_time = np.array([])

        # Main layout: controls left, plot right
        main_frame = Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        self.control_frame = Frame(main_frame, width=230, bd=1, relief="sunken")
        self.control_frame.pack(side="left", fill="y", padx=5, pady=5)
        self.control_frame.pack_propagate(False)

        self.plot_frame = Frame(main_frame)
        self.plot_frame.pack(side="left", fill="both", expand=True)

        # Controls in vertical grid
        r = 0
        Button(self.control_frame, text="Load File", command=self.load_file).grid(
            row=r, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        r += 1
        Checkbutton(self.control_frame, text="Subtract Stimulus", variable=self.subtract_stimulus_flag).grid(
            row=r, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        r += 1
        self.combine_checkbox = Checkbutton(self.control_frame, text="Combine Channels",
                                            variable=self.combine_channels_flag, state="disabled")
        self.combine_checkbox.grid(row=r, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        r += 1
        Checkbutton(self.control_frame, text="Bandpass Filter", variable=self.bandpass_filter_flag).grid(
            row=r, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        r += 1
        Label(self.control_frame, text="Threshold:").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        Entry(self.control_frame, textvariable=self.threshold, width=8).grid(row=r, column=1, sticky="ew", padx=5, pady=2)
        r += 1
        Label(self.control_frame, text="Bandpass width (Hz):").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        Entry(self.control_frame, textvariable=self.bandpass_width, width=8).grid(row=r, column=1, sticky="ew", padx=5, pady=2)
        r += 1
        Label(self.control_frame, text="NFFT exponent:").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        Entry(self.control_frame, textvariable=self.nfft, width=8).grid(row=r, column=1, sticky="ew", padx=5, pady=2)
        r += 1
        Label(self.control_frame, text="noverlap exponent:").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        Entry(self.control_frame, textvariable=self.noverlap, width=8).grid(row=r, column=1, sticky="ew", padx=5, pady=2)
        r += 1
        Label(self.control_frame, text="Frequency Band (Hz):").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        Entry(self.control_frame, textvariable=self.freq_band, width=8).grid(row=r, column=1, sticky="ew", padx=5, pady=2)
        r += 1
        Button(self.control_frame, text="Refresh Plot", command=self.refresh_plot).grid(
            row=r, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        r += 1
        Label(self.control_frame, text="─── FM Detection ───").grid(
            row=r, column=0, columnspan=2, sticky="ew", padx=5, pady=(8, 2))
        r += 1
        Label(self.control_frame, text="FM Threshold (Hz):").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        Entry(self.control_frame, textvariable=self.fm_threshold, width=8).grid(row=r, column=1, sticky="ew", padx=5, pady=2)
        r += 1
        Label(self.control_frame, text="Baseline Window (s):").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        Entry(self.control_frame, textvariable=self.fm_baseline_window, width=8).grid(row=r, column=1, sticky="ew", padx=5, pady=2)
        r += 1
        Label(self.control_frame, text="Min Duration (ms):").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        Entry(self.control_frame, textvariable=self.fm_min_duration, width=8).grid(row=r, column=1, sticky="ew", padx=5, pady=2)
        r += 1
        Button(self.control_frame, text="Detect FMs", command=self.detect_fms).grid(
            row=r, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        r += 1
        self.export_fm_button = Button(self.control_frame, text="Export FMs", command=self.export_fms, state="disabled")
        self.export_fm_button.grid(row=r, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.control_frame.columnconfigure(1, weight=1)

        # Initialize plot
        self.fig, self.axes = plt.subplots(4, 1, figsize=(15, 8))
        self.axes[1].sharex(self.axes[0])
        self.axes[2].sharex(self.axes[0])
        self.canvas = None
        self.toolbar = None

    def load_file(self):
        """Load the log file and data."""
        filepath = filedialog.askopenfilename(title="Select Log File", filetypes=[("Text files", "*.txt")])
        if filepath:
            self.log_data, self.data = self.load_data(filepath)
            has_copy = self.log_data.get("Playback_Copy_Channel", "N/A") != "N/A"
            n_total = int(self.log_data["N_Input_Channels"])
            self.n_recording_channels = n_total - (1 if has_copy else 0)
            if self.n_recording_channels > 1:
                self.combine_checkbox.config(state="normal")
            else:
                self.combine_channels_flag.set(0)
                self.combine_checkbox.config(state="disabled")
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

        self.fm_events = None
        self.instant_freq = np.array([])
        self.instant_time = np.array([])
        if hasattr(self, 'export_fm_button'):
            self.export_fm_button.config(state="disabled")

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
        playback_mode = self.log_data.get("Playback_Mode", "Static")
        ramp_start_offset = float(self.log_data.get("Ramp_Start_Offset", 0))
        ramp_end_offset   = float(self.log_data.get("Ramp_End_Offset", 0))
        clamp_offset      = float(self.log_data.get("Clamp_Offset", 0))
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
        pre_stim_samples = pre_stim_duration * sample_rate

        # Determine channel layout from log
        n_total = int(self.log_data["N_Input_Channels"])
        has_copy = self.log_data.get("Playback_Copy_Channel", "N/A") != "N/A"
        n_recording = n_total - (1 if has_copy else 0)
        print(f"Channels: {n_total} total, {n_recording} recording, copy={'ch' + str(n_total-1) if has_copy else 'none'}, stimulus=ch{n_total-1}")

        # Stimulus is always the last channel
        stimulus = detrend(self.data[f"ch {n_total - 1}"])

        # Recording channel(s): combine if checkbox active and multiple channels present
        if self.combine_channels_flag.get() and n_recording > 1:
            channel_1 = detrend(self.data["ch 0"])
            for ch_idx in range(1, n_recording):
                next_ch = detrend(self.data[f"ch {ch_idx}"])
                corr = np.corrcoef(channel_1[:pre_stim_samples], next_ch[:pre_stim_samples])[0, 1]
                channel_1 = channel_1 + next_ch if corr > 0 else channel_1 - next_ch
        else:
            channel_1 = detrend(self.data["ch 0"])

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
        if playback_mode == "Ramp":
            mode_str = f"Ramp: {ramp_start_offset:+.1f} → {ramp_end_offset:+.1f} Hz"
        elif playback_mode == "Freq. Clamp":
            mode_str = f"Freq. Clamp: {clamp_offset:+.1f} Hz"
        else:
            mode_str = f"Offset: {frequency_offset} Hz"
        self.axes[0].set_title(f"Raw Data — Fish: {fish_id}, {mode_str}, Temp.: {temperature} °C, Cond.: {conductivity} µS")
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
            if playback_mode == "Ramp":
                sweep_lo = fish_freq + min(ramp_start_offset, ramp_end_offset)
                sweep_hi = fish_freq + max(ramp_start_offset, ramp_end_offset)
                low_freq_stim  = max(min_freq, sweep_lo - 50)
                high_freq_stim = min(max_freq, sweep_hi + 50)
            elif playback_mode == "Freq. Clamp":
                low_freq_stim  = fish_freq + clamp_offset - self.bandpass_width.get() / 2
                high_freq_stim = fish_freq + clamp_offset + self.bandpass_width.get() / 2
            else:
                low_freq_stim  = round(float(self.log_data['Dominant_Frequency'])) + round(float(self.log_data['Frequency_Offset'])) - self.bandpass_width.get()/2
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


        self.instant_freq = instant_freq
        self.instant_time = instant_time

        # Pre-stimulus baseline EOD frequency (more precise than log FFT value)
        pre_stim_mask = instant_time < pre_stim_duration
        baseline_fish_freq = float(np.median(instant_freq[pre_stim_mask])) if np.any(pre_stim_mask) else fish_freq

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
            self.axes[2].plot(instant_time, instant_freq, '.', color='steelblue', markersize=2, label='Fish EOD')
        if len(instant_time_stim) > 0:
            self.axes[2].plot(instant_time_stim, instant_freq_stim, '.', color='darkorange', markersize=2, label='Stim. copy')

        # Show warning if no data could be plotted
        if len(instant_time) == 0 and len(instant_time_stim) == 0:
            self.axes[2].text(0.5, 0.5, 'No zero crossings detected\nTry lowering the threshold or adjusting filter settings',
                            ha='center', va='center', transform=self.axes[2].transAxes,
                            fontsize=12, color='red')

        # Mode-specific overlays and stats initialisation
        ramp_crossing_time = np.nan
        ramp_fish_at_crossing = np.nan
        ramp_jar_at_crossing = np.nan
        ramp_jar_rate = np.nan
        clamp_max_jar = clamp_mean_jar = clamp_steady_jar = clamp_latency = np.nan

        if playback_mode == "Ramp":
            t_ramp = np.linspace(pre_stim_duration, pre_stim_duration + stim_duration, stim_duration * 10)
            f_ramp = (fish_freq + ramp_start_offset) + (ramp_end_offset - ramp_start_offset) * np.linspace(0, 1, len(t_ramp))
            self.axes[2].plot(t_ramp, f_ramp, 'r--', linewidth=1.5, label="Expected ramp", zorder=3)
            if ramp_start_offset * ramp_end_offset < 0:
                crossing_frac = abs(ramp_start_offset) / abs(ramp_end_offset - ramp_start_offset)
                ramp_crossing_time = pre_stim_duration + crossing_frac * stim_duration
                self.axes[2].axvline(ramp_crossing_time, color='r', linestyle=':', alpha=0.6, label="DF=0")
                if len(instant_time) > 0:
                    ramp_fish_at_crossing = float(np.interp(ramp_crossing_time, instant_time, instant_freq))
                    ramp_jar_at_crossing = ramp_fish_at_crossing - baseline_fish_freq
                    window_s = 10
                    rate_mask = (instant_time >= ramp_crossing_time - window_s) & \
                                (instant_time <= ramp_crossing_time + window_s)
                    if np.sum(rate_mask) > 1:
                        coeffs = np.polyfit(instant_time[rate_mask], instant_freq[rate_mask], 1)
                        ramp_jar_rate = float(coeffs[0])
            self.axes[2].legend(loc="upper right", fontsize=8)

        elif playback_mode == "Freq. Clamp":
            clamp_target = fish_freq + clamp_offset
            self.axes[2].axhline(clamp_target, color='g', linestyle='--', linewidth=1.5,
                                 label=f"Clamp target ({clamp_target:.1f} Hz)", zorder=3)
            if len(instant_time) > 0:
                stim_if_mask = (instant_time >= pre_stim_duration) & \
                               (instant_time < pre_stim_duration + stim_duration)
                stim_freqs = instant_freq[stim_if_mask]
                stim_times = instant_time[stim_if_mask]
                if len(stim_freqs) > 0:
                    clamp_max_jar  = float(np.max(stim_freqs)  - baseline_fish_freq)
                    clamp_mean_jar = float(np.mean(stim_freqs) - baseline_fish_freq)
                    ss_mask = stim_times >= pre_stim_duration + stim_duration - 10
                    ss_freqs = stim_freqs[ss_mask]
                    clamp_steady_jar = float(np.mean(ss_freqs) - baseline_fish_freq) if len(ss_freqs) > 0 else np.nan
                    above = stim_freqs > (baseline_fish_freq + 1.0)
                    clamp_latency = float(stim_times[above][0] - pre_stim_duration) if np.any(above) else np.nan
            self.axes[2].legend(loc="upper right", fontsize=8)

        else:
            if len(instant_time) > 0 or len(instant_time_stim) > 0:
                self.axes[2].legend(loc="upper right", fontsize=8)

        self.axes[2].set_title("Instantaneous Frequency — click 'Detect FMs' to label events")
        self.axes[2].set_ylabel("Frequency (Hz)")
        self.axes[2].set_xlabel("Time (s)")
        self.axes[2].set_ylim(min_freq, max_freq)

        # Plot statistics
        frequency_band = self.freq_band.get()
        stim_samples = stim_duration * sample_rate
        # dom_freq_fish = self.compute_dominant_frequency(cumulated_cleaned_data[0:pre_stim_samples], sample_rate, min_freq, max_freq)

        periods = {
            "Pre-Stimulus": (0, pre_stim_samples),
            "10s-Stimulus": (pre_stim_samples+sample_rate, pre_stim_samples + 11*sample_rate), # skip the first second
            "Complete Stimulus": (pre_stim_samples+sample_rate, pre_stim_samples + stim_samples-2*sample_rate), # skip first and last second
            "Post-Stimulus": (pre_stim_samples + stim_samples, total_samples)
        }

        stats = {"Dominant Freq (spectrogram)": {}, "Median Inst. Freq": {}}
        for period, (start, end) in periods.items():
            dom_freq = self.compute_dominant_frequency(cumulated_cleaned_data[start:end], sample_rate, max(1, fish_freq-frequency_band), fish_freq+frequency_band)

            # Calculate median instantaneous frequency if data exists
            if len(instant_time) > 0:
                period_mask = (instant_time >= start / sample_rate) & (instant_time < end / sample_rate)
                period_freqs = instant_freq[period_mask]
                freq_median = np.median(period_freqs) if len(period_freqs) > 0 else np.nan
            else:
                freq_median = np.nan

            stats["Dominant Freq (spectrogram)"][period] = round(dom_freq, 2)
            stats["Median Inst. Freq"][period] = round(float(freq_median), 2) if not np.isnan(freq_median) else "N/A"

        def _fmt(v):
            return "N/A" if (isinstance(v, float) and np.isnan(v)) else round(float(v), 2)

        if playback_mode == "Ramp" and not np.isnan(ramp_crossing_time):
            blank = {p: "-" for p in periods}
            stats["Crossing Time (s)"]    = {**blank, "Complete Stimulus": _fmt(ramp_crossing_time - pre_stim_duration)}
            stats["JAR at Crossing (Hz)"] = {**blank, "Complete Stimulus": _fmt(ramp_jar_at_crossing)}
            stats["JAR Rate (Hz/s)"]      = {**blank, "Complete Stimulus": _fmt(ramp_jar_rate)}
        elif playback_mode == "Freq. Clamp":
            blank = {p: "-" for p in periods}
            stats["Max JAR (Hz)"]          = {**blank, "Complete Stimulus": _fmt(clamp_max_jar)}
            stats["Mean JAR (Hz)"]         = {**blank, "Complete Stimulus": _fmt(clamp_mean_jar)}
            stats["Steady-state JAR (Hz)"] = {**blank, "Complete Stimulus": _fmt(clamp_steady_jar)}
            stats["JAR Latency (s)"]       = {**blank, "Complete Stimulus": _fmt(clamp_latency)}

        stats_df = pd.DataFrame(stats).T
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

    def detect_fms(self):
        if len(self.instant_freq) == 0 or len(self.instant_time) == 0:
            messagebox.showwarning("No Data", "No instantaneous frequency data available.\nLoad a file and refresh the plot first.")
            return

        fm_thr = self.fm_threshold.get()
        baseline_win_s = self.fm_baseline_window.get()
        min_dur_ms = self.fm_min_duration.get()
        interp_dt = 0.001  # 1ms interpolation grid

        reg_time = np.arange(self.instant_time[0], self.instant_time[-1], interp_dt)
        reg_freq = np.interp(reg_time, self.instant_time, self.instant_freq)

        win = max(1, int(baseline_win_s / interp_dt))
        baseline = pd.Series(reg_freq).rolling(win, center=True, min_periods=1).median().values
        deviation = reg_freq - baseline

        above = (deviation > fm_thr).astype(int)
        labeled_array, n_raw = ndimage_label(above)

        TYPE_COLORS = {
            "Type 1": "red",
            "Type 2": "royalblue",
            "Type 3": "darkorange",
            "Rise": "green",
            "Yodel": "purple",
            "Yodel (trunc.)": "mediumpurple",
            "Unclassified": "gray",
        }

        events = []
        for i in range(1, n_raw + 1):
            idxs = np.where(labeled_array == i)[0]
            duration_ms = len(idxs) * interp_dt * 1000
            if duration_ms < min_dur_ms:
                continue

            t_start = reg_time[idxs[0]]
            t_end = reg_time[idxs[-1]]
            peak_local = int(np.argmax(deviation[idxs]))
            peak_idx = idxs[peak_local]
            peak_hz = float(deviation[peak_idx])
            peak_time = float(reg_time[peak_idx])
            rise_time_ms = peak_local * interp_dt * 1000
            fall_time_ms = (len(idxs) - 1 - peak_local) * interp_dt * 1000
            rise_fraction = rise_time_ms / duration_ms if duration_ms > 0 else 0.5

            post_start = idxs[-1] + 1
            post_end = min(len(reg_time), post_start + int(0.1 / interp_dt))
            undershoot_hz = float(deviation[post_start:post_end].min()) if post_end > post_start else 0.0
            undershoot_hz = min(0.0, undershoot_hz)

            # Event ends within 50ms of recording end → likely truncated
            at_end = idxs[-1] >= len(reg_time) - int(0.05 / interp_dt)

            # Classification based on frequency-time properties only.
            # Amplitude reduction is not used — unreliable from summed/multi-electrode signal.
            if fall_time_ms > 5000:
                fm_type = "Yodel"
            elif at_end and peak_hz >= 50:
                fm_type = "Yodel (trunc.)"
            elif peak_hz >= 100 and duration_ms > 150:
                fm_type = "Type 3"
            elif peak_hz >= 100 and duration_ms <= 150:
                fm_type = "Type 1"
            elif peak_hz <= 50 and duration_ms <= 50 and abs(rise_fraction - 0.5) < 0.25:
                fm_type = "Type 2"
            elif peak_hz <= 20 and duration_ms <= 50:
                fm_type = "Rise"
            else:
                fm_type = "Unclassified"

            events.append({
                "type": fm_type,
                "t_start": round(float(t_start), 4),
                "t_end": round(float(t_end), 4),
                "duration_ms": round(duration_ms, 2),
                "peak_hz": round(peak_hz, 2),
                "peak_time": round(peak_time, 4),
                "rise_time_ms": round(rise_time_ms, 2),
                "fall_time_ms": round(fall_time_ms, 2),
                "rise_fraction": round(rise_fraction, 3),
                "undershoot_hz": round(undershoot_hz, 3),
                "color": TYPE_COLORS.get(fm_type, "gray"),
            })

        self.fm_events = events
        self.export_fm_button.config(state="normal" if events else "disabled")
        self.draw_fm_overlay()

    def draw_fm_overlay(self):
        if not self.fm_events:
            self.axes[2].set_title("Instantaneous Frequency — 0 FM events detected")
            if self.canvas is not None:
                self.canvas.draw()
            return

        seen_types = set()
        for ev in self.fm_events:
            fm_type = ev["type"]
            color = ev["color"]
            self.axes[2].axvspan(ev["t_start"], ev["t_end"], alpha=0.2, color=color, zorder=1)
            scatter_label = fm_type if fm_type not in seen_types else "_nolegend_"
            peak_freq = float(np.interp(ev["peak_time"], self.instant_time, self.instant_freq))
            self.axes[2].scatter(ev["peak_time"], peak_freq, c=color, s=50, zorder=5, label=scatter_label)
            seen_types.add(fm_type)

        self.axes[2].legend(loc="upper right", fontsize=8, markerscale=1.2)
        n = len(self.fm_events)
        self.axes[2].set_title(f"Instantaneous Frequency — {n} FM event{'s' if n != 1 else ''} detected")
        self.fig.tight_layout()
        if self.canvas is not None:
            self.canvas.draw()

    def export_fms(self):
        if not self.fm_events:
            return
        filepath = filedialog.asksaveasfilename(
            title="Save FM Events",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if filepath:
            export_cols = ["type", "t_start", "t_end", "duration_ms", "peak_hz",
                           "peak_time", "rise_time_ms", "fall_time_ms", "rise_fraction", "undershoot_hz"]
            pd.DataFrame(self.fm_events)[export_cols].to_csv(filepath, index=False)

if __name__ == "__main__":
    root = Tk()
    app = AnalysisGUI(root)
    root.mainloop()
