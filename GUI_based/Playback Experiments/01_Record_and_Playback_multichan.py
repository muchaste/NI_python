#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dynamic analog input recording program with spectrogram option for NI DAQ devices

Authors: Stefan Mucha/chatGPT
"""

import sys
import os
import time
import threading
import collections
import queue
import numpy as np
import nidaqmx
from nidaqmx import constants
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from scipy import signal
from scipy.signal import resample
from datetime import datetime
import gc

# Try to import sounddevice for audio output
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: sounddevice not available. Audio output will be disabled.")

# Get list of DAQ device names
daqSys = nidaqmx.system.System()
daqList = daqSys.devices.device_names
if not daqList:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    QtWidgets.QMessageBox.critical(None, "DAQ Error", "No DAQ detected, check connection.")
    sys.exit(1)

# ---------------- Signal Generation Functions ----------------
def compute_dominant_frequency(data, sample_rate, min_freq, max_freq):
    """Compute the dominant frequency of a quasi-sinusoidal signal within specified bounds."""
    fft_result = np.fft.rfft(data)
    frequencies = np.fft.rfftfreq(len(data), d=1/sample_rate)
    valid_indices = np.where((frequencies >= min_freq) & (frequencies <= max_freq))
    filtered_frequencies = frequencies[valid_indices]
    filtered_fft_result = np.abs(fft_result[valid_indices])
    if len(filtered_fft_result) == 0:
        return 0.0
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
    pre_stim_samples = int(pre_stim_duration * sample_rate)
    stimulus_samples = int(stimulus_duration * sample_rate)
    post_stim_samples = int(post_stim_duration * sample_rate)
    
    pre_stim_signal = [0] * pre_stim_samples
    
    time_array = np.arange(stimulus_samples) / sample_rate
    frequency_with_offset = dominant_freq + offset
    amp_factor_array = np.repeat(float(amp_factor), stimulus_samples)
    gradient = np.arange(sample_rate)/sample_rate
    amp_factor_array[0:sample_rate] = amp_factor_array[0:sample_rate]*gradient
    amp_factor_array[len(amp_factor_array)-sample_rate:] = amp_factor_array[len(amp_factor_array)-sample_rate:]*gradient[::-1]
    stimulus_signal = amp_factor_array * np.sin(2 * np.pi * frequency_with_offset * time_array)
    
    post_stim_signal = [0] * post_stim_samples
    
    synthetic_signal = np.concatenate((pre_stim_signal, stimulus_signal, post_stim_signal))
    
    return synthetic_signal

def generate_sine_wave(freq, sample_rate, amp_factor):
    """Generate a sine wave for test signal output."""
    frequency = float(freq)
    amp = float(amp_factor)
    time_array = np.arange(sample_rate) / sample_rate
    sine_wave = amp * np.sin(2 * np.pi * frequency * time_array)
    return sine_wave

# ---------------- Audio Output Module ----------------
class AudioStreamer:
    """Manages real-time audio output of incoming DAQ data."""
    def __init__(self, daq_sample_rate, audio_sample_rate=24000, buffer_size=4096):
        self.daq_sample_rate = daq_sample_rate
        self.audio_sample_rate = audio_sample_rate
        self.buffer_size = buffer_size
        self.audio_queue = queue.Queue(maxsize=10)  # Smaller queue for lower latency
        self.stream = None
        self.running = False
        self.needs_resampling = (daq_sample_rate != audio_sample_rate)
        
        # Calculate resampling ratio
        if self.needs_resampling:
            self.resample_ratio = audio_sample_rate / daq_sample_rate
        else:
            self.resample_ratio = 1.0
        
        # Internal buffer to accumulate partial data
        self.internal_buffer = np.array([], dtype=np.float32)
    
    def audio_callback(self, outdata, frames, time_info, status):
        """Callback for sounddevice output stream."""
        if status:
            print(f"Audio status: {status}")
        
        # Try to fill internal buffer from queue
        while len(self.internal_buffer) < frames:
            try:
                # Get data from queue with short timeout
                new_data = self.audio_queue.get(timeout=0.001)
                self.internal_buffer = np.concatenate([self.internal_buffer, new_data])
            except queue.Empty:
                # If we don't have enough data, break and use what we have
                break
        
        # Extract requested frames from internal buffer
        if len(self.internal_buffer) >= frames:
            outdata[:] = self.internal_buffer[:frames].reshape(-1, 1)
            self.internal_buffer = self.internal_buffer[frames:]
        else:
            # Not enough data - output what we have and pad with zeros
            available = len(self.internal_buffer)
            if available > 0:
                outdata[:available] = self.internal_buffer.reshape(-1, 1)
                outdata[available:] = 0
                self.internal_buffer = np.array([], dtype=np.float32)
            else:
                outdata.fill(0)
    
    def start(self):
        """Start audio streaming."""
        if not AUDIO_AVAILABLE:
            print("Cannot start audio: sounddevice not available")
            return False
        
        if self.stream is not None:
            return False
        
        try:
            self.running = True
            self.stream = sd.OutputStream(
                samplerate=self.audio_sample_rate,
                channels=1,  # Mono
                callback=self.audio_callback,
                blocksize=self.buffer_size,
                dtype=np.float32
            )
            self.stream.start()
            return True
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self.running = False
            return False
    
    def stop(self):
        """Stop audio streaming."""
        self.running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Error stopping audio stream: {e}")
            finally:
                self.stream = None
        
        # Clear the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # Clear internal buffer
        self.internal_buffer = np.array([], dtype=np.float32)
    
    def add_data(self, data):
        """Add new data to audio buffer (called from DAQ callback)."""
        if not self.running:
            return
        
        # Normalize data to [-1, 1] range for audio output
        # Assuming input voltage range is -10V to +10V
        normalized_data = np.clip(data / 10.0, -1.0, 1.0).astype(np.float32)
        
        # Resample if necessary
        if self.needs_resampling:
            num_output_samples = int(len(normalized_data) * self.resample_ratio)
            resampled_data = resample(normalized_data, num_output_samples)
        else:
            resampled_data = normalized_data
        
        # Add to queue (non-blocking, drop newest if full to prevent buildup)
        try:
            self.audio_queue.put_nowait(resampled_data)
        except queue.Full:
            # Queue is full - just drop this chunk to prevent accumulating delay
            pass

# ---------------- Data Acquisition Module ----------------
class DataAcquisition:
    def __init__(self, input_channels, sample_rate, min_voltage, max_voltage, refresh_rate, plot_duration, output_channel=None, daq_device=None):
        # input_channels: list of strings
        self.input_channels = input_channels
        self.output_channel = output_channel
        self.daq_device = daq_device
        self.sample_rate = sample_rate
        self.min_voltage = min_voltage
        self.max_voltage = max_voltage
        self.refresh_rate = refresh_rate

        self.num_channels = len(input_channels)
        sample_interval = int(self.sample_rate / self.refresh_rate)
        buffer_size = 100000
        if buffer_size % sample_interval != 0:
            for divisor in range(sample_interval, 0, -1):
                if buffer_size % divisor == 0:
                    sample_interval = divisor
                    break
        self.sample_interval = sample_interval

        # One buffer per channel
        self.plot_buffer = [collections.deque(maxlen=int(plot_duration * self.sample_rate)) for _ in range(self.num_channels)]
        self.storage_buffer = [collections.deque() for _ in range(self.num_channels)]

        self.ai_task = nidaqmx.Task()
        for ch in input_channels:
            self.ai_task.ai_channels.add_ai_voltage_chan(
                ch, min_val=self.min_voltage, max_val=self.max_voltage,
                units=constants.VoltageUnits.VOLTS)
        self.ai_task.timing.cfg_samp_clk_timing(self.sample_rate, sample_mode=constants.AcquisitionType.CONTINUOUS)
        self.ai_task.register_every_n_samples_acquired_into_buffer_event(self.sample_interval, self.callback)
        self.running = False

        self.recording_active = False
        self.acquired_samples = 0  # Cumulative sample counter (int limit ~2.1B samples)
        self.samples_to_save = 0
        self.recording_complete_callback = None
        self.recording_start_timestamp = None
        self.logfile_written = False
        self.logfile_callback = None
        
        # Playback-related attributes
        self.playback_active = False
        self.ao_task = None
        self.test_signal_task = None
        
        # Audio output
        self.audio_streamer = AudioStreamer(sample_rate, sample_rate) if AUDIO_AVAILABLE else None
        self.audio_enabled = False

    def start(self):
        self.running = True
        self.recording_start_timestamp = None
        self.logfile_written = False
        self.ai_task.start()

    def stop(self):
        self.running = False
        if self.recording_active:
            self.recording_active = False
            if self.recording_complete_callback is not None:
                self.recording_complete_callback()
        if self.playback_active:
            self.stop_playback()
        if self.test_signal_task is not None:
            try:
                self.test_signal_task.stop()
                self.test_signal_task.close()
                self.test_signal_task = None
            except Exception as e:
                print("Error stopping test signal task:", e)
        # Stop audio output
        if self.audio_enabled and self.audio_streamer is not None:
            self.audio_streamer.stop()
            self.audio_enabled = False
        self.recording_start_timestamp = None
        try:
            self.ai_task.stop()
            self.ai_task.close()
        except Exception as e:
            print("Error stopping DAQ task:", e)

    def callback(self, task_handle, event_type, number_of_samples, callback_data):
        if not self.running:
            return 0
        temp_data = self.ai_task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)

        # temp_data shape: (num_channels, n_samples)
        if self.num_channels == 1:
            temp_data = np.array([temp_data])  # shape (1, n_samples)
        else:
            temp_data = np.array(temp_data)
        # print(f"num_channels={self.num_channels}, len(plot_buffer)={len(self.plot_buffer)}, len(storage_buffer)={len(self.storage_buffer)}, temp_data.shape={temp_data.shape}")

        # Extend buffers per channel
        for ch_idx in range(self.num_channels):
            self.plot_buffer[ch_idx].extend(temp_data[ch_idx])
        
        # Send first channel data to audio output if enabled
        if self.audio_enabled and self.audio_streamer is not None:
            self.audio_streamer.add_data(temp_data[0])
        
        if self.recording_active:
            n_samples = temp_data.shape[1]
            if self.recording_start_timestamp is None:
                current_time = time.time()
                self.recording_start_timestamp = current_time - (n_samples - 1) / self.sample_rate
            if not self.logfile_written and self.recording_start_timestamp is not None:
                if self.logfile_callback:
                    self.logfile_callback()
                self.logfile_written = True
            
            # Check if we need to limit samples to not exceed total recording duration
            if self.acquired_samples + n_samples > self.samples_to_save:
                # Limit samples to exactly reach the target
                n_samples = self.samples_to_save - self.acquired_samples
                temp_data = temp_data[:, :n_samples]
                recording_complete = True
            else:
                recording_complete = False
            
            # Store data to buffers
            for ch_idx in range(self.num_channels):
                self.storage_buffer[ch_idx].extend(temp_data[ch_idx])
            
            # Update acquired samples
            self.acquired_samples += n_samples
            
            # Check if playback recording is complete
            if self.playback_active and time.time() - self.playback_start_time > self.playback_total_duration:
                # Stop playback but don't cleanup immediately (will be done before next recording)
                if self.ao_task is not None:
                    try:
                        self.ao_task.stop()
                        # Write zeros to clear output
                        zero_signal = np.repeat(0, int(self.sample_rate * 0.1))
                        self.ao_task.write(zero_signal, auto_start=True)
                        self.ao_task.stop()
                    except:
                        pass
                self.playback_active = False
                
            if recording_complete:
                self.recording_active = False
                if self.recording_complete_callback is not None:
                    self.recording_complete_callback()
        return 0
    
    def start_playback(self, playback_signal, total_duration):
        """Start synchronized playback with recording."""
        if self.output_channel is None or self.daq_device is None:
            raise ValueError("Output channel and DAQ device must be specified for playback")
        
        # Force cleanup any existing output task to prevent resource conflicts
        self.force_cleanup_output()
        
        self.playback_active = True
        self.playback_total_duration = total_duration
        self.playback_start_time = time.time()
        
        # Create output task synchronized to input sample clock
        self.ao_task = nidaqmx.Task()
        self.ao_task.ao_channels.add_ao_voltage_chan(
            self.output_channel, 
            min_val=self.min_voltage, 
            max_val=self.max_voltage, 
            units=constants.VoltageUnits.VOLTS
        )
        self.ao_task.timing.cfg_samp_clk_timing(
            self.sample_rate, 
            source=f"/{self.daq_device}/ai/SampleClock",
            sample_mode=constants.AcquisitionType.CONTINUOUS, 
            samps_per_chan=len(playback_signal)
        )
        
        # Write signal and start (waits for ai sample clock)
        self.ao_task.write(playback_signal, auto_start=False)
        self.ao_task.start()
    
    def stop_playback(self):
        """Stop playback and clean up output task."""
        if self.ao_task is not None:
            try:
                self.ao_task.stop()
                # Write zeros to clear output
                zero_signal = np.repeat(0, int(self.sample_rate * 0.1))
                self.ao_task.write(zero_signal, auto_start=True)
                self.ao_task.stop()
                self.ao_task.close()
            except Exception as e:
                print("Error stopping playback:", e)
                # Force close even if stop/write failed
                try:
                    if self.ao_task is not None:
                        self.ao_task.close()
                except:
                    pass
            finally:
                self.ao_task = None
        self.playback_active = False
    
    def force_cleanup_output(self):
        """Force cleanup of output task to prevent resource conflicts."""
        if self.ao_task is not None:
            try:
                self.ao_task.close()
            except Exception as e:
                print(f"Force cleanup output task: {e}")
            finally:
                self.ao_task = None
        self.playback_active = False
    
    def start_test_signal(self, amplitude):
        """Start continuous test signal output."""
        if self.output_channel is None or self.daq_device is None:
            raise ValueError("Output channel and DAQ device must be specified for test signal")
        
        test_signal = generate_sine_wave(1000, self.sample_rate, amplitude)
        
        self.test_signal_task = nidaqmx.Task()
        self.test_signal_task.ao_channels.add_ao_voltage_chan(
            self.output_channel, 
            min_val=self.min_voltage, 
            max_val=self.max_voltage, 
            units=constants.VoltageUnits.VOLTS
        )
        self.test_signal_task.timing.cfg_samp_clk_timing(
            self.sample_rate, 
            source=f"/{self.daq_device}/ai/SampleClock",
            sample_mode=constants.AcquisitionType.CONTINUOUS
        )
        
        self.test_signal_task.write(test_signal, auto_start=True)
    
    def stop_test_signal(self):
        """Stop test signal output."""
        if self.test_signal_task is not None:
            try:
                self.test_signal_task.stop()
                zero_signal = np.repeat(0, int(self.sample_rate * 0.1))
                self.test_signal_task.write(zero_signal, auto_start=True)
                self.test_signal_task.stop()
                self.test_signal_task.close()
                self.test_signal_task = None
            except Exception as e:
                print("Error stopping test signal:", e)
    
    def enable_audio_output(self):
        """Enable audio output of first channel."""
        if self.audio_streamer is None:
            return False
        if not self.audio_enabled:
            success = self.audio_streamer.start()
            if success:
                self.audio_enabled = True
            return success
        return True
    
    def disable_audio_output(self):
        """Disable audio output."""
        if self.audio_streamer is not None and self.audio_enabled:
            self.audio_streamer.stop()
            self.audio_enabled = False

# ---------------- File Writing Module ----------------
class FileWriter(threading.Thread):
    def __init__(self, storage_buffer, buffer_lock, filepath, sample_rate, flush_interval=5):
        super().__init__(daemon=True)
        self.storage_buffer = storage_buffer  # List of deques, one per channel
        self.buffer_lock = buffer_lock
        self.filepath = filepath
        self.sample_rate = sample_rate
        self.flush_interval = flush_interval
        self.stop_event = threading.Event()

    def run(self):
        acquired_samples = 0
        dt = 1000 / self.sample_rate  # in ms
        with open(self.filepath, 'ab') as f:
            while not self.stop_event.is_set():
                time.sleep(self.flush_interval)
                with self.buffer_lock:
                    # Find minimum available samples across all channels
                    min_len = min(len(buf) for buf in self.storage_buffer)
                    if min_len == 0:
                        continue
                    # Pop min_len samples from each channel
                    data_chunk = np.array([ [self.storage_buffer[ch].popleft() for _ in range(min_len)] for ch in range(len(self.storage_buffer)) ])
                n_samples = min_len
                time_vec = (acquired_samples + np.arange(n_samples)) * dt
                acquired_samples += n_samples
                # Stack as columns: time, ch1, ch2, ...
                interleaved = np.column_stack((time_vec, data_chunk.T))  # shape (n_samples, num_channels+1)
                interleaved.tofile(f)
                f.flush()
                os.fsync(f.fileno())
        print("FileWriter stopped.")

    def stop(self):
        self.stop_event.set()

# ---------------- PyQt5 GUI Module ----------------
class DataAcquisitionGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NI DAQ Recorder")
        self.resize(1600, 900)
        self.acq = None
        self.file_writer = None
        self.buffer_lock = threading.Lock()
        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(self.update_plot)
        self.spectrogram_lut = pg.colormap.get('viridis').getLookupTable(0.0, 1.0, 256)
        self.img = None
        
        # Recording indicator
        self.recording_indicator_timer = QtCore.QTimer(self)
        self.recording_indicator_timer.timeout.connect(self.toggle_recording_indicator)
        self.recording_indicator_state = False
        
        self.init_ui()

    def init_ui(self):
        controls_layout = QtWidgets.QVBoxLayout()

        # DAQ Settings
        self.daqGroup = QtWidgets.QGroupBox("DAQ Settings")
        daq_layout = QtWidgets.QFormLayout()
        self.daqCombo = QtWidgets.QComboBox()
        self.daqCombo.addItems(daqList)
        daq_layout.addRow("Select DAQ:", self.daqCombo)
        self.chanEdit = QtWidgets.QLineEdit("ai5,ai2")
        daq_layout.addRow("Input Channel(s):", self.chanEdit)
        self.outputChanEdit = QtWidgets.QLineEdit("ao0")
        daq_layout.addRow("Output Channel:", self.outputChanEdit)
        self.sampleRateEdit = QtWidgets.QLineEdit("40000")
        daq_layout.addRow("Sample Rate (Hz):", self.sampleRateEdit)
        self.daqGroup.setLayout(daq_layout)
        controls_layout.addWidget(self.daqGroup)

        # Plot Settings
        self.plotGroup = QtWidgets.QGroupBox("Data Settings")
        plot_layout = QtWidgets.QFormLayout()
        self.plotDurEdit = QtWidgets.QLineEdit("1")
        plot_layout.addRow("Plot Duration (s):", self.plotDurEdit)
        self.refreshRateEdit = QtWidgets.QLineEdit("10")
        plot_layout.addRow("Refresh Rate (Hz):", self.refreshRateEdit)
        self.specMinEdit = QtWidgets.QLineEdit("100")
        plot_layout.addRow("Min Freq (Hz):", self.specMinEdit)
        self.specMaxEdit = QtWidgets.QLineEdit("1400")
        plot_layout.addRow("Max Freq (Hz):", self.specMaxEdit)
        self.specWindowEdit = QtWidgets.QLineEdit("11")
        plot_layout.addRow("Spec. Window Size Exponent:", self.specWindowEdit)
        self.specCheck = QtWidgets.QCheckBox("Plot Spectrogram (Ch1 only)")
        self.specCheck.setToolTip("Enable to plot the spectrogram for the first channel.")
        self.specCheck.setChecked(True)
        plot_layout.addRow(self.specCheck)
        self.domFreqCheck = QtWidgets.QCheckBox("Show Dominant Frequency")
        self.domFreqCheck.setChecked(True)
        plot_layout.addRow(self.domFreqCheck)
        self.domFreqLabel = QtWidgets.QLabel("Dominant Frequency: --- Hz")
        plot_layout.addRow(self.domFreqLabel)
        self.audioOutputCheck = QtWidgets.QCheckBox("Enable Audio Output (Ch1)")
        self.audioOutputCheck.setToolTip("Play first channel through system audio output (48kHz)")
        self.audioOutputCheck.setChecked(False)
        if not AUDIO_AVAILABLE:
            self.audioOutputCheck.setEnabled(False)
            self.audioOutputCheck.setToolTip("Audio output unavailable - install sounddevice: pip install sounddevice")
        plot_layout.addRow(self.audioOutputCheck)
        self.yOffsetEdit = QtWidgets.QDoubleSpinBox()
        self.yOffsetEdit.setRange(0.0, 1000.0)
        self.yOffsetEdit.setSingleStep(0.1)
        self.yOffsetEdit.setValue(2.0)
        plot_layout.addRow("Y Offset (V):", self.yOffsetEdit)
        self.plotGroup.setLayout(plot_layout)
        controls_layout.addWidget(self.plotGroup)

        # Recording Settings
        self.expGroup = QtWidgets.QGroupBox("Experiment Settings")
        exp_layout = QtWidgets.QFormLayout()
        
        # Recording indicator
        self.recordingIndicator = QtWidgets.QLabel("")
        self.recordingIndicator.setAlignment(QtCore.Qt.AlignCenter)
        self.recordingIndicator.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 5px; border-radius: 3px;"
        )
        exp_layout.addRow(self.recordingIndicator)
        
        self.fishIdEdit = QtWidgets.QLineEdit()
        exp_layout.addRow("Fish ID:", self.fishIdEdit)
        self.tempEdit = QtWidgets.QLineEdit()
        exp_layout.addRow("Temp. (C):", self.tempEdit)
        self.condEdit = QtWidgets.QLineEdit()
        exp_layout.addRow("Cond. (uS):", self.condEdit)
        
        # Playback Settings
        self.offsetCombo = QtWidgets.QComboBox()
        self.offsetCombo.addItems(["-200", "-100", "-20", "-10", "-5", "-2.5", "0", "2.5", "5", "10", "20", "100", "200"])
        self.offsetCombo.setCurrentText("0")
        exp_layout.addRow("Frequency Offset (Hz):", self.offsetCombo)
        self.preStimEdit = QtWidgets.QLineEdit("60")
        exp_layout.addRow("Pre-Stim Duration (s):", self.preStimEdit)
        self.stimEdit = QtWidgets.QLineEdit("60")
        exp_layout.addRow("Stimulus Duration (s):", self.stimEdit)
        self.postStimEdit = QtWidgets.QLineEdit("60")
        exp_layout.addRow("Post-Stim Duration (s):", self.postStimEdit)
        self.ampFactorEdit = QtWidgets.QLineEdit("0.1")
        exp_layout.addRow("Amplitude Factor:", self.ampFactorEdit)

        btn_layout = QtWidgets.QHBoxLayout()
        self.connectBtn = QtWidgets.QPushButton("Connect")
        self.connectBtn.setStyleSheet("color: green; font-weight: bold;")
        self.disconnectBtn = QtWidgets.QPushButton("Disconnect")
        self.disconnectBtn.setStyleSheet("color: #003366; font-weight: bold;")
        btn_layout.addWidget(self.connectBtn)
        btn_layout.addWidget(self.disconnectBtn)
        exp_layout.addRow(btn_layout)
        
        btn_layout2 = QtWidgets.QHBoxLayout()
        self.recordBtn = QtWidgets.QPushButton("● Start Recording")
        self.recordBtn.setStyleSheet("color: red; font-weight: bold;")
        btn_layout2.addWidget(self.recordBtn)
        exp_layout.addRow(btn_layout2)
        
        self.testSignalStartBtn = QtWidgets.QPushButton("Test Signal On")
        self.testSignalStartBtn.setStyleSheet("color: green; font-weight: bold;")
        self.testSignalStopBtn = QtWidgets.QPushButton("Test Signal Off")
        self.testSignalStopBtn.setStyleSheet("color: orange; font-weight: bold;")
        self.resetBtn = QtWidgets.QPushButton("Reset DAQ")
        self.resetBtn.setStyleSheet("color: black; font-weight: bold;")
        self.closeBtn = QtWidgets.QPushButton("Close")
        self.closeBtn.setStyleSheet("color: black; font-weight: bold;")
        
        btn_layout3 = QtWidgets.QHBoxLayout()
        btn_layout3.addWidget(self.testSignalStartBtn)
        btn_layout3.addWidget(self.testSignalStopBtn)
        exp_layout.addRow(btn_layout3)
        
        btn_layout4 = QtWidgets.QHBoxLayout()
        btn_layout4.addWidget(self.resetBtn)
        btn_layout4.addWidget(self.closeBtn)
        exp_layout.addRow(btn_layout4)
        self.expGroup.setLayout(exp_layout)
        controls_layout.addWidget(self.expGroup)

        # Plotting panel
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.rawPlotWidget = pg.PlotWidget(title="Raw Data")
        self.specPlotWidget = pg.PlotWidget(title="Spectrogram")
        self.specPlotWidget.setMouseEnabled(x=False, y=False)
        plot_vlayout = QtWidgets.QVBoxLayout()
        plot_vlayout.addWidget(self.rawPlotWidget)
        plot_vlayout.addWidget(self.specPlotWidget)

        # Main layout
        main_layout = QtWidgets.QHBoxLayout()
        controls_widget = QtWidgets.QWidget()
        controls_widget.setLayout(controls_layout)
        main_layout.addWidget(controls_widget, 1)
        plot_widget = QtWidgets.QWidget()
        plot_widget.setLayout(plot_vlayout)
        main_layout.addWidget(plot_widget, 3)
        self.setLayout(main_layout)

        # Connect signals
        self.connectBtn.clicked.connect(self.start_acquisition)
        self.disconnectBtn.clicked.connect(self.stop_acquisition)
        self.recordBtn.clicked.connect(self.start_record)
        self.testSignalStartBtn.clicked.connect(self.start_test_signal)
        self.testSignalStopBtn.clicked.connect(self.stop_test_signal)
        self.resetBtn.clicked.connect(self.reset_device)
        self.closeBtn.clicked.connect(self.safe_close)
        self.audioOutputCheck.stateChanged.connect(self.toggle_audio_output)
        
        # Initial button states
        self.recordBtn.setEnabled(False)
        self.testSignalStartBtn.setEnabled(False)
        self.testSignalStopBtn.setEnabled(False)

    def start_acquisition(self):
        device = self.daqCombo.currentText()
        # Accept comma or whitespace separated channels, strip whitespace
        channel_text = self.chanEdit.text().strip()
        channel_list = [f"{device}/{ch.strip()}" for ch in channel_text.replace(',', ' ').split()]
        try:
            sample_rate = int(self.sampleRateEdit.text())
            min_freq = float(self.specMinEdit.text())
            max_freq = float(self.specMaxEdit.text())
            refresh_rate = int(self.refreshRateEdit.text())
            plot_duration = float(self.plotDurEdit.text())
            spec_window_size = 2 ** int(self.specWindowEdit.text())
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Input Error", f"Invalid input: {e}")
            return
        if sample_rate <= 0 or refresh_rate <= 0 or plot_duration <= 0:
            QtWidgets.QMessageBox.warning(self, "Input Error", "Sample Rate, Refresh Rate, and Plot Duration must be positive.")
            return

        self.sample_rate = sample_rate
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.spec_window_size = spec_window_size
        self.daq_device = device
        self.output_channel = f"{device}/{self.outputChanEdit.text().strip()}"

        self.connectBtn.setEnabled(False)
        self.resetBtn.setEnabled(False)
        self.disconnectBtn.setEnabled(True)
        self.recordBtn.setEnabled(True)
        self.testSignalStartBtn.setEnabled(True)
        self.daqCombo.setEnabled(False)
        self.chanEdit.setEnabled(False)
        self.outputChanEdit.setEnabled(False)
        self.sampleRateEdit.setEnabled(False)
        self.plotDurEdit.setEnabled(False)
        self.refreshRateEdit.setEnabled(False)
        self.specMinEdit.setEnabled(False)
        self.specMaxEdit.setEnabled(False)
        self.specWindowEdit.setEnabled(False)

        self.acq = DataAcquisition(channel_list, sample_rate, -10, 10, refresh_rate, plot_duration, 
                                   output_channel=self.output_channel, daq_device=self.daq_device)
        for buf in self.acq.plot_buffer:
            buf.clear()
        for buf in self.acq.storage_buffer:
            buf.clear()
        self.acq.start()
        interval = int(1000 / refresh_rate)
        self.plot_timer.start(interval)

    def update_plot(self):
        # Plot all channels, color-coded, with adjustable y-offset for clarity
        self.rawPlotWidget.clear()
        if self.acq and self.acq.plot_buffer:
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            y_offset = self.yOffsetEdit.value()  # Adjustable y-offset for channel separation
            for ch_idx, buf in enumerate(self.acq.plot_buffer):
                data = np.array(buf)
                if data.size > 0:
                    color = colors[ch_idx % len(colors)]
                    pen = pg.mkPen(color=color, width=2)
                    offset_data = data + ch_idx * y_offset
                    self.rawPlotWidget.plot(offset_data, pen=pen, name=f"Ch{ch_idx+1}")
            self.rawPlotWidget.setLabel('bottom', "Sample", units='s')
            self.rawPlotWidget.setLabel('left', "Voltage + offset", units='V')
            # Optionally, add channel labels on the left as text
            for ch_idx in range(len(self.acq.plot_buffer)):
                label = pg.TextItem(f"Ch{ch_idx+1}", color=colors[ch_idx % len(colors)])
                label.setPos(0, ch_idx * y_offset)
                self.rawPlotWidget.addItem(label)

        # Spectrogram: only for first channel (for simplicity)
        if self.acq and self.acq.plot_buffer and self.specCheck.isChecked():
            data = np.array(self.acq.plot_buffer[0])
            if data.size > 0 and data.size >= self.spec_window_size:
                f, t, Sxx = signal.spectrogram(
                    data,
                    fs=self.sample_rate,
                    window='hann',
                    nperseg=self.spec_window_size,
                    noverlap=self.spec_window_size // 2,
                    detrend=False,
                    scaling='density',
                    mode='magnitude'
                )
                freq_mask = (f >= self.min_freq) & (f <= self.max_freq)
                Sxx = Sxx[freq_mask, :]
                f = f[freq_mask]

                if self.domFreqCheck.isChecked() and Sxx.size > 0 and f.size > 0:
                    mean_spectrum = np.mean(Sxx, axis=1)
                    dom_idx = np.argmax(mean_spectrum)
                    dom_freq = f[dom_idx]
                    self.domFreqLabel.setText(f"Dominant Frequency: {dom_freq:.1f} Hz")
                    self.specPlotWidget.setTitle(f"Spectrogram (Dominant: {dom_freq:.1f} Hz)")
                else:
                    self.domFreqLabel.setText("Dominant Frequency: --- Hz")
                    self.specPlotWidget.setTitle("Spectrogram")

                self.specPlotWidget.clear()
                if self.img is None or self.img.scene() is None:
                    self.img = pg.ImageItem(axisOrder='row-major')
                self.specPlotWidget.addItem(self.img)
                self.img.setImage(Sxx, autoLevels=True)
                self.img.setLookupTable(self.spectrogram_lut)

                if t.size > 1 and f.size > 1:
                    xscale = (t[-1] - t[0]) / float(Sxx.shape[1]) if Sxx.shape[1] > 1 else 1
                    yscale = (f[-1] - f[0]) / float(Sxx.shape[0]) if Sxx.shape[0] > 1 else 1
                    transform = QtGui.QTransform()
                    transform.scale(xscale, yscale)
                    transform.translate(0, self.min_freq / yscale)
                    self.img.setTransform(transform)
                    self.specPlotWidget.setLimits(xMin=0, xMax=t[-1], yMin=self.min_freq, yMax=f[-1])
                    self.specPlotWidget.setLabel('bottom', "Time", units='s')
                    self.specPlotWidget.setLabel('left', "Frequency", units='Hz')
            else:
                self.specPlotWidget.clear()
        else:
            self.specPlotWidget.clear()

    def toggle_recording_indicator(self):
        """Toggle the recording indicator between visible and hidden for blinking effect."""
        self.recording_indicator_state = not self.recording_indicator_state
        if self.recording_indicator_state:
            self.recordingIndicator.setText("● RECORDING")
            self.recordingIndicator.setStyleSheet(
                "font-size: 16px; font-weight: bold; padding: 5px; "
                "background-color: #ff0000; color: white; border-radius: 3px;"
            )
        else:
            self.recordingIndicator.setText("● RECORDING")
            self.recordingIndicator.setStyleSheet(
                "font-size: 16px; font-weight: bold; padding: 5px; "
                "background-color: #800000; color: #cccccc; border-radius: 3px;"
            )
    
    def start_recording_indicator(self):
        """Start the blinking recording indicator."""
        self.recording_indicator_state = True
        self.recordingIndicator.setText("● RECORDING")
        self.recordingIndicator.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 5px; "
            "background-color: #ff0000; color: white; border-radius: 3px;"
        )
        self.recording_indicator_timer.start(500)  # Blink every 500ms
    
    def stop_recording_indicator(self):
        """Stop the blinking recording indicator."""
        self.recording_indicator_timer.stop()
        self.recordingIndicator.setText("")
        self.recordingIndicator.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 5px; border-radius: 3px;"
        )

    def start_record(self):
        """Start recording with synchronized playback stimulus."""
        self.recordBtn.setEnabled(False)
        
        # Ensure any previous output task is fully cleaned up
        if self.acq is not None:
            self.acq.force_cleanup_output()
            # Small delay to ensure hardware releases resources
            QtCore.QThread.msleep(50)
        
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Playback Data", "", "Binary files (*.bin)")
        if not filepath:
            self.recordBtn.setEnabled(True)
            return
        
        try:
            pre_stim_duration = int(self.preStimEdit.text())
            stim_duration = int(self.stimEdit.text())
            post_stim_duration = int(self.postStimEdit.text())
            offset_frequency = float(self.offsetCombo.currentText())
            amp_factor = float(self.ampFactorEdit.text())
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Input Error", f"Invalid playback parameter: {e}")
            self.recordBtn.setEnabled(True)
            return
        
        total_duration = pre_stim_duration + stim_duration + post_stim_duration
        rec_duration = total_duration
        
        # Compute dominant frequency from current plot buffer
        if self.acq and len(self.acq.plot_buffer[0]) > 0:
            data = np.array(self.acq.plot_buffer[0])
            self.dominant_freq = compute_dominant_frequency(data, self.sample_rate, self.min_freq, self.max_freq)
        else:
            QtWidgets.QMessageBox.warning(self, "Error", "No data available for frequency analysis.")
            self.recordBtn.setEnabled(True)
            return
        
        # Generate synthetic playback signal
        playback_signal = generate_synthetic_signal(
            self.dominant_freq, offset_frequency, pre_stim_duration, 
            stim_duration, post_stim_duration, self.sample_rate, amp_factor
        )
        
        # Setup recording
        self.acq.samples_to_save = int(rec_duration * self.acq.sample_rate)
        self.acq.acquired_samples = 0
        for buf in self.acq.storage_buffer:
            buf.clear()
        self.acq.recording_complete_callback = lambda: QtCore.QTimer.singleShot(0, self.on_recording_complete)
        self.acq.recording_active = True
        self.acq.logfile_written = False
        self.acq.logfile_callback = self.save_log_file
        
        self.acq.target_split_samples = 0
        self.record_filepath = filepath
        with open(self.record_filepath, 'wb'):
            pass
        self.file_writer = FileWriter(self.acq.storage_buffer, self.buffer_lock, self.record_filepath, self.acq.sample_rate)
        self.file_writer.start()
        
        # Start recording indicator
        self.start_recording_indicator()
        
        # Start playback
        self.acq.start_playback(playback_signal, total_duration)

    def on_recording_complete(self):
        if self.file_writer is not None:
            self.file_writer.stop()
            self.file_writer.join()
        self.acq.recording_start_timestamp = None
        self.recordBtn.setEnabled(True)
        
        # Stop recording indicator
        self.stop_recording_indicator()
        
        gc.collect()
        QtWidgets.QMessageBox.information(self, "Finished", "Recording complete")

    def save_log_file(self):
        log_filename = f"log_{os.path.basename(self.record_filepath).split('.')[0]}.txt"
        log_filepath = os.path.join(os.path.dirname(self.record_filepath), log_filename)
        if self.acq and self.acq.recording_start_timestamp is not None:
            dt = datetime.fromtimestamp(self.acq.recording_start_timestamp)
            timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        else:
            timestamp_str = "N/A"
        try:
            pre_stim = int(self.preStimEdit.text())
            stim = int(self.stimEdit.text())
            post_stim = int(self.postStimEdit.text())
            total_duration = pre_stim + stim + post_stim
        except ValueError:
            total_duration = "N/A"
        
        log_data = {
            "N_Input_Channels": self.acq.num_channels,
            "Sample_Rate": self.acq.sample_rate,
            "Fish_ID": self.fishIdEdit.text(),
            "Temperature": self.tempEdit.text(),
            "Conductivity": self.condEdit.text(),
            "Total_Recording_Duration": str(total_duration),
            "Input_Channels": ', '.join(self.acq.input_channels),
            "Output_Channel": self.acq.output_channel if self.acq.output_channel else "N/A",
            "Recording_Start_Timestamp": timestamp_str,
            "Dominant_Frequency": getattr(self, 'dominant_freq', 'N/A'),
            "Frequency_Offset": self.offsetCombo.currentText() if hasattr(self, 'dominant_freq') else "N/A",
            "Min_Frequency": self.min_freq,
            "Max_Frequency": self.max_freq,
            "Stimulus_Duration": self.stimEdit.text() if hasattr(self, 'dominant_freq') else "N/A",
            "Pre_Stimulus_Duration": self.preStimEdit.text() if hasattr(self, 'dominant_freq') else "N/A",
            "Post_Stimulus_Duration": self.postStimEdit.text() if hasattr(self, 'dominant_freq') else "N/A",
            "Amplitude_Factor": self.ampFactorEdit.text() if hasattr(self, 'dominant_freq') else "N/A"
        }
        with open(log_filepath, 'w') as f:
            for key, value in log_data.items():
                f.write(f"{key}: {value}\n")

    def stop_acquisition(self):
        if self.acq:
            self.acq.stop()
        self.plot_timer.stop()
        self.audioOutputCheck.setChecked(False)
        self.connectBtn.setEnabled(True)
        self.resetBtn.setEnabled(True)
        self.disconnectBtn.setEnabled(False)
        self.recordBtn.setEnabled(False)
        self.testSignalStartBtn.setEnabled(False)
        self.testSignalStopBtn.setEnabled(False)
        self.daqCombo.setEnabled(True)
        self.chanEdit.setEnabled(True)
        self.outputChanEdit.setEnabled(True)
        self.sampleRateEdit.setEnabled(True)
        self.plotDurEdit.setEnabled(True)
        self.refreshRateEdit.setEnabled(True)
        self.specMinEdit.setEnabled(True)
        self.specMaxEdit.setEnabled(True)
        self.specWindowEdit.setEnabled(True)
        self.specCheck.setEnabled(True)

    def reset_device(self):
        dev = self.daqCombo.currentText()
        try:
            nidaqmx.system.Device(dev).reset_device()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", str(e))

    def start_test_signal(self):
        """Start continuous test signal output."""
        try:
            amp_factor = float(self.ampFactorEdit.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Input Error", "Invalid amplitude factor.")
            return
        
        self.testSignalStartBtn.setEnabled(False)
        self.testSignalStopBtn.setEnabled(True)
        self.recordBtn.setEnabled(False)
        
        try:
            self.acq.start_test_signal(amp_factor)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to start test signal: {e}")
            self.testSignalStartBtn.setEnabled(True)
            self.testSignalStopBtn.setEnabled(False)
            self.recordBtn.setEnabled(True)
    
    def stop_test_signal(self):
        """Stop test signal output."""
        self.acq.stop_test_signal()
        self.testSignalStartBtn.setEnabled(True)
        self.testSignalStopBtn.setEnabled(False)
        self.recordBtn.setEnabled(True)
    
    def toggle_audio_output(self, state):
        """Toggle audio output on/off based on checkbox state."""
        if self.acq is None:
            return
        
        if state == QtCore.Qt.Checked:
            success = self.acq.enable_audio_output()
            if not success:
                QtWidgets.QMessageBox.warning(self, "Audio Error", 
                    "Failed to start audio output. Check system audio settings.")
                self.audioOutputCheck.setChecked(False)
        else:
            self.acq.disable_audio_output()
    
    def safe_close(self):
        # Stop and close the DAQ task if still running
        if hasattr(self, "acq") and self.acq is not None:
            try:
                if getattr(self.acq, "ai_task", None) is not None:
                    if self.acq.running:
                        self.acq.stop()
            except Exception as e:
                print(f"Error during safe close (DAQ): {e}")
        # Stop file writer thread if running
        if hasattr(self, "file_writer") and self.file_writer is not None:
            try:
                self.file_writer.stop()
                self.file_writer.join(timeout=2)
            except Exception as e:
                print(f"Error stopping file writer: {e}")
        QtWidgets.qApp.quit()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWin = DataAcquisitionGUI()
    mainWin.show()
    sys.exit(app.exec_())
