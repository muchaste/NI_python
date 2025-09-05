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
import numpy as np
import nidaqmx
from nidaqmx import constants
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from scipy import signal
from datetime import datetime

# Get list of DAQ device names
daqSys = nidaqmx.system.System()
daqList = daqSys.devices.device_names
if not daqList:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    QtWidgets.QMessageBox.critical(None, "DAQ Error", "No DAQ detected, check connection.")
    sys.exit(1)

# ---------------- Data Acquisition Module ----------------
class DataAcquisition:
    def __init__(self, input_channels, sample_rate, min_voltage, max_voltage, refresh_rate, plot_duration):
        # input_channels: list of strings
        self.input_channels = input_channels
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
        self.split_samples = 0  # Sample counter for current split file
        self.samples_to_save = 0
        self.recording_complete_callback = None
        self.recording_start_timestamp = None
        self.session_start_timestamp = None  # Start time of entire recording session
        self.target_split_samples = 0  # Exact samples per split file
        self.logfile_written = False
        self.logfile_callback = None

    def start(self):
        self.running = True
        self.recording_start_timestamp = None
        self.session_start_timestamp = None
        self.split_samples = 0
        self.logfile_written = False
        self.ai_task.start()

    def stop(self):
        self.running = False
        if self.recording_active:
            self.recording_active = False
            if self.recording_complete_callback is not None:
                self.recording_complete_callback()
        self.recording_start_timestamp = None
        self.session_start_timestamp = None
        self.split_samples = 0
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
        
        if self.recording_active:
            n_samples = temp_data.shape[1]
            if self.recording_start_timestamp is None:
                current_time = time.time()
                self.recording_start_timestamp = current_time - (n_samples - 1) / self.sample_rate
                # Set session start timestamp only once for the entire recording session
                if self.session_start_timestamp is None:
                    self.session_start_timestamp = self.recording_start_timestamp
                # Reset split sample counter when new recording/split starts
                self.split_samples = 0
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
            
            # Handle precise file splitting for split files
            if hasattr(self, 'target_split_samples') and self.target_split_samples > 0:
                # Check if we need to split this data chunk
                if self.split_samples + n_samples > self.target_split_samples:
                    # Split the data: exact samples for current file, excess for next
                    samples_for_current = self.target_split_samples - self.split_samples
                    current_data = temp_data[:, :samples_for_current]
                    excess_data = temp_data[:, samples_for_current:n_samples]
                    
                    # Store exact samples for current file
                    for ch_idx in range(self.num_channels):
                        self.storage_buffer[ch_idx].extend(current_data[ch_idx])
                    self.split_samples += samples_for_current
                    
                    # The excess will be handled by the GUI split logic
                    # Store excess data temporarily
                    self._excess_data = excess_data
                    self._excess_samples = excess_data.shape[1]
                    
                    # Update acquired samples with the samples actually stored
                    self.acquired_samples += samples_for_current
                else:
                    # Normal processing - all samples go to current file
                    for ch_idx in range(self.num_channels):
                        self.storage_buffer[ch_idx].extend(temp_data[ch_idx])
                    self.split_samples += n_samples
                    self._excess_data = None
                    self._excess_samples = 0
                    
                    # Update acquired samples
                    self.acquired_samples += n_samples
            else:
                # No splitting - normal processing
                for ch_idx in range(self.num_channels):
                    self.storage_buffer[ch_idx].extend(temp_data[ch_idx])
                self.split_samples += n_samples
                
                # Update acquired samples
                self.acquired_samples += n_samples
                
            if recording_complete:
                self.recording_active = False
                if self.recording_complete_callback is not None:
                    self.recording_complete_callback()
        return 0
        return 0

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
        self.init_ui()

    def init_ui(self):
        controls_layout = QtWidgets.QVBoxLayout()

        # DAQ Settings
        self.daqGroup = QtWidgets.QGroupBox("DAQ Settings")
        daq_layout = QtWidgets.QFormLayout()
        self.daqCombo = QtWidgets.QComboBox()
        self.daqCombo.addItems(daqList)
        daq_layout.addRow("Select DAQ:", self.daqCombo)
        self.chanEdit = QtWidgets.QLineEdit("ai0")
        daq_layout.addRow("Input Channel:", self.chanEdit)
        self.sampleRateEdit = QtWidgets.QLineEdit("20000")
        daq_layout.addRow("Sample Rate (Hz):", self.sampleRateEdit)
        self.daqGroup.setLayout(daq_layout)
        controls_layout.addWidget(self.daqGroup)

        # Plot Settings
        self.plotGroup = QtWidgets.QGroupBox("Plot Settings")
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
        self.yOffsetEdit = QtWidgets.QDoubleSpinBox()
        self.yOffsetEdit.setRange(0.0, 1000.0)
        self.yOffsetEdit.setSingleStep(0.1)
        self.yOffsetEdit.setValue(2.0)
        plot_layout.addRow("Y Offset (V):", self.yOffsetEdit)
        self.plotGroup.setLayout(plot_layout)
        controls_layout.addWidget(self.plotGroup)

        # Recording Settings
        self.expGroup = QtWidgets.QGroupBox("Recording Settings")
        exp_layout = QtWidgets.QFormLayout()
        self.recIdEdit = QtWidgets.QLineEdit()
        exp_layout.addRow("ID:", self.recIdEdit)
        self.recDurEdit = QtWidgets.QLineEdit("60")
        exp_layout.addRow("Duration (s):", self.recDurEdit)
        self.splitFileCheck = QtWidgets.QCheckBox("Enable File Splitting")
        self.splitFileCheck.setChecked(False)
        self.splitFileCheck.stateChanged.connect(self.toggle_split_duration)
        self.splitDurEdit = QtWidgets.QLineEdit("10")
        self.splitDurEdit.setEnabled(False)
        exp_layout.addRow(self.splitFileCheck)
        exp_layout.addRow("Split Duration (s):", self.splitDurEdit)

        btn_layout = QtWidgets.QHBoxLayout()
        self.connectBtn = QtWidgets.QPushButton("Connect")
        self.connectBtn.setStyleSheet("color: green; font-weight: bold;")
        self.disconnectBtn = QtWidgets.QPushButton("Disconnect")
        self.disconnectBtn.setStyleSheet("color: #003366; font-weight: bold;")
        self.recordBtn = QtWidgets.QPushButton("● Record")
        self.recordBtn.setStyleSheet("color: red; font-weight: bold;")
        self.resetBtn = QtWidgets.QPushButton("Reset DAQ")
        self.resetBtn.setStyleSheet("color: black; font-weight: bold;")
        self.closeBtn = QtWidgets.QPushButton("Close")
        self.closeBtn.setStyleSheet("color: black; font-weight: bold;")
        btn_layout.addWidget(self.connectBtn)
        btn_layout.addWidget(self.disconnectBtn)
        btn_layout.addWidget(self.recordBtn)
        btn_layout.addWidget(self.resetBtn)
        btn_layout.addWidget(self.closeBtn)
        exp_layout.addRow(btn_layout)
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
        self.resetBtn.clicked.connect(self.reset_device)
        self.closeBtn.clicked.connect(self.safe_close)

    def toggle_split_duration(self):
        self.splitDurEdit.setEnabled(self.splitFileCheck.isChecked())

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

        self.connectBtn.setEnabled(False)
        self.resetBtn.setEnabled(False)
        self.disconnectBtn.setEnabled(True)
        self.recordBtn.setEnabled(True)
        self.daqCombo.setEnabled(False)
        self.chanEdit.setEnabled(False)
        self.sampleRateEdit.setEnabled(False)
        self.plotDurEdit.setEnabled(False)
        self.refreshRateEdit.setEnabled(False)
        self.specMinEdit.setEnabled(False)
        self.specMaxEdit.setEnabled(False)
        self.specWindowEdit.setEnabled(False)

        self.acq = DataAcquisition(channel_list, sample_rate, -10, 10, refresh_rate, plot_duration)
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

        # File splitting logic - precise sample counting
        if getattr(self, 'split_enabled', False) and self.acq and self.acq.split_samples >= self.acq.target_split_samples:
            if self.next_split_idx < len(self.split_points):
                # Stop current file writer
                if self.file_writer is not None:
                    self.file_writer.stop()
                    self.file_writer.join()
                self.save_log_file()
                
                # Calculate precise start time for next split
                split_duration_actual = self.acq.target_split_samples / self.acq.sample_rate
                next_start_time = self.acq.session_start_timestamp + (self.split_counter * split_duration_actual)
                
                # Setup next split file
                self.split_counter += 1
                self.record_filepath = self._split_filename()
                self.file_writer = FileWriter(self.acq.storage_buffer, self.buffer_lock, self.record_filepath, self.acq.sample_rate)
                self.file_writer.start()
                self.acq.logfile_written = False
                self.acq.recording_start_timestamp = next_start_time  # Precise timestamp
                self.acq.split_samples = 0  # Reset split sample counter
                
                # Handle excess data from previous split
                if hasattr(self.acq, '_excess_data') and self.acq._excess_data is not None:
                    for ch_idx in range(self.acq.num_channels):
                        self.acq.storage_buffer[ch_idx].extend(self.acq._excess_data[ch_idx])
                    self.acq.split_samples += self.acq._excess_samples
                    self.acq._excess_data = None
                    self.acq._excess_samples = 0
                
                self.next_split_idx += 1

    def start_record(self):
        self.recordBtn.setEnabled(False)
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Data", "", "Binary files (*.bin)")
        if not filepath:
            self.recordBtn.setEnabled(True)
            return

        rec_duration = float(self.recDurEdit.text())
        self.acq.samples_to_save = int(rec_duration * self.acq.sample_rate)
        self.acq.acquired_samples = 0
        for buf in self.acq.storage_buffer:
            buf.clear()
        self.acq.recording_complete_callback = lambda: QtCore.QTimer.singleShot(0, self.on_recording_complete)
        self.acq.recording_active = True
        self.acq.logfile_written = False
        self.acq.logfile_callback = self.save_log_file

        self.split_enabled = self.splitFileCheck.isChecked()
        if self.split_enabled:
            try:
                self.split_duration = float(self.splitDurEdit.text())
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Input Error", "Invalid split duration.")
                self.recordBtn.setEnabled(True)
                return
            if self.split_duration >= rec_duration:
                QtWidgets.QMessageBox.warning(self, "Input Error", "Split duration must be smaller than total duration.")
                self.recordBtn.setEnabled(True)
                return
            self.split_samples = int(self.split_duration * self.acq.sample_rate)
            self.acq.target_split_samples = self.split_samples  # Set exact target for DataAcquisition
            self.split_counter = 1
            self.base_filepath = os.path.splitext(filepath)[0]
            self.record_filepath = self._split_filename()
            with open(self.record_filepath, 'wb'):
                pass
            self.file_writer = FileWriter(self.acq.storage_buffer, self.buffer_lock, self.record_filepath, self.acq.sample_rate)
            total_samples = self.acq.samples_to_save
            self.split_points = [self.split_samples * i for i in range(1, int(np.ceil(total_samples / self.split_samples)))]
            self.next_split_idx = 0
        else:
            self.acq.target_split_samples = 0  # No splitting
            self.record_filepath = filepath
            with open(self.record_filepath, 'wb'):
                pass
            self.file_writer = FileWriter(self.acq.storage_buffer, self.buffer_lock, self.record_filepath, self.acq.sample_rate)
        self.file_writer.start()

    def _split_filename(self):
        return f"{self.base_filepath}_{self.split_counter:03d}.bin"

    def on_recording_complete(self):
        if self.file_writer is not None:
            self.file_writer.stop()
            self.file_writer.join()
        self.acq.recording_start_timestamp = None
        self.recordBtn.setEnabled(True)
        QtWidgets.QMessageBox.information(self, "Finished", "Recording complete")

    def save_log_file(self):
        log_filename = f"log_{os.path.basename(self.record_filepath).split('.')[0]}.txt"
        log_filepath = os.path.join(os.path.dirname(self.record_filepath), log_filename)
        if self.acq and self.acq.recording_start_timestamp is not None:
            dt = datetime.fromtimestamp(self.acq.recording_start_timestamp)
            timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        else:
            timestamp_str = "N/A"
        log_data = {
            "N_Input_Channels": self.acq.num_channels,
            "Sample_Rate": self.acq.sample_rate,
            "Recording_ID": self.recIdEdit.text(),
            "Total_Recording_Duration": self.recDurEdit.text(),
            "Split_Recording_Duration": self.splitDurEdit.text() if self.splitFileCheck.isChecked() else "N/A",
            "Input_Channels": ', '.join(self.acq.input_channels),
            "Recording_Start_Timestamp": timestamp_str
        }
        with open(log_filepath, 'w') as f:
            for key, value in log_data.items():
                f.write(f"{key}: {value}\n")

    def stop_acquisition(self):
        if self.acq:
            self.acq.stop()
        self.plot_timer.stop()
        self.connectBtn.setEnabled(True)
        self.resetBtn.setEnabled(True)
        self.disconnectBtn.setEnabled(False)
        self.recordBtn.setEnabled(False)
        self.daqCombo.setEnabled(True)
        self.chanEdit.setEnabled(True)
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
