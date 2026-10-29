#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check whether (and how strongly) the playback stimulus is present in recordings
made with 01_Record_and_Playback_multichan.py.

Method: lock-in demodulation with the playback-copy channel as phase reference.
  1. ref(t) = exp(j*phase of copy)   (Hilbert transform -> unit phasor; during
     pre/post-stimulus, where the copy is zero, a synthetic phasor at f_stim)
  2. Each channel is multiplied by conj(ref): the stimulus moves to 0 Hz, the
     fish (5 Hz away) to +-5 Hz, everything else elsewhere.
  3. A Hann-weighted average over LOCKIN_WIN seconds (= low-pass) keeps only the
     0 Hz part -> complex amplitude z = A*exp(j*phi) of the stimulus in volts.
Per channel: stimulus amplitude, baseline (pre/post) amplitude, SNR, gain and
phase stability relative to the copy, plus a Welch-PSD band-power cross-check.
All values are absolute (V), so recordings can be compared directly.

Run in Spyder: fill LOG_FILES or leave it empty to pick files in a dialog.

Authors: Stefan Mucha/Copilot
"""

# %% Settings
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert, welch

LOG_FILES = [
    r"E:\Lizi Kalandadze larval chirps\db27\log_ALB24_m5Hz.txt",
    r"E:\Lizi Kalandadze larval chirps\db37\log_ALB24_m5Hz.txt",
]  # leave empty ([]) to select log files with a file dialog

LOCKIN_WIN = 2.0      # s, Hann window -> fish 5 Hz away is suppressed by > 60 dB
LOCKIN_STEP = 0.5     # s
FADE = 1.0            # s, stimulus fade in/out of the playback program
GUARD = 0.5           # s, extra margin around stimulus on/offset
NPERSEG = 2 ** 16     # Welch segment (3.3 s at 20 kHz -> 0.3 Hz resolution)
BAND_HW = 1.0         # Hz, half-width of stimulus band for PSD band power
FLANK = (1.5, 3.0)    # Hz, inner/outer distance of PSD noise-floor flanks
SEARCH_HW = 5.0       # Hz, search range for f_stim around the logged value
PLOT_SPAN = 15.0      # Hz, PSD plot range around f_stim

# %% Select files
if not LOG_FILES:
    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    LOG_FILES = []
    while True:
        path = filedialog.askopenfilename(
            title=f"Select log file #{len(LOG_FILES) + 1} (Cancel when done)",
            filetypes=[("Log files", "log_*.txt"), ("Text files", "*.txt")])
        if not path:
            break
        LOG_FILES.append(path)
    root.destroy()
if not LOG_FILES:
    raise SystemExit("No log files selected.")

# %% Analyze each recording
results = []
for log_path in LOG_FILES:
    # --- load log + binary data (float64, columns: time_ms, ch1, ..., chN) ---
    with open(log_path) as f:
        log = dict(line.rstrip("\n").split(": ", 1) for line in f if ": " in line)
    folder, name = os.path.split(log_path)
    bin_path = os.path.join(folder, name[len("log_"):].rsplit(".", 1)[0] + ".bin")
    n_ch = int(log["N_Input_Channels"])
    data = np.fromfile(bin_path, dtype="f8").reshape(-1, n_ch + 1)[:, 1:].T
    copy_name = log.get("Playback_Copy_Channel", "N/A")
    if copy_name in ("N/A", ""):
        raise ValueError(f"{log_path}: no playback copy channel recorded")
    names = log["Input_Channels"].split(",") + [copy_name]
    copy = data[-1]

    fs = float(log["Sample_Rate"])
    t = np.arange(data.shape[1]) / fs
    t_on = float(log["Pre_Stimulus_Duration"])
    t_off = t_on + float(log["Stimulus_Duration"])
    s0, s1 = t_on + FADE + GUARD, t_off - FADE - GUARD           # steady-state stimulus
    f_fish_log = float(log["Dominant_Frequency"])

    # --- stimulus frequency from copy channel (zero-padded FFT, steady part) ---
    seg = copy[int(s0 * fs):int(s1 * fs)]
    n_fft = 1 << int(np.ceil(np.log2(len(seg))) + 2)
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n_fft))
    f_ax = np.fft.rfftfreq(n_fft, 1 / fs)
    m = np.abs(f_ax - (f_fish_log + float(log["Frequency_Offset"]))) <= SEARCH_HW
    f_stim = f_ax[m][np.argmax(spec[m])]

    # --- lock-in reference: phase of copy during stimulus, synthetic outside ---
    ref = hilbert(copy)
    ref = ref / np.maximum(np.abs(ref), 1e-12)
    in_stim = (t >= t_on) & (t <= t_off)
    ref = np.where(in_stim, ref, np.exp(1j * 2 * np.pi * f_stim * t))

    # --- windowed demodulation: z = complex stimulus amplitude per window ---
    n_win, n_step = int(LOCKIN_WIN * fs), int(LOCKIN_STEP * fs)
    w = np.hanning(n_win)
    starts = np.arange(0, data.shape[1] - n_win + 1, n_step)
    tc = (starts + n_win / 2) / fs
    z = np.empty((len(names), len(starts)), dtype=complex)
    for k, s in enumerate(starts):
        z[:, k] = 2 * (data[:, s:s + n_win] * np.conj(ref[s:s + n_win]) * w).sum(axis=1) / w.sum()

    half = LOCKIN_WIN / 2
    steady = (tc - half >= s0) & (tc + half <= s1)
    quiet = (tc + half <= t_on - GUARD) | (tc - half >= t_off + GUARD)

    # --- per-channel metrics ---
    ch = {}
    psd = {}
    for i, cname in enumerate(names):
        a = np.abs(z[i, steady])
        h = z[i, steady] / z[-1, steady]                         # complex gain re copy
        a_stim = np.median(a)
        a_quiet = np.median(np.abs(z[i, quiet]))

        # Welch PSD cross-check (absolute V^2/Hz), stimulus vs. pre-stimulus
        f, p = welch(data[i, int(s0 * fs):int(s1 * fs)], fs, nperseg=NPERSEG)
        f0, p0 = welch(data[i, :int((t_on - GUARD) * fs)], fs,
                       nperseg=min(NPERSEG, int((t_on - GUARD) * fs)))
        df = f[1] - f[0]
        band = np.abs(f - f_stim) <= BAND_HW
        flank = (np.abs(f - f_stim) >= FLANK[0]) & (np.abs(f - f_stim) <= FLANK[1])
        a_psd = np.sqrt(2 * p[band].sum() * df)
        snr_psd = 10 * np.log10(p[band].sum() / (np.median(p[flank]) * band.sum()))

        ch[cname] = dict(amp=a_stim, cv=np.std(a) / a_stim, quiet=a_quiet,
                         snr=20 * np.log10(a_stim / a_quiet),
                         gain=20 * np.log10(np.median(np.abs(h))),
                         phase_r=np.abs(np.mean(h / np.abs(h))),
                         amp_psd=a_psd, snr_psd=snr_psd)
        psd[cname] = (f, p, f0, p0)

    results.append(dict(label=os.path.basename(folder) + "/" + name, names=names,
                        f_stim=f_stim, tc=tc, z=z, ch=ch, psd=psd))
    del data, copy, ref

# %% Report
# A_stim: lock-in amplitude (median over steady stimulus), CV: its variation over
# time, A_quiet: same frequency in pre/post-stimulus, phase R: 1 = phase-locked
# to the copy, 0 = random. A_psd/SNR_psd: Welch band power vs. local noise floor.
hdr = (f"{'chan':>5} {'A_stim mV':>10} {'CV %':>6} {'A_quiet mV':>10} {'SNR dB':>7} "
       f"{'gain re copy dB':>15} {'phase R':>7} {'A_psd mV':>9} {'SNR_psd dB':>10}")
for r in results:
    print(f"\n{r['label']}   f_stim = {r['f_stim']:.2f} Hz")
    print(hdr + "\n" + "-" * len(hdr))
    for cname, c in r["ch"].items():
        print(f"{cname:>5} {1e3 * c['amp']:10.4f} {100 * c['cv']:6.1f} {1e3 * c['quiet']:10.4f} "
              f"{c['snr']:7.1f} {c['gain']:15.1f} {c['phase_r']:7.3f} "
              f"{1e3 * c['amp_psd']:9.4f} {c['snr_psd']:10.1f}")

if len(results) > 1:
    ref_r = results[0]
    print(f"\nDifference re. '{ref_r['label']}' (dB):")
    for r in results[1:]:
        for cname in ref_r["names"]:
            a, b = ref_r["ch"][cname], r["ch"][cname]
            print(f"  {r['label']:>30} {cname:>5}: amplitude "
                  f"{20 * np.log10(b['amp'] / a['amp']):+6.1f} dB, "
                  f"gain re copy {b['gain'] - a['gain']:+6.1f} dB")

# %% Plot: PSD around f_stim (left) and lock-in amplitude over time (right)
names = results[0]["names"]
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fig, axes = plt.subplots(len(names), 2, figsize=(13, 3 * len(names)), squeeze=False)
for r, col in zip(results, colors):
    for i, cname in enumerate(names):
        f, p, f0, p0 = r["psd"][cname]
        m = np.abs(f - r["f_stim"]) <= PLOT_SPAN
        m0 = np.abs(f0 - r["f_stim"]) <= PLOT_SPAN
        axes[i, 0].plot(f[m] - r["f_stim"], 10 * np.log10(p[m]), color=col,
                        label=f"{r['label']} stim")
        axes[i, 0].plot(f0[m0] - r["f_stim"], 10 * np.log10(p0[m0]), color=col, ls=":",
                        label=f"{r['label']} pre")
        axes[i, 0].set_ylabel(f"{cname}\nPSD (dB re V²/Hz)")
        axes[i, 1].plot(r["tc"], 20 * np.log10(np.abs(r["z"][i])), color=col, label=r["label"])
        axes[i, 1].set_ylabel(f"{cname}\namplitude at f_stim (dB re 1 V)")
axes[0, 0].legend(fontsize=8)
axes[0, 1].legend(fontsize=8)
axes[-1, 0].set_xlabel("Frequency re. f_stim (Hz)")
axes[-1, 1].set_xlabel("Time (s)")
fig.tight_layout()
plt.show()
