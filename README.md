# NI_python
Python code for data acquisition using National Instruments DAQs and the nidaqmx API

# GUI-based programs
Apps that let you visualize and record analog input from National Instruments devices. They are based on [PyQT5](https://pypi.org/project/PyQt5/) and [PyQtGraph](https://www.pyqtgraph.org/).

# Getting Started (for new users)

## Step 1 — Install Miniforge3
Download and install **[Miniforge3](https://github.com/conda-forge/miniforge/releases/latest)** (choose the Windows x86_64 installer).  
During installation, keep the default options. This installs conda and Python for you.

## Step 2 — Download this repository
Click the green **Code** button on GitHub and choose **Download ZIP**, then unzip it to a folder of your choice (e.g. `C:\Users\YourName\NI_python`).

## Step 3 — Create the Python environment (one-time setup)
1. Open the **Miniforge Prompt** from the Start menu.
2. Navigate to the folder where you unzipped the repository, e.g.:
   ```
   cd C:\Users\YourName\NI_python
   ```
3. Run the following command to install all required packages automatically:
   ```
   conda env create -f environment.yml
   ```
   This may take a few minutes. You only need to do this once.

## Step 4 — Run a script
**Easiest way:** double-click any of the `Run_*.bat` files in the repository folder.

| Batch file | Script launched |
|---|---|
| `Run_01_Record.bat` | Multi-channel recording |
| `Run_02_Analyze_Freq.bat` | Wave-type frequency analysis |
| `Run_03_Analyze_Pulse.bat` | Pulse-type analysis |
| `Run_04_Record_LED.bat` | Recording with LED sync |
| `Run_Playback_01_Record.bat` | Playback experiment recording |
| `Run_Playback_02_Analyze.bat` | Playback experiment analysis |
| `Run_Playback_03_Analyze_FM.bat` | WAV / FM analysis |

> **Note:** The **analysis** scripts (`Analyze_*.bat`) work on any Windows PC without any NI hardware.  
> The **recording** scripts (`Record_*.bat`) require an NI DAQ device to be connected **and** the [NI-DAQmx driver](https://www.ni.com/en/support/downloads/drivers/download.ni-daq-mx.html) to be installed.

## Troubleshooting
- If double-clicking a `.bat` file shows an error about `conda` not being found, open the **Miniforge Prompt** and run the script from there:
  ```
  conda activate ni_python
  python "GUI_based\02_Analyze_ai_multichan_freq.py"
  ```
- If you encounter issues with NI-DAQmx, refer to [NI's official documentation](https://nidaqmx-python.readthedocs.io/en/stable/).

# System Requirements
* Windows OS (required for NI-DAQmx hardware)
* [NI-DAQmx driver](https://www.ni.com/en/support/downloads/drivers/download.ni-daq-mx.html) installed — **only needed for recording scripts**
* A supported National Instruments DAQ device — **only needed for recording scripts**

# 01_Record_ai_multichan.py
Acquire analog input on a various number of channels.
Execute in your fav IDE or simply use your command line/Anaconda prompt like so:
```python "Path\to\gui_program.py"```
Record files with optional file splitting (e.g., for large files) in a .bin file and metadata in a .txt logfile.

![Data Acquisition](https://github.com/muchaste/NI_python/releases/download/v0.1-alpha/data_acquisition_smaller.gif)


# 02_Analyze_ai_multichan.py
Tkinter app for visualization of recordings. Reads in metadata from the logfile.
![Data Analysis](https://github.com/muchaste/NI_python/releases/download/v0.1-alpha/data_analysis_module.png)

# Old Versions
The first generation of apps was based on [TKinter](https://docs.python.org/3/library/tkinter.html) and inspired by [example code by DavidFI](https://forums.ni.com/t5/Example-Code/Python-Voltage-Continuous-Input-py/ta-p/3938650). These apps can be found in the [Tkinter](https://github.com/muchaste/NI_python/tree/main/GUI_based/Tkinter) folder - however, they are deprecated and will not be developed anymore.
