#!/usr/bin/env python3

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from rtlsdr import RtlSdr
import sys
import time

# --- Configuration ---
SAMPLE_RATE_HZ = 3.2e6 
NUM_SAMPLES = 8192      
GAIN_DB = 40            

# --- Calculated / Derived Parameters ---
EFFECTIVE_BW_HZ = 2.0e6 
STEP_FREQ_HZ = EFFECTIVE_BW_HZ 
FFT_RESOLUTION_HZ = SAMPLE_RATE_HZ / NUM_SAMPLES

# Define the desired sweep range
SWEEP_START_HZ = 100.0e6
SWEEP_END_HZ = 120.0e6

# Global SDR Device
SDR_DEVICE = None

pg.setConfigOptions(antialias=True)

# --- SDR Initialization ---

def initialize_sdr():
    """Initializes the SDR device and calculates the center frequencies needed for the sweep."""
    global SDR_DEVICE
    
    try:
        # Initialize the SDR device
        sdr = RtlSdr()
        sdr.sample_rate = SAMPLE_RATE_HZ
        sdr.gain = GAIN_DB
        
        # 1. Determine the list of center frequencies needed for the sweep
        first_center_freq = SWEEP_START_HZ + EFFECTIVE_BW_HZ / 2
        
        center_freqs = np.arange(
            first_center_freq, 
            SWEEP_END_HZ + STEP_FREQ_HZ, 
            STEP_FREQ_HZ
        )
        
        if len(center_freqs) == 0:
            center_freqs = np.array([first_center_freq])

        SDR_DEVICE = sdr
        
        print(f"--- RTL-SDR Device Initialized ---")
        print(f"Sweep Range: {SWEEP_START_HZ/1e6:.1f} MHz to {SWEEP_END_HZ/1e6:.1f}")
        print(f"Total {len(center_freqs)} capture windows needed.")
        
        return sdr, center_freqs
        
    except Exception as e:
        print(f"\n[ERROR] RTL-SDR device not found or error: {e}")
        SDR_DEVICE = None
        return None, None

# --- PyQtGraph Application Class ---

class SDRWorker(QtCore.QObject):
    dataReady = QtCore.Signal(np.ndarray, np.ndarray)
    
    def __init__(self, sdr, center_freqs, start_index, end_index):
        super().__init__()
        self.sdr = sdr
        self.center_freqs = center_freqs 
        self.start_index = start_index
        self.end_index = end_index
        self.running = True
        
        # This variable is shared with the main thread
        self.new_center_freqs = None 

    @QtCore.pyqtSlot()
    def process_sweep(self):
        """Runs continuously in the worker thread."""
        while self.running:
            
            # 1. Check if Main Thread sent new frequencies (Outer Loop Check)
            if self.new_center_freqs is not None:
                self.center_freqs = self.new_center_freqs 
                self.new_center_freqs = None 
                print(f"[Worker] Range updated. Starting sweep...")

            full_freq_axis = []
            full_power_db = []
            
            try:
                for center_freq in self.center_freqs:
                    
                    # 2. INTERRUPT CHECK: 
                    # Check if frequencies changed *during* the sweep
                    if self.new_center_freqs is not None:
                        print("[Worker] Sweep interrupted by user change.")
                        break # Break 'for' loop to restart 'while' loop immediately

                    self.sdr.center_freq = center_freq
                    
                    # Blocking call (safe in worker thread)
                    samples = self.sdr.read_samples(NUM_SAMPLES)
                    
                    # --- Frequency Analysis (FFT) ---
                    # 1. Optional: Apply a Window Function to reduce spectral leakage (Recommended)
                    # This prevents the "noise floor" from looking artificially high due to sharp cutoffs
                    windowed_samples = samples * np.hanning(NUM_SAMPLES)

                    # 2. Calculate FFT
                    spectrum = np.fft.fft(windowed_samples)

                    # 3. CRITICAL FIX: Normalize by the number of samples
                    spectrum = spectrum / NUM_SAMPLES

                    spectrum_shifted = np.fft.fftshift(spectrum)

                    # 4. Calculate Power in dB
                    # We use abs() to get magnitude. 
                    # resulting unit is dBFS (dB relative to Full Scale, where 0 is clipping)
                    power_db = 20 * np.log10(np.abs(spectrum_shifted) + 1e-10)
                    
                    # Create frequency axis
                    freq_offset = np.linspace(-SAMPLE_RATE_HZ/2, SAMPLE_RATE_HZ/2, NUM_SAMPLES)
                    full_window_freq_axis = freq_offset + center_freq
                    
                    # --- Stitching ---
                    power_slice = power_db[self.start_index:self.end_index]
                    freq_slice = full_window_freq_axis[self.start_index:self.end_index]
                    
                    full_power_db.extend(power_slice)
                    full_freq_axis.extend(freq_slice)

                # Only emit data if we weren't interrupted
                if self.new_center_freqs is None:
                    freq_data_mhz = np.array(full_freq_axis) / 1e6
                    power_data_db = np.array(full_power_db)
                    self.dataReady.emit(freq_data_mhz, power_data_db)

                # Small sleep to prevent CPU hogging and allow update checks
                QtCore.QThread.msleep(100) 
                
            except Exception as e:
                print(f"\n[ERROR] Worker Thread Error: {e}")
                self.running = False
                break 


class RealTimeSweepApp(QtWidgets.QMainWindow):

    @QtCore.pyqtSlot()
    def update_sweep_range(self):
        """Reads input fields and directly updates the worker variable."""
        
        try:
            # 1. Read inputs
            new_start_mhz = float(self.start_freq_input.text())
            new_end_mhz = float(self.end_freq_input.text())
            
            new_start_hz = new_start_mhz * 1e6
            new_end_hz = new_end_mhz * 1e6

            if new_start_hz >= new_end_hz:
                QtWidgets.QMessageBox.warning(self, "Error", "Start > End")
                return

            # 2. Recalculate center frequencies
            first_center_freq = new_start_hz + EFFECTIVE_BW_HZ / 2
            
            new_center_freqs = np.arange(
                first_center_freq, 
                new_end_hz + STEP_FREQ_HZ, 
                STEP_FREQ_HZ
            )
            
            if len(new_center_freqs) == 0:
                new_center_freqs = np.array([first_center_freq])
            
            # 3. Update GUI Plot Limits
            self.plot_item.setXRange(new_start_mhz, new_end_mhz, padding=0)
            self.curve.setData([], []) # Clear old trace


            # Reset Waterfall Data
            self.waterfall_plot_item.setXRange(new_start_mhz, new_end_mhz, padding=0)
            # We reset the buffer size variables so they get recreated in the update loop
            self.current_spectrum_width = 0 

            # 4. DIRECT UPDATE: Bypass the Event Loop and set variable directly
            self.worker.new_center_freqs = new_center_freqs
            
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Input Error", "Please enter valid numbers.")
        except Exception as e:
            print(f"Error during range update: {e}")

    def __init__(self, sdr_device, center_freqs):
        super().__init__()
        
        self.sdr = sdr_device
        self.center_freqs = center_freqs 

        # Calculate stitching indices
        fraction_to_keep = EFFECTIVE_BW_HZ / SAMPLE_RATE_HZ
        samples_to_keep = int(NUM_SAMPLES * fraction_to_keep)
        samples_to_discard = (NUM_SAMPLES - samples_to_keep) // 2
        
        self.start_index = samples_to_discard
        self.end_index = NUM_SAMPLES - samples_to_discard
        
        # --- GUI Setup ---
        main_container = QtWidgets.QWidget()
        self.setCentralWidget(main_container)
        root_layout = QtWidgets.QVBoxLayout(main_container)
        
        # Control Panel
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.addWidget(QtWidgets.QLabel("Start Freq (MHz):"))
        self.start_freq_input = QtWidgets.QLineEdit(f"{SWEEP_START_HZ/1e6:.1f}")
        control_layout.addWidget(self.start_freq_input)
        
        control_layout.addWidget(QtWidgets.QLabel("End Freq (MHz):"))
        self.end_freq_input = QtWidgets.QLineEdit(f"{SWEEP_END_HZ/1e6:.1f}")
        control_layout.addWidget(self.end_freq_input)
        
        self.set_range_button = QtWidgets.QPushButton("Set Range")
        self.set_range_button.clicked.connect(self.update_sweep_range)
        control_layout.addWidget(self.set_range_button)

        root_layout.addLayout(control_layout)
        
        # Plot Widget
        self.central_widget = pg.GraphicsLayoutWidget()
        root_layout.addWidget(self.central_widget) 
        
        # --- TOP PLOT (Spectrum) ---
        self.plot_item = self.central_widget.addPlot(row=0, col=0)
        self.plot_item.setTitle("Real-Time RTL-SDR Frequency Sweep")
        self.plot_item.setLabel('bottom', "Frequency", units='MHz')
        self.plot_item.setLabel('left', "Power", units='dB')
        self.plot_item.setXRange(SWEEP_START_HZ/1e6, SWEEP_END_HZ/1e6, padding=0)
        self.plot_item.setYRange(-90, -40) 

        self.curve = self.plot_item.plot(pen=pg.mkPen(color=(0, 255, 0, 80), width=1))
        
        # --- BOTTOM PLOT (Waterfall) ---
        self.WATERFALL_HEIGHT = 200
        self.current_spectrum_width = 0 

        self.waterfall_plot_item = self.central_widget.addPlot(row=1, col=0)
        self.waterfall_plot_item.setLabel('left', "Time History")
        self.waterfall_plot_item.setXLink(self.plot_item) 
        self.waterfall_plot_item.setYRange(0, self.WATERFALL_HEIGHT, padding=0)
        self.waterfall_plot_item.hideAxis('left') 

        self.waterfall_image = pg.ImageItem()
        self.waterfall_plot_item.addItem(self.waterfall_image)

        # Setup Color Map
        colormap = pg.colormap.get('viridis') 
        self.waterfall_image.setLookupTable(colormap.getLookupTable())
        
        # LEVEL FIX: Map -100 dB (Purple) to 0 dB (Yellow)
        # Adjust these numbers if your signal is still too bright or too dim!
        self.waterfall_image.setLevels([-100, -40]) 

        # Initialize buffer with a very low value (silence)
        self.waterfall_data = np.full((self.WATERFALL_HEIGHT, 10), -100.0)

        # --- THREADING SETUP ---
        self.thread = QtCore.QThread()
        self.worker = SDRWorker(
            sdr=sdr_device, 
            center_freqs=center_freqs, 
            start_index=self.start_index,  
            end_index=self.end_index       
        )

        self.worker.moveToThread(self.thread)
        self.worker.dataReady.connect(self.update_plot_slot)
        self.thread.started.connect(self.worker.process_sweep)
        self.thread.start()
        
    @QtCore.pyqtSlot(np.ndarray, np.ndarray)
    def update_plot_slot(self, freq_data_mhz, power_data_db):
        if len(freq_data_mhz) < 2:
            return

        # 1. Update the spectrum line
        self.curve.setData(freq_data_mhz, power_data_db)

        # --- Update Waterfall Buffer ---
        width = len(power_data_db)
        
        # Resize buffer if the number of samples changed (e.g., range change)
        if self.current_spectrum_width != width:
            self.current_spectrum_width = width
            self.waterfall_data = np.full((self.WATERFALL_HEIGHT, width), -100.0)

        # Scroll up and insert new data
        self.waterfall_data = np.roll(self.waterfall_data, -1, axis=0)
        self.waterfall_data[-1, :] = power_data_db

        # Update the image pixel data
        self.waterfall_image.setImage(self.waterfall_data.T, autoLevels=False)

        # Force the Image to fit the Frequency Axis EXACTLY
        x_start = freq_data_mhz[0]
        x_end = freq_data_mhz[-1]
        width_mhz = x_end - x_start
        
        # This stretches the pixel grid to fill the MHz coordinates
        self.waterfall_image.setRect(QtCore.QRectF(x_start, 0, width_mhz, self.WATERFALL_HEIGHT))

    def closeEvent(self, event):
        self.worker.running = False 
        self.thread.quit()
        self.thread.wait() 
        if self.sdr:
            self.sdr.close()
        super().closeEvent(event)


# --- Main Execution ---
if __name__ == "__main__":
    sdr, center_freqs = initialize_sdr()
    
    if sdr is not None:
        app = QtWidgets.QApplication.instance()
        if not app:
            app = QtWidgets.QApplication(sys.argv)
        
        window = RealTimeSweepApp(sdr, center_freqs)
        window.show()
        
        try:
            sys.exit(app.exec()) 
        except Exception as e:
            print(f"Main loop error: {e}")