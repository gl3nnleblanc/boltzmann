"""
    Author: G.S. LeBlanc
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import sys
from scipy.interpolate import interp1d

from ctypes import *
from dwfconstants import *

SAMPLE_FREQ=8192
SAMPLES=8192

class ADSUnit:
    """
    A class representing an ADS Unit to read from,
    for the sake of abstracting away the API.
    """

    def __init__(self):
        """ Initializes the ADS unit, performing setup """
        dwf = cdll.LoadLibrary("libdwf.so")
        try:
            self.hdwf = c_int()
            self.sts = c_byte()
            self.version = create_string_buffer(16)
            dwf.FDwfGetVersion(self.version)

            print("Opening device")
            dwf.FDwfDeviceOpen(c_int(-1), byref(self.hdwf))
            if self.hdwf.value == hdwfNone.value:
                print("failed to open device")
                quit()
        except Exception as e:
            print("An error occured :(")
            print(e)
            dwf.FDwfDeviceCloseAll()

    def read_sample(self,
                    frequency=SAMPLE_FREQ,
                    sample_count=SAMPLES,
                    voltage_range=5):
        """
            Reads samples from the ADS.
            Args:
                frequency (float): The sampling frequency.
                sample_count (int): The number of samples to take.
                voltage_range (float): The range of voltage recorded.
            Returns:
                An array of sampled points
        """
        try:
            dwf = cdll.LoadLibrary("libdwf.so")
            rgdSamples = (c_double*sample_count)()

            print("Preparing to read sample...")

            #set up acquisition
            dwf.FDwfAnalogInFrequencySet(self.hdwf, c_double(frequency))
            dwf.FDwfAnalogInBufferSizeSet(self.hdwf, c_int(sample_count))
            dwf.FDwfAnalogInChannelEnableSet(self.hdwf, c_int(0), c_bool(True))
            dwf.FDwfAnalogInChannelRangeSet(self.hdwf, c_int(0), c_double(voltage_range))

            #wait at least 2 seconds for the offset to stabilize
            #time.sleep(2)

            #begin acquisition
            dwf.FDwfAnalogInConfigure(self.hdwf, c_bool(False), c_bool(True))
            print("   waiting to finish")

            while True:
                dwf.FDwfAnalogInStatus(self.hdwf, c_int(1), byref(self.sts))
                if self.sts.value == DwfStateDone.value :
                    break
                time.sleep(0.01)
            print( "Acquisition finished")

            dwf.FDwfAnalogInStatusData(self.hdwf, 0, rgdSamples, sample_count)
            rgpy=[0.0]*len(rgdSamples)
            for i in range(0,len(rgpy)):
                rgpy[i]=rgdSamples[i]
            return rgpy
        except Exception as e:
            print("An error occured :<")
            print(e)
            dwf.FDwfDeviceCloseAll()
            return

    def show_version(self):
        print("DWF Version: ", str(self.version.value))

    def close(self):
        dwf = cdll.LoadLibrary("libdwf.so")
        dwf.FDwfDeviceCloseAll()


if __name__ == "__main__":
    # Gets gain table
    gains = pd.read_csv("../gain.csv")

    # Instantiate ADS
    ads = ADSUnit()

    num_samples = int(input("Number of samples to average?\n"))
    doing_background = input("Collecting background noise? (Y/N)\n")
    while not (doing_background == 'Y' or doing_background == 'N'):
        doing_background = input("Please enter Y or N.\n")
    cursor = str(input("Cursor value?\n"))

    # Perform averaging over sample stream
    avg = np.zeros(SAMPLES//2 - 1)
    for i in range(num_samples):
        print(f"Sample {i+1}:")
        # Read from ADS
        data = np.array(ads.read_sample())
        # Normalize
        data = data * 1e9 / SAMPLE_FREQ
        fourier = abs(np.fft.fft(data))
        fourier = fourier[1:len(fourier)//2] # Exclude negative frequency
        avg = (avg * i + fourier) / (i + 1)

    # Close ADS
    ads.close()

    # Get frequency array
    freq = np.fft.fftfreq(len(data), d=1/SAMPLE_FREQ)
    freq = freq[1:len(freq)//2]

    # Normalize gain
    gains = pd.read_csv("../gain.csv")
    interp_freq = gains["freq"]
    interp_gain = gains["gain"]
    f = interp1d(interp_freq, interp_gain, kind='cubic')

    avg = avg / f(freq)

    # Background
    if doing_background == 'Y':
        background = pd.DataFrame()
        background["freq"] = freq
        background["val"] = avg
        background.to_csv("../background.csv")
    else:
        background = pd.read_csv("../background.csv")
        avg -= background["val"]

    # Plot
    plt.plot(freq, avg, color="orange", label="Data")
    #plt.plot(freq, [cursor for _ in enumerate(freq)], color="red", label="Cursor")
    plt.title(f"Average spectrum over {num_samples} samples")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("nV / sqrt(Hz)")
    plt.legend()
    plt.show()
