import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# Audio Parameters
RATE = 44100  # Sampling rate
CHANNELS = 1  # Mono
CHUNK = 1024  # Size of audio chunks (samples per frame)
FFT_WINDOW_SIZE = 1024  # FFT window size
FPS = RATE // CHUNK  # Frames per second (for real-time updates)

# Note Range
FREQ_MIN = 10  # Minimum frequency to analyze
FREQ_MAX = 1000  # Maximum frequency to analyze
TOP_NOTES = 3  # Number of dominant notes to display

# Note Names
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Helper Functions
def freq_to_number(f):
    """Convert frequency to MIDI note number."""
    return 69 + 12 * np.log2(f / 440.0)

def number_to_freq(n):
    """Convert MIDI note number to frequency."""
    return 440 * 2.0**((n - 69) / 12.0)

def note_name(n):
    """Get the note name from a MIDI note number."""
    return NOTE_NAMES[n % 12] + str(int(n / 12 - 1))

def find_top_notes(fft, xf, num=TOP_NOTES):
    """Find the top dominant frequencies and their corresponding notes."""
    if np.max(fft) < 0.001:
        return []
    indices = np.argsort(fft)[-num:][::-1]
    results = []
    for idx in indices:
        freq = xf[idx]
        if freq < FREQ_MIN or freq > FREQ_MAX:
            continue
        amplitude = fft[idx]
        note = note_name(int(round(freq_to_number(freq))))
        results.append((freq, note, amplitude))
    return results

# Real-Time Audio Processing
def real_time_audio_processing():
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Open a Stream
    stream = p.open(format=pyaudio.paInt16,  # 16-bit resolution
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Listening... Press Ctrl+C to stop.")

    # Frequency Bins
    xf = np.fft.rfftfreq(FFT_WINDOW_SIZE, 1 / RATE)
    window = 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, FFT_WINDOW_SIZE, False)))

    try:
        while True:
            # Read audio data
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(float)
            
            # Apply FFT
            fft_data = np.abs(np.fft.rfft(audio_data * window))
            
            # Normalize FFT
            fft_data /= np.max(fft_data, initial=1)
            
            # Find dominant notes
            top_notes = find_top_notes(fft_data, xf)
            if top_notes:
                #print("Dominant Notes:", [(f"{note}: {freq:.1f}Hz", f"Amplitude: {amp:.2f}") for freq, note, amp in top_notes])
                print([(f"{note} ") for freq,note, amp in top_notes])
            else:
                print("No dominant notes detected.")
            
    except KeyboardInterrupt:
        print("\nStopped Listening.")
    
    # Cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()

# Run the Real-Time Audio Processing
real_time_audio_processing()
