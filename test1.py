import pyaudio
import numpy as np
import tkinter as tk
from scipy.fftpack import fft
from threading import Thread
import time

# Audio Parameters
RATE = 44100  # Sampling rate
CHANNELS = 1  # Mono
CHUNK = 1024  # Size of audio chunks (samples per frame)
FFT_WINDOW_SIZE = 1024  # FFT window size
UPDATE_INTERVAL = 1  # Update GUI every 5 seconds

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

# GUI Class
class AudioAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Audio Analyzer")
        self.root.geometry("400x200")

        self.label = tk.Label(root, text="Listening...", font=("Arial", 16))
        self.label.pack(pady=20)

        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=20)

        self.running = True
        self.audio_thread = Thread(target=self.real_time_audio_processing)
        self.audio_thread.start()

    def update_gui(self, text):
        self.result_label.config(text=text)

    def stop(self):
        self.running = False
        self.root.destroy()

    def real_time_audio_processing(self):
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        xf = np.fft.rfftfreq(FFT_WINDOW_SIZE, 1 / RATE)
        window = 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, FFT_WINDOW_SIZE, False)))

        while self.running:
            # Read audio data
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(float)

            # Apply FFT
            fft_data = np.abs(np.fft.rfft(audio_data * window))
            fft_data /= np.max(fft_data, initial=1)

            # Find dominant notes
            top_notes = find_top_notes(fft_data, xf)

            # Prepare the result string
            if top_notes:
                print(top_notes[0])
                result = "\n".join([f"{note}: {freq:.1f} Hz (Amp: {amp:.2f})" for freq, note, amp in top_notes])
            else:
                result = "No dominant notes detected."

            # Update GUI
            self.root.after(0, self.update_gui, result)
            time.sleep(UPDATE_INTERVAL)  # Wait for the next update

        # Cleanup
        stream.stop_stream()
        stream.close()
        p.terminate()

# Main Program
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzerApp(root)

    # Bind the window close event to stop the audio thread
    root.protocol("WM_DELETE_WINDOW", app.stop)
    root.mainloop()
