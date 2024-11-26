import pyaudio
import numpy as np
import wave

# Define the recording parameters
RATE = 44100  # Sample rate (44.1 kHz)
CHANNELS = 1  # Mono audio
DURATION = 10  # Duration in seconds
CHUNK = 1024  # Size of each audio chunk

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open a stream
stream = p.open(format=pyaudio.paInt16,  # 16-bit resolution
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Recording...")

# Record audio in chunks
frames = []
for _ in range(0, int(RATE / CHUNK * DURATION)):
    data = stream.read(CHUNK)
    frames.append(data)

# Stop the stream and close it
print("Recording finished.")
stream.stop_stream()
stream.close()
p.terminate()

# Save the recorded audio to a WAV file
with wave.open('recorded_audio.wav', 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print("Audio saved as 'recorded_audio.wav'.")
