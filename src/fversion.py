import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import wave
import os
import threading
from tflite_runtime.interpreter import Interpreter
import time

# === Setup ===
sample_rate = 16000  # Hz
duration = 1  # seconds
output_file = 'output.wav'
model_path = os.path.join(os.path.dirname(__file__), 'trained_model__f16.tflite')
stop_flag = False

# Load TFLite model
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess audio
def load_audio(file_path):
    with wave.open(file_path, 'rb') as wf:
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0  # Normalize to [-1, 1]
        return audio

# Thread for quitting on "q"
def wait_for_quit():
    global stop_flag
    while True:
        user_input = input("Type 'q' and press Enter to stop:\n")
        if user_input.lower() == 'q':
            stop_flag = True
            break

# Start the input thread
threading.Thread(target=wait_for_quit, daemon=True).start()

print("=== Starting Drone Detection Loop ===")

try:
    while not stop_flag:
        print("Recording 1s sample...")
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        write(output_file, sample_rate, recording)

        audio = load_audio(output_file)

        # === BOOST: amplify signal by 10x ===
        boosted_audio = audio * 10.0
        max_val = np.max(np.abs(boosted_audio))
        if max_val > 1.0:
            boosted_audio = boosted_audio / max_val
            print("Boosted audio normalized to avoid clipping.")
        else:
            print("Boosted audio (no normalization needed).")

        audio = boosted_audio

        if audio.shape[0] != 16000:
            print(f"Invalid audio length: {audio.shape[0]} samples. Skipping...")
            continue

        input_data = np.expand_dims(audio, axis=(0, -1))

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        probability = output[0][0]
        print(f"Model output: {probability:.4f}")
        if probability >= 0.35:
            print("Drone detected!\n")
        else:
            print("No drone detected.\n")

        time.sleep(0.1)

    print("=== Detection Stopped by User ===")

except KeyboardInterrupt:
    print("Interrupted by user.")
