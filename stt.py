from faster_whisper import WhisperModel

# Load the model (downloads it first time)
# "tiny" is smallest/fastest, "base" is good balance
model = WhisperModel("tiny", device="cpu", compute_type="int8")

# Path to your audio file
audio_file = "output/output.wav"

# Perform transcription
segments, info = model.transcribe(audio_file, beam_size=1)

# Print the transcription
for segment in segments:
    print(segment.text)