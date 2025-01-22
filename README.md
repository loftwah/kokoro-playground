# 🎙️ Kokoro TTS - Complete Beginner-Friendly Guide

## 🚀 Getting Started

First, let's set up everything you need. Don't worry if you're new to this - we'll go through it step by step.

```bash
# Create a project directory and move into it
mkdir kokoro-project
cd kokoro-project

# Create a special folder (virtual environment) to keep all our stuff organized
# This helps avoid conflicts with other Python projects you might have
python3 -m venv venv

# Now we'll activate this environment
# On Windows, use: venv\Scripts\activate
# On Mac/Linux, use:
source venv/bin/activate

# Install all the programs we need
# Don't worry if you see a lot of text - that's normal!
pip install phonemizer torch transformers scipy munch

# Install espeak-ng - this helps convert text to speech sounds
# On Ubuntu/Debian Linux:
sudo apt-get install espeak-ng
# On Mac (using Homebrew):
# brew install espeak
# On Windows, download from: https://github.com/espeak-ng/espeak-ng/releases

# Install Git LFS (Large File Storage) - THIS IS ESSENTIAL!
# Without this, you won't get the model files properly!

# On Ubuntu/Debian:
sudo apt-get install git-lfs

# On Mac:
brew install git-lfs

# On Windows:
# Download and install from: https://git-lfs.com

# IMPORTANT: Initialize Git LFS
git lfs install

# Now get the Kokoro software (this will take a while - it's downloading the model files!)
git clone https://huggingface.co/hexgrad/Kokoro-82M
cd Kokoro-82M

# Verify you got everything (if any of these are missing, the clone didn't work properly)
ls Kokoro-82M/kokoro-v0_19.pth    # Should see the main model file
ls Kokoro-82M/voices              # Should see the voice files
ls Kokoro-82M/models.py           # Should see the model code
ls Kokoro-82M/kokoro.py           # Should see the generation code

# Make sure Python can find the model code
export PYTHONPATH="$PYTHONPATH:$PWD/Kokoro-82M"
```

## 🎭 Understanding the Voices

Kokoro comes with several different voices you can use. Think of it like having different actors ready to read your text. They're organized by accent and gender:

### 🇺🇸 American English Voices

- `af` - The "champion" voice (a perfect blend of Bella & Sarah)
- `af_bella` - Bella's voice (female, American accent)
- `af_sarah` - Sarah's voice (female, American accent)
- `af_nicole` - Nicole's voice (female, American accent)
- `af_sky` - Sky's voice (female, American accent)
- `am_adam` - Adam's voice (male, American accent)
- `am_michael` - Michael's voice (male, American accent)

### 🇬🇧 British English Voices

- `bf_emma` - Emma's voice (female, British accent)
- `bf_isabella` - Isabella's voice (female, British accent)
- `bm_george` - George's voice (male, British accent)
- `bm_lewis` - Lewis's voice (male, British accent)

## 📚 How to Use Kokoro (With Examples!)

### The Simple Way (Using Command Line)

This is the easiest way to get started. Open your terminal and try these commands:

```bash
# Want to see all available voices? Try this:
python tts.py --list

# Let's make your first text-to-speech! This uses the default voice:
python tts.py --text "Hello! This is my first time using Kokoro!"

# Try different voices (notice how we specify the voice we want):
python tts.py --voice af_bella --text "Hi, I'm Bella! I have an American accent!"
python tts.py --voice bm_george --text "Hello chaps! I'm George, with a British accent!"

# Save the audio with a specific name (it'll go in the 'output' folder):
python tts.py --voice af_sky --text "Hi, I'm Sky!" --output my_first_audio.wav
```

### The More Flexible Way (Using Python)

This method gives you more control. Here's a simple example with lots of comments explaining what's happening:

```python
# Import all the tools we need
from models import build_model
from kokoro import generate
import torch
import scipy.io.wavfile as wavfile

# Set up the program to use your GPU if you have one, or CPU if you don't
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")  # This tells you which it's using

# Load the main Kokoro system
model = build_model('Kokoro-82M/kokoro-v0_19.pth', device)

# Let's try the default voice first (the champion voice)
voice = torch.load('Kokoro-82M/voices/af.pt', weights_only=True).to(device)
text = "Welcome! This is my first time using Kokoro with Python!"
audio, _ = generate(model, text, voice, lang='a')  # 'a' means American accent
wavfile.write('output/my_first_python_tts.wav', 24000, audio)

# Now let's try a British voice
british_voice = torch.load('Kokoro-82M/voices/bm_george.pt', weights_only=True).to(device)
text = "Jolly good! Now I'm speaking with a British accent!"
audio, _ = generate(model, text, british_voice, lang='b')  # 'b' means British accent
wavfile.write('output/british_voice_test.wav', 24000, audio)
```

## 🎨 Making Your Own Custom Voices

The really cool part about Kokoro is that you can mix different voices together to create your own custom voice! Here are three ways to do it, from simple to advanced:

### Method 1: Quick and Simple Mix

This is the easiest way to make a custom voice:

```python
# First, let's create a new voice by mixing two existing ones
import torch
from pathlib import Path

# Get the voices we want to mix
print("Loading voices to mix...")
bella = torch.load('Kokoro-82M/voices/af_bella.pt', weights_only=True)
sarah = torch.load('Kokoro-82M/voices/af_sarah.pt', weights_only=True)

# Mix them together (70% Bella, 30% Sarah)
print("Creating mix: 70% Bella, 30% Sarah")
mixed_voice = 0.7 * bella + 0.3 * sarah

# Save our new voice (IMPORTANT: save it in the 'voices' folder!)
print("Saving the new voice...")
torch.save(mixed_voice, 'Kokoro-82M/voices/my_first_mix.pt')
print("✅ Saved new voice as 'my_first_mix.pt'")
```

Now you can use your new voice just like any other voice!

```bash
# Try your new voice from the command line:
python tts.py --voice my_first_mix --text "Hello! This is my very own custom voice!"
```

### Method 2: Create and Test Immediately

This method shows you how to create a voice and test it right away:

```python
import torch
from pathlib import Path
from models import build_model
from kokoro import generate
import scipy.io.wavfile as wavfile

def create_and_test_custom_voice():
    print("Starting custom voice creation...")

    # 1. Load the voices we want to mix
    print("Loading voices...")
    bella = torch.load('voices/af_bella.pt', weights_only=True)
    nicole = torch.load('Kokoro-82M/voices/af_nicole.pt', weights_only=True)

    # 2. Create the mix (60% Bella, 40% Nicole)
    print("Mixing voices: 60% Bella, 40% Nicole")
    custom_voice = 0.6 * bella + 0.4 * nicole

    # 3. Save our new voice
    print("Saving new voice...")
    torch.save(custom_voice, 'voices/bella_nicole_mix.pt')

    # 4. Let's test it right away!
    print("Setting up Kokoro to test the voice...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model('kokoro-v0_19.pth', device)

    # 5. Load our custom voice
    print("Loading our new voice for testing...")
    voice = torch.load('voices/bella_nicole_mix.pt', weights_only=True).to(device)

    # 6. Generate some test audio
    print("Generating test audio...")
    text = "This is my custom voice mixing Bella and Nicole! How does it sound?"
    audio, _ = generate(model, text, voice, lang='a')

    # 7. Save the test audio
    print("Saving test audio...")
    wavfile.write('output/my_custom_voice_test.wav', 24000, audio)
    print("✅ Done! Check 'output/my_custom_voice_test.wav' to hear your new voice!")

# Run everything!
create_and_test_custom_voice()
```

### Method 3: Advanced Voice Mixing (Mix Multiple Voices)

This is the most flexible way to create custom voices:

```python
import torch
from pathlib import Path

def mix_voices(voices_dict, output_name):
    """
    Mix multiple voices together with different ratios

    voices_dict: A dictionary showing which voices to mix and how much of each to use
                For example: {'af_bella': 0.6, 'af_sarah': 0.4}
    output_name: What to call your new voice (don't include .pt)
    """
    print(f"\nStarting to create voice mix: {output_name}")
    voices = []
    weights = []

    # Load all the voices we want to mix
    print("Loading voices:")
    for voice_file, ratio in voices_dict.items():
        print(f"- Loading {voice_file} (ratio: {ratio})")
        voice = torch.load(f'Kokoro-82M/voices/{voice_file}.pt', weights_only=True)
        voices.append(voice)
        weights.append(ratio)

    # Make sure our ratios add up to 1.0 (100%)
    total = sum(weights)
    weights = [w/total for w in weights]

    # Mix all the voices together
    print("Mixing voices...")
    mixed = sum(voice * weight for voice, weight in zip(voices, weights))

    # Save our new voice
    output_path = Path('Kokoro-82M/voices') / f'{output_name}.pt'
    torch.save(mixed, output_path)
    print(f"✅ Successfully saved mixed voice to: {output_path}")
    return mixed

# Example: Let's create a super-mix of three voices!
print("Creating a custom mix of three voices...")

voices_to_mix = {
    'af_bella': 0.4,     # 40% Bella
    'af_sarah': 0.3,     # 30% Sarah
    'af_nicole': 0.3     # 30% Nicole
}
mixed_voice = mix_voices(voices_to_mix, 'my_triple_mix')

# Now let's test our new voice!
print("\nTesting the new voice mix...")

# Set up Kokoro
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = build_model('kokoro-v0_19.pth', device)

# Load our custom voice
print("Loading the custom voice...")
custom_voice = torch.load('Kokoro-82M/voices/my_triple_mix.pt', weights_only=True).to(device)

# Generate a test audio
print("Generating test audio...")
text = "Hello everyone! This voice is a custom mix of Bella, Sarah, and Nicole!"
audio, _ = generate(model, text, custom_voice, lang='a')
wavfile.write('output/triple_mix_test.wav', 24000, audio)
print("✅ Done! Check 'output/triple_mix_test.wav' to hear your new voice!")

print("\nYou can also use your new voice from the command line:")
print("python tts.py --voice my_triple_mix --text \"Testing my new voice!\"")
```

## 🚨 Troubleshooting Common Problems

If things aren't working, don't worry! Here are some common issues and how to fix them:

### Problem 1: "espeak-ng not found" error

```bash
# On Ubuntu/Debian Linux, try:
sudo apt-get update
sudo apt-get install espeak-ng

# On Mac:
brew install espeak

# On Windows:
# Download and install from: https://github.com/espeak-ng/espeak-ng/releases
```

### Problem 2: CUDA (GPU) errors

If you have a graphics card but see errors:

```python
# Add this at the start of your script
import torch
torch.cuda.empty_cache()  # This clears your GPU memory
```

## 📝 Tips for Getting the Best Results

1. **Voice Selection Tips:**

   - For American accents, always use `lang='a'`
   - For British accents, always use `lang='b'`
   - Match the language to the voice (don't use British voices with 'a' or vice versa)

2. **Text Tips:**

   - Use proper punctuation (periods, commas, etc.)
   - Break long texts into shorter sentences
   - Use quotation marks for dialogue
   - Add commas where you want slight pauses

3. **Voice Mixing Tips:**

   - Start with 50-50 mixes to test compatibility
   - Keep track of your mixing ratios
   - Try mixing similar accents together first
   - Save all custom voices in the `voices` folder
   - Give your custom voices clear, descriptive names

4. **Performance Tips:**
   - Using a GPU is faster but not required
   - Clear your GPU memory between large batches
   - Start with short text samples when testing new voices

## 🎮 Need Help?

Join the [Kokoro Discord](https://discord.gg/QuGxSWBfQy) where you can:

- Get help with any problems
- Share your custom voice mixes
- See what others are creating
- Stay updated on new features

Remember: There's no such thing as a stupid question. Everyone was a beginner once!
