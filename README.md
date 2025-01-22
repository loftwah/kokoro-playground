# 🎙️ Kokoro TTS - The Ultimate Guide

## 🚀 Quick Start

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install phonemizer torch transformers scipy munch

# Install espeak-ng (required for phonemes)
sudo apt-get install espeak-ng
```

## 🎭 Available Voices

### 🇺🇸 American English Voices

- `af` - The champion voice (50-50 mix of Bella & Sarah)
- `af_bella` - Bella's voice
- `af_sarah` - Sarah's voice
- `af_nicole` - Nicole's voice
- `af_sky` - Sky's voice
- `am_adam` - Adam's voice
- `am_michael` - Michael's voice

### 🇬🇧 British English Voices

- `bf_emma` - Emma's voice
- `bf_isabella` - Isabella's voice
- `bm_george` - George's voice
- `bm_lewis` - Lewis's voice

## 🔧 Voice Mixing Magic

Want to create your own voice mix? It's super easy! Here's how to mix voices:

```python
import torch

# Load two voices you want to mix
voice1 = torch.load('voices/af_bella.pt', weights_only=True)
voice2 = torch.load('voices/af_sarah.pt', weights_only=True)

# Create a 50-50 mix
mixed_voice = torch.mean(torch.stack([voice1, voice2]), dim=0)

# Want a custom ratio? (e.g., 70% voice1, 30% voice2)
mixed_voice = 0.7 * voice1 + 0.3 * voice2

# Save your creation
torch.save(mixed_voice, 'voices/my_custom_voice.pt')
```

## 🎯 Running the TTS

```python
from models import build_model
from kokoro import generate
import torch

# Set up the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = build_model('kokoro-v0_19.pth', device)

# Load a voice (e.g., the champion mix)
voice = torch.load('voices/af.pt', weights_only=True).to(device)

# Generate speech
text = "Time to deploy to production! (╯°□°)╯︵ ┻━┻"
audio, phonemes = generate(model, text, voice, lang='a')  # 'a' for American, 'b' for British

# Save to WAV (24kHz)
import scipy.io.wavfile as wavfile
wavfile.write('output.wav', 24000, audio)
```

## 🧙‍♂️ Pro Tips

1. **Language Selection**:

   - Use `lang='a'` for American English
   - Use `lang='b'` for British English
   - The first letter of the voice name tells you which to use!

2. **Voice Quality**:

   - All voices are high quality
   - The `af` (Bella + Sarah mix) was the champion in the TTS arena
   - Mix and match to create your perfect voice!

3. **Performance**:
   - GPU recommended but not required
   - Works fine on CPU, just slower
   - Model is only 82M parameters (tiny but mighty!)

## 🚨 Troubleshooting

```bash
# If espeak-ng gives you trouble:
sudo apt-get update
sudo apt-get install espeak-ng
```

If you see CUDA errors but have a GPU:

```python
# Try this before loading the model
torch.cuda.empty_cache()
```

## 🎉 Fun Facts

- Kokoro is tiny (82M params) but beat models 15x larger
- Trained on less than 100 hours of audio
- Total training cost was only about $400 on A100s
- Apache 2.0 licensed - go wild!

## 🎮 Voice Mixing Playground

Here's a script to experiment with voice mixing:

```python
import torch

def mix_voices(voice1_name, voice2_name, ratio=0.5):
    """
    Mix two voices with a given ratio
    ratio: Amount of voice1 (0.0 to 1.0)
    """
    v1 = torch.load(f'voices/{voice1_name}.pt', weights_only=True)
    v2 = torch.load(f'voices/{voice2_name}.pt', weights_only=True)
    return (ratio * v1) + ((1 - ratio) * v2)

# Example: Create a 70-30 mix of Bella and Sarah
custom_voice = mix_voices('af_bella', 'af_sarah', 0.7)
torch.save(custom_voice, 'voices/my_mix.pt')
```

Remember: You can mix any number of voices and adjust the ratios to create your perfect TTS voice!

## 📝 Notes

- Model works best with clear, well-punctuated text
- Great for long-form content and narration
- Have fun mixing voices to create new ones!

## 🎩 Credits

Big thanks to:

- @yl4579 for StyleTTS 2
- @rzvzn for training Kokoro
- @Pendrokar for the TTS arena

Need help? Join the [Kokoro Discord](https://discord.gg/QuGxSWBfQy)
