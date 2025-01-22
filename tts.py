import sys
import torch
import warnings
import scipy.io.wavfile as wavfile
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

# Add Kokoro-82M to Python path
kokoro_path = Path("Kokoro-82M")
sys.path.append(str(kokoro_path.absolute()))

# Import Kokoro modules
from models import build_model
from kokoro import generate

def load_model(device):
    """Try different methods to load the model"""
    model_paths = [
        kokoro_path / "kokoro-v0_19.pth",
        kokoro_path / "fp16" / "kokoro-v0_19-half.pth",
    ]
    
    for model_path in model_paths:
        print(f"\nTrying to load model: {model_path}")
        try:
            # Try loading without weights_only first
            model = build_model(str(model_path), device)
            print(f"Successfully loaded model from {model_path}")
            return model
        except Exception as e:
            print(f"Failed to load model from {model_path}: {str(e)}")
            try:
                # Try alternate loading method
                state_dict = torch.load(str(model_path), map_location=device)
                if isinstance(state_dict, dict) and 'net' in state_dict:
                    model = build_model(state_dict['net'], device)
                    print(f"Successfully loaded model using alternate method from {model_path}")
                    return model
            except Exception as e2:
                print(f"Alternate loading method also failed: {str(e2)}")
    
    raise Exception("Failed to load model using any available method")

def main():
    try:
        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Load model using our new function
        model = load_model(device)
        
        # Load voice
        voice_name = 'af'  # Default voice (Bella & Sarah mix)
        voice_path = kokoro_path / "voices" / f"{voice_name}.pt"
        voicepack = torch.load(voice_path, weights_only=True).to(device)
        print(f'Loaded voice: {voice_name}')
        
        # Get text input
        text = input("Enter text to synthesize (or press Enter for default): ").strip()
        if not text:
            text = "Hey there! I'm Loftwah, a Senior DevOps Engineer. I automate all the things, turn coffee into code, and occasionally flip tables when the production pipeline breaks. But don't worry, I always catch them before they hit the ground! (╯°□°)╯︵ ┻━┻  ┬─┬ノ(°□°ノ)"
        
        # Generate audio
        print("\nGenerating audio...")
        audio, phonemes = generate(model, text, voicepack, lang=voice_name[0])
        
        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Save audio
        output_file = output_dir / "output.wav"
        wavfile.write(str(output_file), 24000, audio)
        print(f"\nAudio saved to: {output_file}")
        print("\nPhonemes used:")
        print(phonemes)

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure all model files are properly downloaded")
        print("2. Check if CUDA is properly installed (if using GPU)")
        print("3. Try using CPU instead of GPU by modifying the script")
        sys.exit(1)

if __name__ == "__main__":
    main()