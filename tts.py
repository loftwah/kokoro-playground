import sys
import torch
import warnings
import scipy.io.wavfile as wavfile
from pathlib import Path
import argparse
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

def check_requirements():
    """Check if all required components are present"""
    requirements = [
        ('Kokoro-82M/kokoro-v0_19.pth', 'Main model file'),
        ('Kokoro-82M/voices', 'Voice directory'),
        ('Kokoro-82M/models.py', 'Model definition'),
        ('Kokoro-82M/kokoro.py', 'Generation code')
    ]
    
    missing = []
    for req, desc in requirements:
        if not Path(req).exists():
            missing.append(f"{desc} ({req})")
    
    if missing:
        print("\nMissing required files:")
        for item in missing:
            print(f"- {item}")
        print("\nPlease ensure all Kokoro-82M files are in place")
        sys.exit(1)

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
            model = build_model(str(model_path), device)
            print(f"Successfully loaded model from {model_path}")
            return model
        except Exception as e:
            print(f"Failed to load model from {model_path}: {str(e)}")
            try:
                state_dict = torch.load(str(model_path), map_location=device)
                if isinstance(state_dict, dict) and 'net' in state_dict:
                    model = build_model(state_dict['net'], device)
                    print(f"Successfully loaded model using alternate method from {model_path}")
                    return model
            except Exception as e2:
                print(f"Alternate loading method also failed: {str(e2)}")
    
    raise Exception("Failed to load model using any available method")

def process_long_text(model, text, voicepack, lang, max_chars=200):
    """Process longer text by splitting it into manageable chunks"""
    # Split text at sentence boundaries
    sentences = text.replace('...', '…').replace('。', '.').replace('!', '! ').replace('?', '? ').split('.')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Group sentences into chunks
    for sentence in sentences:
        if current_length + len(sentence) > max_chars and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Process each chunk
    all_audio = []
    all_phonemes = []
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nProcessing chunk {i}/{len(chunks)}...")
        audio, phonemes = generate(model, chunk, voicepack, lang=lang)
        # Convert numpy array to tensor if necessary
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        all_audio.append(audio)
        all_phonemes.extend(phonemes)
    
    # Combine all audio chunks and convert back to numpy array
    final_audio = torch.cat(all_audio)
    return final_audio.numpy(), all_phonemes

def main():
    parser = argparse.ArgumentParser(description='Kokoro TTS Generator')
    parser.add_argument('--text', help='Text to synthesize')
    parser.add_argument('--voice', default='af', help='Voice to use (default: af)')
    parser.add_argument('--output', help='Output filename (will be saved in output directory)')
    parser.add_argument('--list', action='store_true', help='List available voices')
    args = parser.parse_args()
    
    try:
        # Check requirements first
        check_requirements()

        if args.list:
            print("\nAvailable voices:")
            print("af - Default (Bella & Sarah mix)")
            print("af_bella - Bella (American)")
            print("af_sarah - Sarah (American)")
            print("af_nicole - Nicole (American)")
            print("af_sky - Sky (American)")
            print("am_adam - Adam (American)")
            print("am_michael - Michael (American)")
            print("bf_emma - Emma (British)")
            print("bf_isabella - Isabella (British)")
            print("bm_george - George (British)")
            print("bm_lewis - Lewis (British)")
            return

        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Load model
        model = load_model(device)
        
        # Load voice
        voice_path = kokoro_path / "voices" / f"{args.voice}.pt"
        if not voice_path.exists():
            print(f"\nError: Voice file not found: {voice_path}")
            print("Run 'python tts.py --list' to see available voices")
            sys.exit(1)
            
        voicepack = torch.load(voice_path, weights_only=True).to(device)
        print(f'Loaded voice: {args.voice}')
        
        # Get text input
        text = args.text if args.text else input("Enter text to synthesize: ").strip()
        
        # Generate audio
        print("\nGenerating audio...")
        if len(text) > 500:  # Adjust this threshold as needed
            audio, phonemes = process_long_text(model, text, voicepack, lang=args.voice[0])
        else:
            audio, phonemes = generate(model, text, voicepack, lang=args.voice[0])
        
        # Save audio
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Always use output directory
        if args.output:
            output_file = output_dir / args.output
        else:
            output_file = output_dir / f"output_{args.voice}.wav"
            
        wavfile.write(str(output_file), 24000, audio)
        print(f"\nAudio saved to: {output_file}")

    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\nUsage:")
        print("List voices:  python tts.py --list")
        print("Basic usage: python tts.py --text 'Hello world'")
        print("Change voice: python tts.py --voice af_bella --text 'Hello world'")
        print("Custom output: python tts.py --text 'Hello' --output custom.wav")
        print("\nAll files are saved in the 'output' directory")
        print("Run 'python tts.py --list' to see all available voices")
    else:
        main()