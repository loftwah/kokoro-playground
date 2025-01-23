import sys
import torch
import warnings
import scipy.io.wavfile as wavfile
from pathlib import Path
import argparse
import numpy as np
from pydub import AudioSegment
import io

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

def crossfade(a, b, overlap_samples=2000):
    """Crossfade two audio segments"""
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)
    
    # Create fade curves
    fade_out = torch.linspace(1, 0, overlap_samples)
    fade_in = torch.linspace(0, 1, overlap_samples)
    
    # Apply crossfade
    a_end = a[-overlap_samples:] * fade_out
    b_start = b[:overlap_samples] * fade_in
    
    # Combine with crossfade
    result = torch.cat([
        a[:-overlap_samples],
        (a_end + b_start),
        b[overlap_samples:]
    ])
    
    return result

def process_long_text(model, text, voicepack, lang, max_chars=200):
    """Process longer text by splitting it into manageable chunks with context"""
    # Split text at sentence boundaries
    sentences = text.replace('...', '…').replace('。', '.').replace('!', '! ').replace('?', '? ').split('.')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Group sentences into chunks WITHOUT repeating content
    for sentence in sentences:
        if current_length + len(sentence) > max_chars and current_chunk:
            chunks.append(' '.join(current_chunk))
            # Start fresh with new chunk
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
        print(f"Chunk text: {chunk[:50]}...")  # Debug print to see chunk boundaries
        audio, phonemes = generate(model, chunk, voicepack, lang=lang)
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        all_audio.append(audio)
        all_phonemes.extend(phonemes)
    
    # Combine chunks with crossfading but without overlapping content
    final_audio = all_audio[0]
    for next_audio in all_audio[1:]:
        final_audio = crossfade(final_audio, next_audio)
    
    return final_audio.numpy(), all_phonemes

def main():
    parser = argparse.ArgumentParser(description='Kokoro TTS Generator')
    parser.add_argument('--text', help='Text to synthesize')
    parser.add_argument('--voice', default='af', help='Voice to use (default: af)')
    parser.add_argument('--output', help='Output filename (will be saved in output directory)')
    parser.add_argument('--format', default='mp3', choices=['mp3', 'wav', 'ogg', 'aac', 'opus'],
                      help='Output audio format (default: mp3)')
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
        
        # Determine output filename and format
        if args.output:
            output_file = output_dir / args.output
            # If no extension in provided filename, add the format
            if not output_file.suffix:
                output_file = output_file.with_suffix(f'.{args.format}')
        else:
            output_file = output_dir / f"output_{args.voice}.{args.format}"
            
        # First save as WAV in memory
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, 24000, audio)
        wav_buffer.seek(0)
        
        # Convert to desired format
        audio_segment = AudioSegment.from_wav(wav_buffer)
        
        # Format-specific export settings
        format_settings = {
            'mp3': {'format': 'mp3', 'bitrate': '192k'},
            'wav': {'format': 'wav'},
            'ogg': {'format': 'ogg', 'codec': 'libvorbis', 'bitrate': '192k'},
            'aac': {'format': 'm4a', 'codec': 'aac', 'bitrate': '192k'},
            'opus': {'format': 'opus', 'codec': 'libopus', 'bitrate': '160k'}
        }
        
        # Get export settings for chosen format
        export_settings = format_settings[args.format]
        
        try:
            # For OGG format
            if args.format == 'ogg':
                audio_segment.export(str(output_file), format='ogg', parameters=['-acodec', 'libvorbis'])
            # For AAC format
            elif args.format == 'aac':
                audio_segment.export(str(output_file), format='adts', parameters=['-acodec', 'aac'])
            # For other formats
            else:
                audio_segment.export(str(output_file), **export_settings)
            print(f"\nAudio saved to: {output_file}")
        except Exception as e:
            # Fallback to MP3 if the chosen format fails
            print(f"\nWarning: Failed to save in {args.format} format ({str(e)}). Falling back to MP3...")
            fallback_file = output_file.with_suffix('.mp3')
            audio_segment.export(str(fallback_file), format='mp3', bitrate='192k')
            print(f"Audio saved to: {fallback_file}")

    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\nUsage:")
        print("List voices:  python tts.py --list")
        print("Basic usage: python tts.py --text 'Hello world'")
        print("Change voice: python tts.py --voice af_bella --text 'Hello world'")
        print("Custom format: python tts.py --text 'Hello' --format ogg")
        print("Custom output: python tts.py --text 'Hello' --output custom.wav")
        print("\nSupported formats: mp3, wav, ogg, aac, opus")
        print("All files are saved in the 'output' directory")
        print("Run 'python tts.py --list' to see all available voices")
    else:
        main()