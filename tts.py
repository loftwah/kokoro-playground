import sys
import warnings
import argparse
from pathlib import Path
from kokoro import KPipeline
import soundfile as sf

# Suppress warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description='Kokoro TTS Generator')
    parser.add_argument('--text', help='Text to synthesize')
    parser.add_argument('--voice', default='af_heart', help='Voice to use (default: af_heart)')
    parser.add_argument('--output', help='Output filename (will be saved in output directory)')
    parser.add_argument('--format', default='wav', choices=['wav', 'mp3', 'ogg'],
                      help='Output audio format (default: wav)')
    parser.add_argument('--list', action='store_true', help='List available voices')
    args = parser.parse_args()

    if args.list:
        print("\nAvailable voices:")
        print("\nAmerican English (a):")
        print("af_heart - Heart (Default)")
        print("af_bella - Bella")
        print("af_sarah - Sarah")
        print("af_nicole - Nicole")
        print("af_sky - Sky")
        print("am_adam - Adam")
        print("am_michael - Michael")
        print("\nBritish English (b):")
        print("bf_emma - Emma")
        print("bf_isabella - Isabella") 
        print("bm_george - George")
        print("bm_lewis - Lewis")
        print("\nFrench (f):")
        print("ff_siwis - SIWIS")
        print("\nHindi (h):")
        print("hf_alpha - Alpha")
        print("hf_beta - Beta")
        print("\nSee VOICES.md for complete list")
        return

    try:
        # Get text input
        text = args.text if args.text else input("Enter text to synthesize: ").strip()
        
        # Initialize the pipeline with the appropriate language code
        lang_code = args.voice[0]  # First letter of voice name indicates language
        pipeline = KPipeline(lang_code=lang_code)
        
        # Generate audio
        print("\nGenerating audio...")
        generator = pipeline(text, voice=args.voice)
        
        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Process each generated segment
        for i, (gs, ps, audio) in enumerate(generator):
            # Determine output filename
            if args.output:
                if len(list(generator)) == 1:  # Only one segment
                    output_file = output_dir / args.output
                else:
                    # Multiple segments - append number
                    name = Path(args.output).stem
                    suffix = Path(args.output).suffix
                    output_file = output_dir / f"{name}_{i}{suffix}"
            else:
                output_file = output_dir / f"output_{args.voice}_{i}.{args.format}"
            
            # Save audio
            sf.write(str(output_file), audio, 24000)
            print(f"Saved segment {i+1} to: {output_file}")

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