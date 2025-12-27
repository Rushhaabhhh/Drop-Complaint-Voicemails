"""
Main script to process voicemail files
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from src.voicemail_processor import VoicemailProcessor
import argparse


def main():
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Process voicemail audio files')
    parser.add_argument('--audio-dir', type=str, default='data/audio_files',
                      help='Directory containing audio files')
    parser.add_argument('--output', type=str, default='results/timestamps.json',
                      help='Output JSON file for results')
    parser.add_argument('--verbose', action='store_true', default=True,
                      help='Print detailed progress')
    parser.add_argument('--single-file', type=str, default=None,
                      help='Process a single audio file instead of all files')
    
    args = parser.parse_args()
    
    # Check for API keys
    deepgram_key = os.getenv('DEEPGRAM_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not deepgram_key and not openai_key:
        print("ℹ️  No API key found in environment variables.")
        print("   The system will work with beep and silence detection only.")
        print("   For better accuracy, add DEEPGRAM_API_KEY to your .env file.")
        print("   Get a free key at: https://console.deepgram.com/signup\n")
    
    # Initialize processor
    processor = VoicemailProcessor(
        deepgram_api_key=deepgram_key,
        openai_api_key=openai_key
    )
    
    # Get audio files
    if args.single_file:
        # Single file mode
        audio_files = [Path(args.single_file)]
        if not audio_files[0].exists():
            print(f"❌ Audio file not found: {args.single_file}")
            return
        print(f"Processing single file: {args.single_file}")
    else:
        # Directory mode
        audio_dir = Path(args.audio_dir)
        
        # Check if the path is actually a file (common mistake)
        if audio_dir.is_file():
            print(f"⚠️  You provided a file path to --audio-dir: {audio_dir}")
            print(f"   Use --single-file instead for individual files:")
            print(f"   python main.py --single-file {audio_dir}")
            return
        
        if not audio_dir.exists():
            print(f"❌ Audio directory not found: {audio_dir}")
            print(f"   Please check the path or create test files by running:")
            print(f"   python tests/generate_test_audio.py")
            return
        
        # Try multiple audio formats
        audio_files = []
        for pattern in ['*.wav', '*.mp3', '*.m4a', '*.ogg', '*.flac']:
            audio_files.extend(audio_dir.glob(pattern))
        
        # Also try files without extensions (like test1, test2)
        for item in audio_dir.iterdir():
            if item.is_file() and not item.suffix and not item.name.startswith('.'):
                audio_files.append(item)
        
        # Remove duplicates and sort
        audio_files = sorted(set(audio_files), key=lambda x: x.name)
        
        if not audio_files:
            print(f"❌ No audio files found in {audio_dir}")
            print(f"   Looked for: .wav, .mp3, .m4a, .ogg, .flac files")
            print(f"\n   Files in directory:")
            for item in audio_dir.iterdir():
                print(f"     - {item.name}")
            return
    
    print(f"\n🎙️  Voicemail Compliance Drop System")
    print(f"{'='*80}\n")
    print(f"Found {len(audio_files)} audio file(s) to process\n")
    
    # Process all files
    all_results = {}
    
    for audio_file in audio_files:
        results = processor.process_voicemail(str(audio_file), verbose=args.verbose)
        all_results[audio_file.name] = results
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📊 Summary Results")
    print(f"{'='*80}")
    
    for filename, result in all_results.items():
        print(f"\n{filename}:")
        print(f"  Drop Time: {result['drop_timestamp']:.2f}s" if result['drop_timestamp'] else "  Drop Time: N/A")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Reasoning: {result['reasoning']}")
        
        signals = result['detected_signals']
        print(f"  Beep: {'Yes' if signals['beep_detected'] else 'No'}" +
              (f" (at {signals['beep_timestamp']:.2f}s)" if signals['beep_detected'] else ""))
        print(f"  Greeting End: {'Yes' if signals['greeting_ended'] else 'No'}" +
              (f" (at {signals['greeting_end_timestamp']:.2f}s)" if signals['greeting_ended'] else ""))
    
    print(f"\n{'='*80}")
    print(f"✅ Results saved to: {output_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()