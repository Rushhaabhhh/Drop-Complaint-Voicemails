"""
Enhanced Main Script - Run voicemail processing with improved accuracy
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
    parser = argparse.ArgumentParser(
        description='Enhanced Voicemail Drop Compliance System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in directory
  python main.py --audio-dir data/audio_files
  
  # Process single file
  python main.py --single-file path/to/voicemail.wav
  
  # Silent mode (minimal output)
  python main.py --audio-dir data/audio_files --quiet
        """
    )
    
    parser.add_argument(
        '--audio-dir',
        type=str,
        default='data/audio_files',
        help='Directory containing audio files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/timestamps.json',
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--single-file',
        type=str,
        default=None,
        help='Process a single audio file'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output (opposite of verbose)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'csv', 'both'],
        default='json',
        help='Output format'
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    # Check API keys
    deepgram_key = os.getenv('DEEPGRAM_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    print(f"\n{'='*80}")
    print(f"🎙️  Enhanced Voicemail Drop Compliance System")
    print(f"{'='*80}\n")
    
    if not deepgram_key and not openai_key:
        print("⚠️  No transcription API key found")
        print("   System will use beep + silence detection only")
        print("   For better accuracy, add to .env file:")
        print("   • DEEPGRAM_API_KEY (recommended - free tier at deepgram.com)")
        print("   • OPENAI_API_KEY (alternative)\n")
    elif deepgram_key:
        print("✅ Deepgram API configured - full transcription enabled\n")
    else:
        print("✅ OpenAI API configured - full transcription enabled\n")
    
    # Initialize processor
    processor = VoicemailProcessor(
        deepgram_api_key=deepgram_key,
        openai_api_key=openai_key
    )
    
    # Get audio files
    audio_files = []
    
    if args.single_file:
        file_path = Path(args.single_file)
        if not file_path.exists():
            print(f"❌ File not found: {args.single_file}")
            return
        audio_files = [file_path]
    else:
        audio_dir = Path(args.audio_dir)
        
        if audio_dir.is_file():
            print(f"⚠️  You provided a file to --audio-dir")
            print(f"   Use: python main.py --single-file {audio_dir}")
            return
        
        if not audio_dir.exists():
            print(f"❌ Directory not found: {audio_dir}")
            print(f"   Create directory and add voicemail files")
            return
        
        # Find audio files
        patterns = ['*.wav', '*.mp3', '*.m4a', '*.ogg', '*.flac', '*.aac']
        for pattern in patterns:
            audio_files.extend(audio_dir.glob(pattern))
        
        # Include files without extensions
        for item in audio_dir.iterdir():
            if item.is_file() and not item.suffix:
                audio_files.append(item)
        
        audio_files = sorted(set(audio_files), key=lambda x: x.name)
        
        if not audio_files:
            print(f"❌ No audio files found in {audio_dir}")
            print(f"   Supported: .wav, .mp3, .m4a, .ogg, .flac, .aac")
            return
    
    print(f"Found {len(audio_files)} file(s) to process\n")
    
    # Process all files
    all_results = {}
    
    for i, audio_file in enumerate(audio_files, 1):
        if verbose:
            print(f"[{i}/{len(audio_files)}]")
        
        results = processor.process_voicemail(str(audio_file), verbose=verbose)
        all_results[audio_file.name] = results
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # JSON output
    if args.format in ['json', 'both']:
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # CSV output
    if args.format in ['csv', 'both']:
        csv_path = output_path.with_suffix('.csv')
        _save_csv(all_results, csv_path)
    
    # Print summary table
    _print_summary_table(all_results)
    
    # Print detailed results
    if verbose:
        _print_detailed_results(all_results)
    
    print(f"\n{'='*80}")
    print(f"✅ Results saved to: {output_path}")
    if args.format == 'both':
        print(f"✅ CSV saved to: {csv_path}")
    print(f"{'='*80}\n")


def _print_summary_table(all_results: dict):
    """Print summary table"""
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY TABLE")
    print(f"{'='*80}\n")
    
    # Header
    print(f"{'File':<30} {'Drop Time':<12} {'Method':<12} {'Confidence':<12}")
    print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    
    # Rows
    for filename, result in all_results.items():
        name = filename[:27] + "..." if len(filename) > 30 else filename
        drop_time = f"{result['drop_timestamp']:.2f}s" if result['drop_timestamp'] else "N/A"
        method = result['method'].upper()
        confidence = f"{result['confidence']:.2f}"
        
        print(f"{name:<30} {drop_time:<12} {method:<12} {confidence:<12}")
    
    print(f"\n{'='*80}")


def _print_detailed_results(all_results: dict):
    """Print detailed results for each file"""
    print(f"\n{'='*80}")
    print(f"📋 DETAILED RESULTS")
    print(f"{'='*80}\n")
    
    for filename, result in all_results.items():
        print(f"📁 {filename}")
        print(f"   ⏰ Drop: {result['drop_timestamp']:.2f}s" if result['drop_timestamp'] else "   ⏰ Drop: N/A")
        print(f"   🎯 Method: {result['method'].upper()}")
        print(f"   📊 Confidence: {result['confidence']:.2f}")
        print(f"   💭 {result['reasoning']}")
        
        signals = result['detected_signals']
        
        # Beep
        beep = signals['beep']
        if beep['detected']:
            print(f"   🔔 Beep: {beep['timestamp']:.2f}s "
                  f"({beep['frequency']:.0f}Hz, conf: {beep['confidence']:.2f})")
        
        # Speech
        speech = signals['speech']
        if speech['greeting_ended']:
            print(f"   💬 Greeting: {speech['timestamp']:.2f}s "
                  f"(\"{speech['phrase']}\", conf: {speech['confidence']:.2f})")
        
        # Transcript preview
        transcript = speech['transcript']
        if transcript:
            preview = transcript[:80] + "..." if len(transcript) > 80 else transcript
            print(f"   📝 \"{preview}\"")
        
        print()


def _save_csv(all_results: dict, csv_path: Path):
    """Save results to CSV"""
    import csv
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Filename',
            'Drop_Timestamp',
            'Method',
            'Confidence',
            'Beep_Detected',
            'Beep_Time',
            'Greeting_Ended',
            'Greeting_Time',
            'Transcript'
        ])
        
        # Data
        for filename, result in all_results.items():
            signals = result['detected_signals']
            writer.writerow([
                filename,
                f"{result['drop_timestamp']:.2f}" if result['drop_timestamp'] else "",
                result['method'],
                f"{result['confidence']:.2f}",
                "Yes" if signals['beep']['detected'] else "No",
                f"{signals['beep']['timestamp']:.2f}" if signals['beep']['timestamp'] else "",
                "Yes" if signals['speech']['greeting_ended'] else "No",
                f"{signals['speech']['timestamp']:.2f}" if signals['speech']['timestamp'] else "",
                signals['speech']['transcript']
            ])


if __name__ == "__main__":
    main()