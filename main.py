#!/usr/bin/env python3
"""Voicemail greeting end detector - MAIN CLI."""

import sys
import os
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from state_machine import VoicemailAnalyzer
from results_aggregator import ResultsAggregator


def analyze_file(path: Path, analyzer: VoicemailAnalyzer) -> None:
    """Analyze single audio file."""
    try:
        print(f"\n🎯 Analyzing: {path}")
        v = analyzer.analyze_file(str(path))
        
        print("=" * 70)
        print(f"📁 File: {path.name}")
        print("-" * 70)
        print(f"⏱️  Decision time:      {v.decision_time_sec:.2f}s")
        print(f"⏱️  Total duration:     {v.total_duration_sec:.2f}s")
        print(f"💯 Confidence:         {v.confidence:.2%}")
        print(f"🔊 Beep detected:      {'YES ✅' if v.beep_detected else 'NO ❌'}")
        print(f"📝 Reason:             {v.reason}")
        print("=" * 70)
        return v
    except Exception as e:
        print(f"❌ ERROR analyzing {path}: {e}")
        return None

def find_audio_files(root: Path) -> list:
    """Find all audio files."""
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def main():
    parser = argparse.ArgumentParser(
        description="Voicemail greeting end detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --file test_samples/vm1_output.wav
  python main.py --dir test_samples --out results
        """
    )
    parser.add_argument("--file", type=str, help="Single audio file")
    parser.add_argument("--dir", type=str, help="Directory of audio files")
    parser.add_argument("--out", type=str, default=None, help="Output directory for JSON/CSV")
    parser.add_argument("--chunk-ms", type=int, default=250, help="Chunk size (ms)")
    parser.add_argument("--silence-db", type=float, default=-40.0, help="Silence threshold (dB)")
    parser.add_argument("--silence-long", type=float, default=1.5, help="Silence timeout (s)")

    args = parser.parse_args()

    if not args.file and not args.dir:
        parser.print_help()
        return

    print("\n🚀 Voicemail Greeting End Detector")
    print("=" * 70)

    try:
        analyzer = VoicemailAnalyzer(
            chunk_size_ms=args.chunk_ms,
            silence_threshold_db=args.silence_db,
            silence_long_sec=args.silence_long,
        )
    except Exception as e:
        print(f"❌ Failed to create analyzer: {e}")
        sys.exit(1)

    aggregator = ResultsAggregator()

    # Single file
    if args.file:
        path = Path(args.file)
        if not path.is_file():
            print(f"❌ File not found: {path}")
            return
        v = analyze_file(path, analyzer)
        if v:
            aggregator.add_decision(analyzer.to_call_decision(str(path), v))

    # Directory
    if args.dir:
        root = Path(args.dir)
        if not root.is_dir():
            print(f"❌ Directory not found: {root}")
            return
        files = find_audio_files(root)
        if not files:
            print(f"❌ No audio files found in {root}")
            print("   Supported: .wav, .mp3, .flac, .ogg, .m4a")
        else:
            print(f"📁 Found {len(files)} file(s)\n")
            for f in files:
                v = analyze_file(f, analyzer)
                if v:
                    aggregator.add_decision(analyzer.to_call_decision(str(f), v))

    # Save results
    if args.out and aggregator.decisions:
        out_dir = Path(args.out)
        aggregator.save_json(out_dir / "results.json")
        aggregator.save_csv(out_dir / "results.csv")
        print(f"\n💾 Results saved to {out_dir}/")

    print("\n✅ Complete!\n")


if __name__ == "__main__":
    main()
