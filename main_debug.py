#!/usr/bin/env python3
import sys
sys.path.insert(0, './src')
from state_machine import VoicemailAnalyzer
from pathlib import Path
import numpy as np

analyzer = VoicemailAnalyzer(chunk_size_ms=250)

def debug_analyze(file_path):
    print(f"\n🔍 DEBUGGING {file_path}")
    v = analyzer.analyze_file(file_path)
    print(f"Decision: {v.decision_time_sec:.2f}s, Beep: {v.beep_detected}, Reason: {v.reason}")
    
    # Force beep detector debug
    streamer = analyzer.streamer
    streamer.load_file(file_path)
    beep_detector = analyzer.beep
    
    max_peak_db = -120
    max_freq = 0
    max_chunk = 0
    
    for i, chunk in enumerate(streamer.stream()):
        ba = beep_detector.process_chunk(chunk)
        if ba.peak_magnitude_db > max_peak_db:
            max_peak_db = ba.peak_magnitude_db
            max_freq = ba.peak_frequency
            max_chunk = i * 0.25  # 250ms chunks
        
        if ba.beep_detected:
            print(f"  CHUNK {i} ({i*0.25:.2f}s): BEEP! freq={ba.peak_frequency:.0f}Hz, mag={ba.peak_magnitude_db:.1f}dB, purity={ba.spectral_purity_db:.1f}dB")
    
    print(f"MAX PEAK: {max_peak_db:.1f}dB @ {max_freq:.0f}Hz (chunk {max_chunk:.2f}s)")
    print("---")

debug_analyze("test_samples/vm4_output.wav")
