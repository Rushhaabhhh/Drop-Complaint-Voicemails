# Drop Compliant Voicemail System

## Project Summary
Drop Compliant Voicemail System automatically detects voicemail greeting endings for TCPA-compliant message drops. Uses multi-signal analysis: Deepgram STT recognizes "leave a message" phrases, FFT detects 745-880Hz beeps, and silence monitoring confirms quiet periods. Processes 100ms audio chunks with priority logic (beep > speech + silence > extended silence > 20s timeout). Delivers precise drop timestamps, confidence scores, and audit-ready JSON reports for telemarketing and lead automation.

## 📱 Demo Video : https://www.loom.com/share/196d5e66202343adb4c09ffbf2e6ff82

## 🚀 Quick Start

```bash
# Clone repo
git clone https://github.com/Rushhaabhhh/Drop-Complaint-Voicemails.git
cd Drop-Complaint-Voicemails

# Install dependencies
pip install -r requirements.txt

# Setup environment (optional but recommended)
cp .env.example .env
# Add your API keys to .env:
# DEEPGRAM_API_KEY=your_deepgram_key

# Process test samples
python main.py --audio-dir test_samples

# Process your own audio files
python main.py --audio-dir your_audio_folder_name
```

## 🎯 Features

- ✅ **Multi-Signal Detection** - Beep (FFT), Speech (Deepgram STT), Silence  
- ✅ **Real-time Processing** - 100ms audio chunks  
- ✅ **High Accuracy** - 0.76-0.95 confidence on primary signals  
- ✅ **TCPA Compliant** - Respects standard voicemail timing  
- ✅ **No API Fallback** - Works with just beep + silence detection  
- ✅ **Batch Processing** - Multiple files with summary reports  
- ✅ **Detailed Logging** - Confidence scores, timestamps, reasoning  

## 🏗️ Architecture

```
Audio Stream (100ms chunks)
        ↓
 ┌─────────────────────┐
 │   Signal Processors │
 ├─────────────────────┤
 │ • FFT Beep Detector │───> 745-880Hz → Drop +200ms
 │ • Deepgram STT      │───> LLM Greeting Analysis
 │ • Silence Monitor   │───> 1.5s+ silence after speech
 └─────────────────────┘
        ↓
 ┌─────────────────────┐
 │  Decision Engine    │
 │  Priority:          │
 │  Beep > Speech >    │
 │  Silence > Timeout  │
 │                     │
 └─────────────────────┘
        ↓
   Drop Timestamp + Confidence
```

## 🎮 Example Output

```
================================================================================
🎙️  Enhanced Voicemail Drop Compliance System
================================================================================

✅ Deepgram API configured - full transcription enabled

Found 7 file(s) to process

[1/7]
======================================================================
📞 Processing: test_samples/vm1_output.wav
======================================================================
  🔇 Silence: 1.1s                              
  ✅ Greeting end: 4.80s ("leave your message", conf: 0.95)
  🔔 Beep: 10.40s (745Hz, conf: 0.80)                    
  🎯 DROP at 10.60s                               
     Method: BEEP
     Confidence: 0.80
     Reasoning: Beep detected at 10.40s (conf: 0.80)

[...6 more files...]

================================================================================
📊 SUMMARY TABLE
================================================================================

File                           Drop Time    Method       Confidence  
------------------------------ ------------ ------------ ------------
vm1_output.wav                 10.60s       BEEP         0.80        
vm2_output.wav                 12.10s       SILENCE      0.60        
vm3_output.wav                 12.70s       SILENCE      0.60        
vm4_output.wav                 6.50s        SILENCE      0.72        
vm5_output.wav                 17.50s       SILENCE      0.60        
vm6_output.wav                 4.90s        SPEECH       0.76        
vm7_output.wav                 11.90s       BEEP         0.90        
================================================================================
```

## 🧠 Detection Logic

### Priority Hierarchy (50% Beep | 35% Speech | 15% Silence)

```
1. BEEP DETECTED (conf > 0.75)
   → Drop at beep_timestamp + 200ms
   → Examples: vm1 (10.60s, 745Hz), vm7 (11.90s, 880Hz)

2. GREETING END + 1.5s SILENCE (conf > 0.70)
   → Phrases: "leave your message", "thank you", etc.
   → Example: vm6 (4.90s, "leave a message" @ 0.95)

3. EXTENDED SILENCE (>3.0s)
   → Conservative fallback
   → Examples: vm2, vm3, vm4, vm5

4. TIMEOUT (20s) - Safety net
```

## 📊 Results Format

### JSON Output (`results/timestamps.json`)
```json
{
  "vm1_output.wav": {
    "drop_timestamp": 10.60,
    "method": "BEEP",
    "confidence": 0.80,
    "reasoning": "Beep detected at 10.40s (conf: 0.80)",
    "detected_signals": {
      "beep": {"detected": true, "timestamp": 10.40, "frequency": 745, "confidence": 0.80},
      "speech": {"greeting_ended": true, "timestamp": 4.80, "phrase": "leave your message", "confidence": 0.95}
    }
  }
}
```

## 🛠️ Tech Stack

- **Audio Processing**: `pydub`, `librosa` (FFT analysis)
- **Speech-to-Text**: Deepgram (primary)
- **LLM Analysis**: OpenAI GPT for greeting detection
- **Real-time**: 100ms audio chunk streaming
- **Output Formats**: JSON, CSV, rich console tables

## 🧪 Test Data

**7 diverse voicemail samples** in `test_samples/`:
- Standard beeps (745Hz, 880Hz)
- Short greetings (3s) to long greetings (13s)
- Various phrasing styles
- Silence-only scenarios
