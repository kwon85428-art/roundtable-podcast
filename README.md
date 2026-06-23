---
name: roundtable-podcast
description: Turn energy-expert-roundtable scripts into voice-cloned podcasts using CosyVoice zero-shot voice cloning. Use when the user wants to produce an audio podcast from a roundtable discussion script with cloned scholar voices.
---

# Roundtable Podcast Skill

Turn an energy-expert-roundtable discussion script into a podcast MP3 with cloned scholar voices using CosyVoice zero-shot voice cloning.

## What This Skill Does

1. Takes a roundtable script (Markdown format with `**Speaker**：` markers)
2. Takes 1-3 reference audio samples per scholar (≥10s each, ≤30s for CosyVoice limit)
3. Uses CosyVoice-300M-SFT for host voice (built-in `中文男`) + CosyVoice zero-shot cloning for scholars
4. Generates all dialogue segments as individual WAV files
5. Concatenates with appropriate silence gaps into a single podcast MP3

## Prerequisites

- Python 3.9+ with venv
- ~12GB disk for CosyVoice models (300M + 300M-SFT)
- macOS/Linux with ffmpeg installed
- CPU inference: ~15-20x real-time factor (20 min podcast ≈ 5 hours on CPU)
- NVIDIA GPU (optional): ~1-2x RTF with CUDA

## Setup (one-time)

```bash
# 1. Create venv and install
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchaudio cosyvoice modelscope
pip install lightning pyarrow onnx tensorboard protobuf soundfile

# 2. Clone CosyVoice with Matcha-TTS submodule
git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice
cd CosyVoice && git submodule update --init --depth 1 third_party/Matcha-TTS && cd ..

# 3. Download models (run once)
python -c "
import os, modelscope.hub.api as api
os.environ['MODELSCOPE_CACHE'] = './.modelscope_cache'
os.makedirs('./.modelscope_cache/credentials', exist_ok=True)
api.ModelScopeConfig.path_credential = './.modelscope_cache/credentials'
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
"
```

## Script Format

The roundtable script uses Markdown with bold speaker markers:

```markdown
**主持人**：
各位听众，欢迎收听今天的能源圆桌。

**戴金星**：
这个问题问得好。我讲一个判断——威远气田的发现，有偶然性，但偶然背后有必然。

**邹才能**：
我基本同意戴老师的判断，但我想换一个角度来补充。
```

Rules:
- Speaker names must match config: `主持人`, `戴金星`, `邹才能`, `马永生` (customizable)
- Use Chinese colon `：` or ASCII `:`
- Blank lines between paragraphs are OK (skipped)
- Act headers (`## 第X幕`) are skipped
- Table rows and code blocks are skipped

## Voice Sample Requirements

- Format: 16kHz mono WAV (use ffmpeg to convert)
- Duration: 10–30 seconds per scholar (CosyVoice max 30s for zero-shot)
- Quality: Clean speech, minimal background noise, representative vocal character
- Prompt text: A short sentence matching a segment of the audio

```bash
# Convert MP3 to CosyVoice-compatible WAV
ffmpeg -i scholar_audio.mp3 -ac 1 -ar 16000 -sample_fmt s16 scholar_name.wav

# Trim to 25s for zero-shot cloning
ffmpeg -i scholar_audio.wav -ss 5 -t 25 scholar_name_25s.wav
```

## Workflow

### Step 1: Prepare voice samples

Place scholar voice samples in `voice_samples/`:
```
voice_samples/
  dai_jinxing_25s.wav
  zou_caineng_25s.wav
  ma_yongsheng_25s.wav
```

### Step 2: Configure speakers

Edit `SPEAKER_CFG` in `batch_generate.py`:

```python
SPEAKER_CFG = {
    "戴金星": {"wav": "dai_jinxing_25s.wav", "text": "匹配音频的文本。", "id": "dai_jinxing"},
    "邹才能": {"wav": "zou_caineng_25s.wav", "text": "匹配音频的文本。", "id": "zou_caineng"},
    "马永生": {"wav": "ma_yongsheng_25s.wav", "text": "匹配音频的文本。", "id": "ma_yongsheng"},
}
HOST_SPK = "中文男"  # from CosyVoice SFT built-in speakers
```

### Step 3: Generate TTS segments

```bash
source .venv/bin/activate
python batch_generate.py
```

Outputs: `audio_cloned/0000_host.wav`, `0001_host.wav`, `0002_dai_jinxing.wav`, ...
Progress: `manifest_cloned.json`

### Step 4: Concatenate into podcast

```bash
python concat_cloned.py
```

Outputs: `weiyuan-podcast-cloned.mp3` with ID3 metadata

### Optional Step 5: Validate with quick test

```bash
python generate_cloned.py test
# Generates audio_cloned/test_host.wav, test_dai.wav, test_zou.wav, test_ma.wav
```

## File Architecture

```
podcasts/<topic>/
  script.md              # Roundtable script (Markdown)
  voice_samples/          # Scholar reference audio (16kHz mono WAV)
  pretrained_models/      # CosyVoice models (auto-downloaded)
  audio_cloned/           # Generated TTS segments (WAV)
  batch_generate.py       # Main TTS generation script
  concat_cloned.py        # FFmpeg concatenation script
  generate_cloned.py      # Quick test script
  manifest_cloned.json    # Segment manifest (auto-generated)
  weiyuan-podcast-cloned.mp3  # Final podcast output
```

## Customization

### Adding a new scholar

1. Add voice sample to `voice_samples/`
2. Add entry to `SPEAKER_CFG` in `batch_generate.py`
3. Ensure the speaker name matches the script marker (`**新学者**：`)

### Changing voice sample duration

The 25s trim is a safe default. The CosyVoice limit is 30s. Longer original audio (>30s) must be trimmed. Choose a clip with clean, representative speech.

### Using GPU acceleration

Change the device selection in `batch_generate.py`:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### Adjusting silence gaps

Edit `concat_cloned.py`:
```python
GAP_SS = 0.4   # same speaker gap (seconds)
GAP_NS = 0.7   # new speaker gap
GAP_ACT = 1.5  # act/scene break
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `prompt wav not found` | Missing voice sample | Run ffmpeg conversion step |
| `audio longer than 30s` | Reference audio too long | Trim to ≤30s with ffmpeg |
| `KeyError: '中文男'` | Wrong model loaded | Use CosyVoice-300M-SFT, not 300M |
| `ModuleNotFoundError: matcha` | Matcha-TTS submodule missing | `git submodule update --init` |
| `ModuleNotFoundError: lightning` | Missing dependency | `pip install lightning` |
| Segment too short / truncated | CosyVoice chunks long text | Ensure `batch_generate.py` loops over generator (not just `next()`) |
| MPS not supported | Apple Silicon GPU | Falls back to CPU automatically |
| `Operation not permitted: .modelscope` | macOS sandbox | Set `MODELSCOPE_CACHE` + patch `path_credential` |

## Integration with energy-expert-roundtable

This skill is the audio production companion to `energy-expert-roundtable`. The workflow:

1. **energy-expert-roundtable** → produces structured discussion script
2. **roundtable-podcast** (this skill) → clones scholar voices → generates MP3

Together they form an end-to-end pipeline: roundtable topic → structured debate → voice-cloned podcast.
