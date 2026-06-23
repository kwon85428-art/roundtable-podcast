#!/usr/bin/env python3
"""CosyVoice zero-shot voice cloning — generate podcast TTS with cloned scholar voices."""

import os
import sys
import json
import time
import torch
import torchaudio

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add CosyVoice + Matcha-TTS to path (before any imports)
_COSYVOICE_DIR = os.path.join(PROJECT_DIR, 'CosyVoice')
_MATCHA_DIR = os.path.join(_COSYVOICE_DIR, 'third_party', 'Matcha-TTS')
sys.path.insert(0, _COSYVOICE_DIR)
sys.path.insert(0, _MATCHA_DIR)

from cosyvoice.cli.cosyvoice import AutoModel

# ── Config ──────────────────────────────────────────────
MODEL_DIR = os.path.join(PROJECT_DIR, 'pretrained_models', 'CosyVoice-300M-SFT')
VOICE_SAMPLES_DIR = os.path.join(PROJECT_DIR, 'voice_samples')
AUDIO_OUT_DIR = os.path.join(PROJECT_DIR, 'audio_cloned')
MANIFEST_PATH = os.path.join(PROJECT_DIR, 'manifest.json')

# Voice cloning config — each scholar's reference audio
SPEAKER_CONFIG = {
    "戴金星": {
        "prompt_wav": os.path.join(VOICE_SAMPLES_DIR, "dai_jinxing_25s.wav"),
        "prompt_text": "这个问题问得好。我讲一个判断，威远气田的发现，有偶然性，但偶然背后有必然。",
        "spk_id": "dai_jinxing",
    },
    "邹才能": {
        "prompt_wav": os.path.join(VOICE_SAMPLES_DIR, "zou_caineng_25s.wav"),
        "prompt_text": "我基本同意戴老师的判断，但我想换一个角度来补充。",
        "spk_id": "zou_caineng",
    },
    "马永生": {
        "prompt_wav": os.path.join(VOICE_SAMPLES_DIR, "ma_yongsheng_25s.wav"),
        "prompt_text": "两位老师讲得都很好，我想从一个更宏观的视角来补充。",
        "spk_id": "ma_yongsheng",
    },
}

# The host will use the built-in CosyVoice speaker
HOST_SPK = "中文男"


def load_model():
    """Load CosyVoice model."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading CosyVoice model from {MODEL_DIR} on {device}...")
    cosyvoice = AutoModel(model_dir=MODEL_DIR)
    # Move to device
    if device == "mps":
        try:
            cosyvoice.model.to(device)
        except Exception:
            print("  MPS not supported, falling back to CPU")
            device = "cpu"
    print(f"  Device: {device}, Sample rate: {cosyvoice.sample_rate}")
    return cosyvoice, device


def register_speakers(cosyvoice):
    """Register zero-shot speakers from reference audio."""
    print("\nRegistering zero-shot speakers...")
    for name, cfg in SPEAKER_CONFIG.items():
        prompt_wav = cfg["prompt_wav"]
        prompt_text = cfg["prompt_text"]
        spk_id = cfg["spk_id"]

        if not os.path.exists(prompt_wav):
            print(f"  ⚠ {name}: prompt wav not found at {prompt_wav}")
            continue

        print(f"  {name} ({spk_id}): {prompt_wav}")
        success = cosyvoice.add_zero_shot_spk(prompt_text, prompt_wav, spk_id)
        if success:
            print(f"    ✓ registered")
        else:
            print(f"    ✗ FAILED")

    # Save spkinfo for potential reuse
    cosyvoice.save_spkinfo()
    print("  Speakers saved.")


def generate_segment(cosyvoice, speaker, text, index):
    """Generate TTS for one dialogue segment."""
    if speaker in SPEAKER_CONFIG:
        spk_id = SPEAKER_CONFIG[speaker]["spk_id"]
        generator = cosyvoice.inference_zero_shot(
            text, "", "", zero_shot_spk_id=spk_id, stream=False
        )
    else:
        # Host — use SFT speaker
        generator = cosyvoice.inference_sft(text, HOST_SPK, stream=False)

    # Collect all chunks (model may split long text)
    chunks = []
    for result in generator:
        chunks.append(result["tts_speech"])
    if len(chunks) == 1:
        return chunks[0]
    else:
        return torch.cat(chunks, dim=1)


def main():
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)

    # Load manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Load model
    cosyvoice, device = load_model()

    # Register cloned speakers
    register_speakers(cosyvoice)

    # Generate audio for each segment
    print(f"\nGenerating {len(manifest)} segments...")
    new_manifest = []

    for seg in manifest:
        idx = seg["index"]
        speaker = seg["speaker"]
        text = seg["text_preview"]  # this is truncated — need full text

        # Get full text from original manifest data
        # We stored text_preview but need full text
        # Actually the text_preview is only 60 chars. We need the full text.
        # Let's use text_length and regenerate properly.
        # For now, we'll parse the segments from the script again.

    print("\n⚠ The manifest only stores truncated text. Please use generate_full.py instead.")
    print("This script validates the model and speaker registration works.")


def test_single():
    """Quick test: generate one line per speaker."""
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
    cosyvoice, device = load_model()
    register_speakers(cosyvoice)

    test_lines = {
        "host": ("主持人", "各位听众朋友，欢迎收听今天的能源圆桌。我是张宏。"),
        "dai": ("戴金星", "威远气田的发现有偶然性，但偶然背后有必然。"),
        "zou": ("邹才能", "如果储层再好的白云岩底下没有烃源岩，气从哪里来？"),
        "ma": ("马永生", "威远给我最大的启示是，重大勘探突破发生在认识转变的时刻。"),
    }

    for label, (speaker, text) in test_lines.items():
        print(f"\n[{label}] {speaker}: {text}")
        t0 = time.time()
        speech = generate_segment(cosyvoice, speaker, text, 0)
        elapsed = time.time() - t0

        out_path = os.path.join(AUDIO_OUT_DIR, f"test_{label}.wav")
        torchaudio.save(out_path, speech, cosyvoice.sample_rate)
        dur = speech.shape[1] / cosyvoice.sample_rate
        print(f"  → {out_path} ({dur:.1f}s, {elapsed:.1f}s wall)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_single()
    else:
        main()
