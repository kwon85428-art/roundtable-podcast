#!/usr/bin/env python3
"""Batch generate all podcast segments with CosyVoice cloned voices."""

import os
import sys
import re
import json
import time
import torch
import torchaudio

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(PROJECT_DIR, 'CosyVoice'))
sys.path.insert(0, os.path.join(PROJECT_DIR, 'CosyVoice', 'third_party', 'Matcha-TTS'))

from cosyvoice.cli.cosyvoice import AutoModel

# ── Config ──
MODEL_DIR = os.path.join(PROJECT_DIR, 'pretrained_models', 'CosyVoice-300M-SFT')
SCRIPT_PATH = os.path.join(PROJECT_DIR, 'script.md')
AUDIO_OUT = os.path.join(PROJECT_DIR, 'audio_cloned')
VOICE_DIR = os.path.join(PROJECT_DIR, 'voice_samples')

SPEAKER_CFG = {
    "戴金星": {"wav": "dai_jinxing_25s.wav", "text": "这个问题问得好。我讲一个判断，威远气田的发现，有偶然性，但偶然背后有必然。", "id": "dai_jinxing"},
    "邹才能": {"wav": "zou_caineng_25s.wav", "text": "我基本同意戴老师的判断，但我想换一个角度来补充。", "id": "zou_caineng"},
    "马永生": {"wav": "ma_yongsheng_25s.wav", "text": "两位老师讲得都很好，我想从一个更宏观的视角来补充。", "id": "ma_yongsheng"},
}
HOST_SPK = "中文男"


def parse_full_script(path):
    """Parse script.md → list of (speaker, full_text)."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    segments = []
    current_speaker = None
    current_lines = []

    for line in text.split("\n"):
        # Skip markdown headers, tables, code blocks
        if re.match(r"^##\s", line) or re.match(r"^###\s", line):
            continue
        if line.strip().startswith("|") or line.strip().startswith("```"):
            continue
        if line.strip() == "---":
            continue

        # Speaker marker
        m = re.match(r"^\*\*(主持人|戴金星|邹才能|马永生)\*\*[：:]", line)
        if m:
            if current_speaker and current_lines:
                segments.append((current_speaker, "\n".join(current_lines).strip()))
            current_speaker = m.group(1)
            rest = line[m.end():].strip()
            current_lines = [rest] if rest else []
            continue

        if current_speaker and line.strip():
            current_lines.append(line.strip())

    if current_speaker and current_lines:
        segments.append((current_speaker, "\n".join(current_lines).strip()))

    return segments


def clean_for_tts(text):
    """Strip markdown formatting for TTS."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main():
    os.makedirs(AUDIO_OUT, exist_ok=True)

    # Parse script
    print("Parsing script...")
    segments = parse_full_script(SCRIPT_PATH)
    print(f"  {len(segments)} dialogue segments\n")

    # Load model
    device = "cpu"  # MPS not supported
    print(f"Loading model from {MODEL_DIR} on {device}...")
    cosyvoice = AutoModel(model_dir=MODEL_DIR)
    print(f"  Sample rate: {cosyvoice.sample_rate}")

    # Register cloned speakers
    print("\nRegistering speakers...")
    for name, cfg in SPEAKER_CFG.items():
        wav_path = os.path.join(VOICE_DIR, cfg["wav"])
        ok = cosyvoice.add_zero_shot_spk(cfg["text"], wav_path, cfg["id"])
        print(f"  {name}: {'✓' if ok else '✗ FAILED'}")
    cosyvoice.save_spkinfo()

    # List available SFT speakers
    sft_spks = cosyvoice.list_available_spks()
    print(f"  SFT speakers: {sft_spks}")
    if HOST_SPK not in sft_spks:
        print(f"  ⚠ '{HOST_SPK}' not in SFT speakers! Available: {sft_spks}")
        host_spk = sft_spks[0] if sft_spks else HOST_SPK
        print(f"  → Using '{host_spk}' instead")
    else:
        host_spk = HOST_SPK

    # Generate each segment
    print(f"\n{'='*60}")
    print(f"Generating {len(segments)} segments...")
    print(f"{'='*60}\n")

    manifest = []
    total_wall = 0

    for i, (speaker, raw_text) in enumerate(segments):
        text = clean_for_tts(raw_text)
        filename = f"{i:04d}_{SPEAKER_CFG.get(speaker, {}).get('id', 'host')}.wav"
        out_path = os.path.join(AUDIO_OUT, filename)

        # Skip if already generated
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            dur = os.path.getsize(out_path) / (cosyvoice.sample_rate * 2)
            print(f"[{i:03d}/{len(segments)}] {speaker} ({len(text)} chars) — SKIP (already exists, {dur:.0f}s)\n")
            manifest.append({
                "index": i, "speaker": speaker, "file": filename,
                "duration_s": round(dur, 2), "wall_s": 0,
                "text_chars": len(text), "cached": True,
            })
            continue

        print(f"[{i:03d}/{len(segments)}] {speaker} ({len(text)} chars)")
        preview = text[:100] + "..." if len(text) > 100 else text
        print(f"  「{preview}」")

        t0 = time.time()
        try:
            if speaker in SPEAKER_CFG:
                spk_id = SPEAKER_CFG[speaker]["id"]
                gen = cosyvoice.inference_zero_shot(text, "", "", zero_shot_spk_id=spk_id, stream=False)
            else:
                gen = cosyvoice.inference_sft(text, host_spk, stream=False)

            # Collect ALL speech chunks (model may split long text)
            chunks = []
            for result in gen:
                chunks.append(result["tts_speech"])
            if len(chunks) == 1:
                speech = chunks[0]
            else:
                speech = torch.cat(chunks, dim=1)
            torchaudio.save(out_path, speech, cosyvoice.sample_rate)

            elapsed = time.time() - t0
            total_wall += elapsed
            dur = speech.shape[1] / cosyvoice.sample_rate
            rtf = elapsed / dur if dur > 0 else 0

            print(f"  → {filename} ({dur:.1f}s audio, {elapsed:.1f}s wall, RTF={rtf:.1f}x)\n")

            manifest.append({
                "index": i, "speaker": speaker, "file": filename,
                "duration_s": round(dur, 2), "wall_s": round(elapsed, 1),
                "text_chars": len(text),
            })

        except Exception as e:
            print(f"  ✗ FAILED: {e}\n")
            manifest.append({
                "index": i, "speaker": speaker, "file": None,
                "error": str(e), "text_chars": len(text),
            })

    # Save manifest
    manifest_path = os.path.join(PROJECT_DIR, "manifest_cloned.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Summary
    n_ok = sum(1 for m in manifest if m.get("file"))
    total_dur = sum(m.get("duration_s", 0) for m in manifest)
    print(f"\n{'='*60}")
    print(f"COMPLETE: {n_ok}/{len(manifest)} segments")
    print(f"  Total audio: {total_dur:.0f}s ({total_dur/60:.1f} min)")
    print(f"  Total wall time: {total_wall:.0f}s ({total_wall/60:.1f} min)")
    print(f"  Manifest: {manifest_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
