#!/usr/bin/env python3
"""Concatenate cloned audio segments into final podcast MP3."""

import os
import json
import subprocess

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(PROJECT_DIR, "audio_cloned")
MANIFEST = os.path.join(PROJECT_DIR, "manifest_cloned.json")

GAP_SS = 0.4   # same speaker gap
GAP_NS = 0.7   # new speaker gap
GAP_ACT = 1.5  # act/scene break
SILENCE_INTRO = 0.5

def gen_silence(dur, path):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=22050:cl=mono",
         "-t", str(dur), "-q:a", "9", "-acodec", "libmp3lame", path],
        capture_output=True,
    )

def main():
    with open(MANIFEST, "r") as f:
        manifest = json.load(f)

    # Generate silence files
    silence_dir = os.path.join(AUDIO_DIR, "_silence")
    os.makedirs(silence_dir, exist_ok=True)
    s04 = os.path.join(silence_dir, "s04.mp3")
    s07 = os.path.join(silence_dir, "s07.mp3")
    s15 = os.path.join(silence_dir, "s15.mp3")
    s05 = os.path.join(silence_dir, "s05.mp3")
    gen_silence(GAP_SS, s04)
    gen_silence(GAP_NS, s07)
    gen_silence(GAP_ACT, s15)
    gen_silence(SILENCE_INTRO, s05)

    # Build concat list
    concat_file = os.path.join(PROJECT_DIR, "concat_cloned.txt")
    prev_speaker = None
    prev_act = ""
    act_boundaries = {0, 3, 7, 11, 17}  # approximate act starts

    with open(concat_file, "w") as f:
        f.write(f"file '{s05}'\n")

        for seg in manifest:
            if seg.get("error") or not seg.get("file"):
                continue

            audio_path = os.path.join(AUDIO_DIR, seg["file"])
            if not os.path.exists(audio_path):
                continue

            idx = seg["index"]
            speaker = seg["speaker"]

            if idx in act_boundaries and idx > 0:
                gap = s15
            elif speaker == prev_speaker:
                gap = s04
            else:
                gap = s07

            if idx > 0:
                f.write(f"file '{gap}'\n")
            f.write(f"file '{audio_path}'\n")

            prev_speaker = speaker

        f.write(f"file '{s05}'\n")

    # Concat
    output = os.path.join(PROJECT_DIR, "weiyuan-podcast-cloned.mp3")
    print(f"Concatenating → {output} ...")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
         "-acodec", "libmp3lame", "-b:a", "128k", "-ar", "22050",
         "-metadata", "title=威远气田发现机理 — 三位学者圆桌对话（声音克隆版）",
         "-metadata", "artist=戴金星、邹才能、马永生 (CosyVoice 克隆)",
         "-metadata", "album=能源圆桌",
         output],
        check=True,
    )

    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", output],
        capture_output=True, text=True,
    ).stdout.strip())

    size = os.path.getsize(output)
    print(f"Done! {output}")
    print(f"  Duration: {dur:.0f}s ({dur/60:.1f} min)")
    print(f"  Size: {size/1024:.0f} KB ({size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
