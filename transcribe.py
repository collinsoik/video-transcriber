#!/usr/bin/env python3
"""
Video Transcriber - Extract transcripts from video files using Fireworks AI Whisper-v3.

Usage:
    export FIREWORKS_API_KEY="your-key"
    python transcribe.py video.mp4
    python transcribe.py video.mp4 --output transcript.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import requests

FIREWORKS_WHISPER_URL = "https://audio-prod.api.fireworks.ai/v1/audio/transcriptions"
FIREWORKS_MODEL = "whisper-v3"


def format_timestamp(seconds: float) -> str:
    """Convert seconds to human-readable format like '12m34s'."""
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


def check_ffmpeg():
    """Verify FFmpeg is installed and accessible."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg not found.", file=sys.stderr)
        print("Install it:", file=sys.stderr)
        print("  macOS:   brew install ffmpeg", file=sys.stderr)
        print("  Ubuntu:  sudo apt install ffmpeg", file=sys.stderr)
        print("  Windows: choco install ffmpeg", file=sys.stderr)
        sys.exit(1)


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """Extract audio from video as 16kHz mono WAV (optimal for Whisper)."""
    print(f"Extracting audio from {video_path.name}...")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                  # Discard video stream
        "-acodec", "pcm_s16le", # 16-bit PCM
        "-ar", "16000",         # 16kHz sample rate
        "-ac", "1",             # Mono
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Audio extracted: {output_path.name} ({size_mb:.1f} MB)")
    return output_path


def transcribe(audio_path: Path, api_key: str) -> dict:
    """Send audio to Fireworks AI Whisper-v3 API and return structured transcript."""
    print(f"Transcribing {audio_path.name} via Fireworks AI Whisper-v3...")
    print("(This may take several minutes for long recordings.)")

    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    timeout = max(120, int(file_size_mb / 10 * 60) + 60)

    with open(audio_path, "rb") as f:
        response = requests.post(
            FIREWORKS_WHISPER_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (audio_path.name, f, "audio/wav")},
            data={
                "model": FIREWORKS_MODEL,
                "temperature": "0",
                "vad_model": "silero",
                "response_format": "verbose_json",
            },
            timeout=timeout,
        )

    if response.status_code != 200:
        raise Exception(
            f"Fireworks API error {response.status_code}: {response.text}"
        )

    result = response.json()

    transcript = {
        "audio_file": audio_path.name,
        "language": result.get("language", "en"),
        "duration": result.get("duration", 0),
        "segments": [],
    }

    for segment in result.get("segments", []):
        seg_data = {
            "start": segment.get("start", 0),
            "end": segment.get("end", 0),
            "text": segment.get("text", "").strip(),
        }

        if "words" in segment:
            seg_data["words"] = [
                {
                    "word": w.get("word", ""),
                    "start": w.get("start", 0),
                    "end": w.get("end", 0),
                }
                for w in segment["words"]
            ]

        transcript["segments"].append(seg_data)

    print(f"Transcription complete: {len(transcript['segments'])} segments, "
          f"{format_timestamp(transcript['duration'])} duration")

    return transcript


def print_transcript(transcript: dict):
    """Print a clean, human-readable transcript to stdout."""
    duration = transcript.get("duration", 0)
    lang = transcript.get("language", "unknown")

    print(f"\n{'=' * 60}")
    print(f"TRANSCRIPT")
    print(f"Duration: {format_timestamp(duration)}  |  Language: {lang}")
    print(f"Segments: {len(transcript['segments'])}")
    print(f"{'=' * 60}\n")

    for segment in transcript["segments"]:
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"]
        print(f"[{start} -> {end}]  {text}")

    print(f"\n{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe video files using Fireworks AI Whisper-v3",
        epilog="Requires: FFmpeg installed, FIREWORKS_API_KEY env variable set.",
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to the video file to transcribe",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for JSON transcript (default: <video_name>_transcript.json)",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the extracted WAV file instead of deleting it",
    )

    args = parser.parse_args()

    # Validate video file
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    # Validate API key
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        print("Error: FIREWORKS_API_KEY environment variable not set.", file=sys.stderr)
        print("Get a key at https://fireworks.ai and run:", file=sys.stderr)
        print('  export FIREWORKS_API_KEY="your-key-here"', file=sys.stderr)
        sys.exit(1)

    # Check ffmpeg
    check_ffmpeg()

    # Determine output path
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = video_path.parent / f"{video_path.stem}_transcript.json"

    # Extract audio
    if args.keep_audio:
        audio_path = video_path.parent / f"{video_path.stem}.wav"
        extract_audio(video_path, audio_path)
    else:
        tmp_dir = tempfile.mkdtemp()
        audio_path = Path(tmp_dir) / f"{video_path.stem}.wav"
        extract_audio(video_path, audio_path)

    try:
        # Transcribe
        transcript = transcribe(audio_path, api_key)

        # Save JSON
        with open(output_path, "w") as f:
            json.dump(transcript, f, indent=2)
        print(f"\nTranscript saved to: {output_path}")

        # Print human-readable transcript
        print_transcript(transcript)

    finally:
        # Clean up temp audio if not keeping it
        if not args.keep_audio and audio_path.exists():
            audio_path.unlink()
            audio_path.parent.rmdir()


if __name__ == "__main__":
    main()
