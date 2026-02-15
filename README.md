# Video Transcriber

A minimal command-line tool that extracts transcripts from video files using
[Fireworks AI's Whisper-v3 API](https://fireworks.ai).

## How It Works

1. Extracts the audio track from your video using FFmpeg
2. Sends the audio to Fireworks AI's hosted Whisper-v3 model
3. Saves a timestamped transcript as JSON
4. Prints a clean text transcript to your terminal

## Prerequisites

- **Python 3.8+**
- **FFmpeg** installed and on your PATH
- **Fireworks AI API key** ([fireworks.ai](https://fireworks.ai))

### Install FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows
choco install ffmpeg
```

## Setup

```bash
git clone https://github.com/collinsoik/video-transcriber.git
cd video-transcriber
pip install -r requirements.txt
export FIREWORKS_API_KEY="your-api-key-here"
```

## Usage

```bash
# Basic usage - saves transcript JSON next to the video
python transcribe.py path/to/lecture.mp4

# Specify output path
python transcribe.py lecture.mp4 -o my_transcript.json

# Keep the extracted audio file
python transcribe.py lecture.mp4 --keep-audio
```

## Output

The tool produces two outputs:

**1. JSON file** (saved to disk) with segment-level timestamps:

```json
{
  "audio_file": "lecture.wav",
  "language": "en",
  "duration": 4738.45,
  "segments": [
    {
      "start": 2.52,
      "end": 25.26,
      "text": "Welcome to today's lecture..."
    }
  ]
}
```

**2. Text transcript** (printed to stdout):

```
============================================================
TRANSCRIPT
Duration: 1h18m58s  |  Language: en
Segments: 412
============================================================

[0m02s -> 0m25s]  Welcome to today's lecture...
[0m30s -> 0m59s]  Today we will cover...

============================================================
```

## CLI Options

| Flag | Description |
|------|-------------|
| `video` | Path to the video file (required) |
| `-o, --output` | Custom output path for the JSON transcript |
| `--keep-audio` | Keep the extracted WAV file (deleted by default) |
