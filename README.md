# 🎹 Nasum Transcriptor

Extract and transcribe piano parts from YouTube music videos into sheet music.

![Nasum Transcriptor](https://img.shields.io/badge/Piano-Transcription-d4a574?style=for-the-badge)

## Features

- 🎬 **YouTube Integration** - Paste any YouTube link with music
- 🎵 **AI Source Separation** - Isolates piano from other instruments using Demucs
- 🎼 **Audio-to-MIDI Transcription** - Converts audio to musical notation using basic-pitch
- 📄 **Sheet Music Rendering** - View transcribed music directly in browser
- ⬇️ **Export Options** - Download MIDI files and ABC notation

## Architecture

```
┌─────────────────┐     ┌─────────────────────────────────────┐
│                 │     │           Backend (Python)           │
│   React + Vite  │────▶│  FastAPI + Demucs + basic-pitch     │
│    Frontend     │◀────│                                      │
│                 │     │  1. Download audio (yt-dlp)          │
└─────────────────┘     │  2. Separate instruments (Demucs)    │
                        │  3. Transcribe to MIDI (basic-pitch) │
                        │  4. Convert to ABC notation          │
                        └─────────────────────────────────────┘
```

## Prerequisites

- **Node.js** 18+ (for frontend)
- **Python** 3.9+ (for backend)
- **uv** - Fast Python package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))
- **FFmpeg** (required for audio processing)

### Installing FFmpeg

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use:
```bash
choco install ffmpeg
```

## Setup

### Backend Setup

Using [uv](https://docs.astral.sh/uv/) for fast Python package management:

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies and create virtual environment:**
   ```bash
   cd backend
   uv sync
   ```

3. **Start the backend server:**
   ```bash
   uv run python main.py
   ```
   
   The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Start the development server:**
   ```bash
   npm run dev
   ```
   
   The app will be available at `http://localhost:5173`

## Usage

1. Open `http://localhost:5173` in your browser
2. Paste a YouTube URL containing music with piano
3. Click "Extract Piano"
4. Wait for processing (this may take several minutes for long videos)
5. View the sheet music preview and download MIDI/ABC files

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/transcribe` | POST | Start a transcription job |
| `/api/status/{job_id}` | GET | Get job status |
| `/api/download/{job_id}/midi` | GET | Download MIDI file |
| `/api/download/{job_id}/abc` | GET | Download ABC notation |
| `/api/health` | GET | Health check |

## Tech Stack

### Frontend
- **React 18** - UI framework
- **Vite** - Build tool
- **TypeScript** - Type safety
- **Framer Motion** - Animations
- **abcjs** - Sheet music rendering
- **Lucide React** - Icons

### Backend
- **FastAPI** - Web framework
- **yt-dlp** - YouTube audio download
- **Demucs** - AI source separation (by Meta)
- **basic-pitch** - Audio-to-MIDI transcription (by Spotify)
- **pretty_midi** - MIDI processing

## Notes

- **Processing Time**: Transcription can take 5-15 minutes depending on video length
- **GPU Acceleration**: If you have a CUDA-compatible GPU, Demucs will automatically use it for faster processing
- **Best Results**: Works best with:
  - Clear piano recordings
  - Less complex arrangements
  - Good audio quality
- **Limitations**: 
  - Complex polyphonic music may have transcription errors
  - Very fast passages might not be accurately captured

## Troubleshooting

### "CUDA out of memory"
Set Demucs to use CPU by setting environment variable:
```bash
export DEMUCS_DEVICE=cpu
```

### "yt-dlp download failed"
Update yt-dlp to the latest version:
```bash
uv add yt-dlp --upgrade
```

### Backend won't start
Make sure all dependencies are installed and FFmpeg is available in your PATH.

## License

MIT License - feel free to use and modify for your needs.

---

Built with ❤️ for musicians
