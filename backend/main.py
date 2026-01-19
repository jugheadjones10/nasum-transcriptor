"""
Nasum Transcriptor - Vocal Melody & Chord Transcription Backend
Transcribes vocal melodies and detects chord progressions from YouTube music
"""

import os

# MUST be set before any PyTorch imports - enables CPU fallback for MPS unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import asyncio
import json
import logging
import re
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np
import yt_dlp
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Nasum Transcriptor API",
    description="Transcribe vocal melodies with chord notations from YouTube music",
    version="2.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for job status
jobs: dict = {}

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=2)

# Ensure directories exist
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


class TranscriptionRequest(BaseModel):
    youtube_url: str


class ContinueRequest(BaseModel):
    job_id: str


class SeparatedTrack(BaseModel):
    name: str
    url: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str
    cached: Optional[bool] = None
    video_title: Optional[str] = None
    separated_tracks: Optional[List[SeparatedTrack]] = None
    midi_url: Optional[str] = None
    abc_notation: Optional[str] = None
    chords: Optional[List[dict]] = None
    error: Optional[str] = None


# Chord templates for detection
CHORD_TEMPLATES = {
    "C": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    "Cm": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    "C#": [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    "C#m": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    "D": [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    "Dm": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    "D#": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    "D#m": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    "E": [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    "Em": [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    "F": [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    "Fm": [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    "F#": [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    "F#m": [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    "G": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    "Gm": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    "G#": [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    "G#m": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    "A": [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    "Am": [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    "A#": [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    "A#m": [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    "B": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
    "Bm": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
}


def extract_video_id(youtube_url: str) -> Optional[str]:
    """Extract video ID from various YouTube URL formats."""
    # Try different URL patterns
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)

    # Try parsing as query parameter
    parsed = urlparse(youtube_url)
    if parsed.query:
        params = parse_qs(parsed.query)
        if "v" in params:
            return params["v"][0]

    return None


def get_cache_path(video_id: str) -> Path:
    """Get the cache directory path for a video ID."""
    return CACHE_DIR / video_id


def is_cached(video_id: str) -> bool:
    """Check if separation results are cached for this video."""
    cache_path = get_cache_path(video_id)
    if not cache_path.exists():
        return False

    # Check if we have at least vocals and some form of instrumental
    has_vocals = (cache_path / "vocals.wav").exists()
    has_instrumental = (cache_path / "instrumental.wav").exists() or (
        cache_path / "other.wav"
    ).exists()
    has_original = (cache_path / "original.wav").exists()

    if not (has_vocals and has_instrumental and has_original):
        return False

    # Check if metadata exists
    if not (cache_path / "metadata.json").exists():
        return False

    return True


def get_cached_metadata(video_id: str) -> Optional[dict]:
    """Get cached metadata for a video."""
    cache_path = get_cache_path(video_id)
    metadata_path = cache_path / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            return json.load(f)
    return None


def save_cache_metadata(video_id: str, metadata: dict):
    """Save metadata to cache."""
    cache_path = get_cache_path(video_id)
    cache_path.mkdir(exist_ok=True)

    with open(cache_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def copy_from_cache(video_id: str, job_dir: Path) -> dict:
    """Copy cached stems to job directory."""
    cache_path = get_cache_path(video_id)
    stems = {}

    # Handle both old (demucs) and new (audio-separator) stem names
    stem_names = ["vocals", "instrumental", "drums", "bass", "other", "original"]

    for stem_name in stem_names:
        src = cache_path / f"{stem_name}.wav"
        if src.exists():
            dest = job_dir / f"{stem_name}.wav"
            shutil.copy(src, dest)
            stems[stem_name] = dest
            logger.info(f"📦 Copied from cache: {stem_name}")

    # Ensure "other" exists for chord detection (use instrumental if no "other")
    if "instrumental" in stems and "other" not in stems:
        stems["other"] = stems["instrumental"]

    return stems


def save_to_cache(video_id: str, stems: dict, metadata: dict):
    """Save stems to cache for future use."""
    cache_path = get_cache_path(video_id)
    cache_path.mkdir(exist_ok=True)

    for stem_name, stem_path in stems.items():
        if stem_path and Path(stem_path).exists():
            dest = cache_path / f"{stem_name}.wav"
            if not dest.exists():
                shutil.copy(stem_path, dest)
                logger.info(f"💾 Cached: {stem_name}")

    save_cache_metadata(video_id, metadata)
    logger.info(f"💾 Cache saved for video: {video_id}")


def download_audio(youtube_url: str, output_path: Path) -> tuple[Path, dict]:
    """Download audio from YouTube video. Returns (audio_path, video_info)."""
    logger.info(f"📥 Downloading audio from: {youtube_url}")

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "outtmpl": str(output_path / "audio"),
        "quiet": False,
        "no_warnings": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_info = {
            "title": info.get("title", "Unknown"),
            "id": info.get("id", ""),
            "duration": info.get("duration", 0),
            "uploader": info.get("uploader", ""),
        }
        logger.info(f"📥 Downloaded: {video_info['title']}")

    audio_path = output_path / "audio.wav"
    if not audio_path.exists():
        for f in output_path.glob("*.wav"):
            audio_path = f
            break
        else:
            for f in output_path.glob("audio.*"):
                audio_path = f
                break

    logger.info(f"✅ Audio saved to: {audio_path}")
    return audio_path, video_info


def separate_all_stems(audio_path: Path, output_path: Path) -> dict:
    """Separate audio into stems using audio-separator with BS Roformer (SOTA)."""
    import torch

    # Disable MPS to avoid FFT/ComplexFloat issues on Apple Silicon
    # Must be done before importing Separator
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False

    from audio_separator.separator import Separator

    logger.info("🎤 Separating audio with audio-separator (BS Roformer)...")
    logger.info("   This may take several minutes (running on CPU)...")

    # Initialize separator - will use CPU since MPS is disabled
    separator = Separator(
        output_dir=str(output_path),
        output_format="wav",
    )

    # Load BS Roformer model (best for vocals) - will download on first use
    # Alternative models: "htdemucs_ft.yaml", "MDX23C-8KFFT-InstVoc_HQ.ckpt"
    try:
        separator.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
        logger.info("✅ Loaded BS Roformer model")
    except Exception as e:
        logger.warning(f"⚠️ BS Roformer failed, trying fallback model: {e}")
        # Fallback to a more compatible model
        separator.load_model(model_filename="htdemucs_ft.yaml")
        logger.info("✅ Loaded htdemucs_ft model (fallback)")

    # Perform separation
    logger.info(f"🔄 Processing: {audio_path}")
    output_files = separator.separate(str(audio_path))

    logger.info("✅ Source separation complete")

    # Map output files to stem names
    stems = {}
    stems["original"] = audio_path

    for output_file in output_files:
        output_path_file = Path(output_file)
        filename_lower = output_path_file.stem.lower()

        # Determine stem type from filename
        if "vocal" in filename_lower or "vocals" in filename_lower:
            stems["vocals"] = output_path_file
            logger.info(f"✅ Found vocals: {output_path_file}")
        elif (
            "instrumental" in filename_lower
            or "instrum" in filename_lower
            or "no_vocal" in filename_lower
        ):
            stems["instrumental"] = output_path_file
            logger.info(f"✅ Found instrumental: {output_path_file}")
        elif "drum" in filename_lower:
            stems["drums"] = output_path_file
            logger.info(f"✅ Found drums: {output_path_file}")
        elif "bass" in filename_lower:
            stems["bass"] = output_path_file
            logger.info(f"✅ Found bass: {output_path_file}")
        elif "other" in filename_lower:
            stems["other"] = output_path_file
            logger.info(f"✅ Found other: {output_path_file}")
        else:
            # If we can't identify, check if it's the only non-vocal output
            if "vocals" in stems and "instrumental" not in stems:
                stems["instrumental"] = output_path_file
                logger.info(f"✅ Found instrumental (inferred): {output_path_file}")

    # If we only got vocals + instrumental, that's fine for our use case
    # The instrumental can be used for chord detection
    if "instrumental" in stems and "other" not in stems:
        stems["other"] = stems["instrumental"]

    return stems


def copy_stems_for_serving(job_id: str, stems: dict) -> List[SeparatedTrack]:
    """Copy stems to a servable location and return track info."""
    job_dir = OUTPUT_DIR / job_id
    tracks = []

    for name, path in stems.items():
        if path and Path(path).exists():
            dest = job_dir / f"{name}.wav"
            if Path(path) != dest and not dest.exists():
                shutil.copy(path, dest)

            tracks.append(
                SeparatedTrack(
                    name=name.replace("_", " ").title(),
                    url=f"/api/audio/{job_id}/{name}",
                )
            )

    return tracks


def transcribe_vocal_melody(audio_path: Path, output_path: Path) -> Path:
    """Transcribe vocal melody to MIDI using basic-pitch."""
    logger.info("🎼 Transcribing vocal melody to MIDI...")

    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict_and_save

    predict_and_save(
        [str(audio_path)],
        str(output_path),
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
    )

    for p in output_path.glob("*.mid"):
        logger.info(f"✅ Vocal melody MIDI generated: {p}")
        return p

    raise FileNotFoundError("MIDI file not generated")


def detect_chords(audio_path: Path, hop_length: int = 22050) -> List[dict]:
    """Detect chord progression from accompaniment audio."""
    logger.info("🎸 Detecting chord progression...")

    import librosa

    y, sr = librosa.load(str(audio_path), sr=22050)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

    template_names = list(CHORD_TEMPLATES.keys())
    template_matrix = np.array([CHORD_TEMPLATES[name] for name in template_names])

    chords = []
    prev_chord = None

    for i in range(chroma.shape[1]):
        frame_chroma = chroma[:, i]

        if np.sum(frame_chroma) > 0:
            frame_chroma = frame_chroma / np.sum(frame_chroma)

        best_match = None
        best_score = -1

        for j, template in enumerate(template_matrix):
            template_norm = (
                template / np.sum(template) if np.sum(template) > 0 else template
            )
            score = np.dot(frame_chroma, template_norm)
            if score > best_score:
                best_score = score
                best_match = template_names[j]

        time_sec = i * hop_length / sr

        if best_match != prev_chord and best_score > 0.3:
            chords.append(
                {
                    "time": round(time_sec, 2),
                    "chord": best_match,
                    "confidence": round(float(best_score), 2),
                }
            )
            prev_chord = best_match

    simplified_chords = []
    for i, chord in enumerate(chords):
        if i == len(chords) - 1:
            simplified_chords.append(chord)
        elif chords[i + 1]["time"] - chord["time"] >= 0.5:
            simplified_chords.append(chord)

    logger.info(f"✅ Detected {len(simplified_chords)} chord changes")
    return simplified_chords


def midi_to_abc_with_chords(midi_path: Path, chords: List[dict]) -> str:
    """Convert MIDI melody to ABC notation with chord symbols."""
    logger.info("📜 Generating lead sheet with melody and chords...")

    import pretty_midi

    midi_data = pretty_midi.PrettyMIDI(str(midi_path))

    all_notes = []
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                all_notes.append(
                    {
                        "pitch": note.pitch,
                        "start": note.start,
                        "end": note.end,
                        "duration": note.end - note.start,
                        "velocity": note.velocity,
                    }
                )

    all_notes.sort(key=lambda x: x["start"])
    logger.info(f"   Found {len(all_notes)} melody notes")

    if not all_notes:
        return "X:1\nT:Transcribed Melody\nM:4/4\nL:1/8\nK:C\nz8|"

    key = estimate_key(chords) if chords else "C"

    abc = "X:1\n"
    abc += "T:Transcribed Vocal Melody\n"
    abc += "C:Auto-transcribed\n"
    abc += "M:4/4\n"
    abc += "L:1/8\n"
    abc += f"K:{key}\n"
    abc += "%%stretchlast\n"

    note_names = ["C", "^C", "D", "^D", "E", "F", "^F", "G", "^G", "A", "^A", "B"]

    def pitch_to_abc(pitch: int) -> str:
        octave = pitch // 12 - 1
        note_idx = pitch % 12
        note = note_names[note_idx]

        if octave >= 5:
            note = note.lower()
            octave_marks = "'" * (octave - 5)
        else:
            octave_marks = "," * (4 - octave)

        return note + octave_marks

    bpm = midi_data.estimate_tempo() if midi_data.estimate_tempo() > 0 else 120
    seconds_per_beat = 60.0 / bpm
    seconds_per_measure = seconds_per_beat * 4

    measures = []

    for note in all_notes[:300]:
        measure_idx = int(note["start"] / seconds_per_measure)
        beat_in_measure = (note["start"] % seconds_per_measure) / seconds_per_beat

        while len(measures) <= measure_idx:
            measures.append({"notes": [], "chords": []})

        measures[measure_idx]["notes"].append(
            {"abc": pitch_to_abc(note["pitch"]), "beat": beat_in_measure}
        )

    for chord in chords:
        measure_idx = int(chord["time"] / seconds_per_measure)
        if measure_idx < len(measures):
            measures[measure_idx]["chords"].append(chord["chord"])

    for i, measure in enumerate(measures[:48]):
        if measure["chords"]:
            chord_str = measure["chords"][0]
            abc += f'"{chord_str}"'

        if measure["notes"]:
            notes_str = " ".join([n["abc"] for n in measure["notes"][:8]])
            abc += notes_str
        else:
            abc += "z4"

        if (i + 1) % 4 == 0:
            abc += "|\n"
        else:
            abc += "|"

    logger.info(f"✅ Lead sheet generated with {len(measures)} measures")
    return abc


def estimate_key(chords: List[dict]) -> str:
    """Estimate the key from chord progression."""
    if not chords:
        return "C"

    chord_counts = {}
    for c in chords:
        chord = c["chord"]
        chord_counts[chord] = chord_counts.get(chord, 0) + 1

    if chord_counts:
        most_common = max(chord_counts, key=chord_counts.get)
        return (
            most_common.replace("m", "") if most_common.endswith("m") else most_common
        )

    return "C"


def run_separation_sync(job_id: str, youtube_url: str):
    """Phase 1: Download and separate audio (with caching)."""
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    try:
        # Extract video ID for caching
        video_id = extract_video_id(youtube_url)

        if video_id and is_cached(video_id):
            # Use cached results!
            logger.info(
                f"[{job_id[:8]}] 📦 Using cached separation for video: {video_id}"
            )

            jobs[job_id]["status"] = "loading_cache"
            jobs[job_id]["message"] = "Loading from cache..."
            jobs[job_id]["progress"] = 20
            jobs[job_id]["cached"] = True

            # Get cached metadata
            metadata = get_cached_metadata(video_id)
            if metadata:
                jobs[job_id]["video_title"] = metadata.get("title", "")

            # Copy from cache
            stems = copy_from_cache(video_id, job_dir)

            # Store stem paths
            jobs[job_id]["_stems"] = {k: str(v) for k, v in stems.items()}
            jobs[job_id]["_video_id"] = video_id

            # Create track list for serving
            tracks = copy_stems_for_serving(job_id, stems)

            # Ready for review
            jobs[job_id]["status"] = "waiting_review"
            jobs[job_id]["message"] = (
                "✨ Loaded from cache! Listen to the tracks and click Continue."
            )
            jobs[job_id]["progress"] = 50
            jobs[job_id]["separated_tracks"] = [t.model_dump() for t in tracks]

            logger.info(f"[{job_id[:8]}] ✅ Loaded from cache, waiting for review")
            return

        # Not cached - do full processing
        jobs[job_id]["cached"] = False

        # Step 1: Download audio
        logger.info(f"[{job_id[:8]}] Starting download...")
        jobs[job_id]["status"] = "downloading"
        jobs[job_id]["message"] = "Downloading audio from YouTube..."
        jobs[job_id]["progress"] = 10

        audio_path, video_info = download_audio(youtube_url, job_dir)
        jobs[job_id]["video_title"] = video_info.get("title", "")

        # Update video_id from actual download if we couldn't extract it earlier
        if not video_id:
            video_id = video_info.get("id", "")

        jobs[job_id]["_video_id"] = video_id

        # Step 2: Separate all stems
        jobs[job_id]["status"] = "separating"
        jobs[job_id]["message"] = (
            "Separating audio into stems (vocals, drums, bass, other)..."
        )
        jobs[job_id]["progress"] = 30

        stems = separate_all_stems(audio_path, job_dir)

        # Save to cache for future use
        if video_id:
            save_to_cache(video_id, stems, video_info)

        # Copy stems for serving
        tracks = copy_stems_for_serving(job_id, stems)

        # Store stem paths for later use
        jobs[job_id]["_stems"] = {k: str(v) for k, v in stems.items()}

        # Ready for review
        jobs[job_id]["status"] = "waiting_review"
        jobs[job_id]["message"] = (
            "Separation complete! Listen to the tracks and click Continue when ready."
        )
        jobs[job_id]["progress"] = 50
        jobs[job_id]["separated_tracks"] = [t.model_dump() for t in tracks]

        logger.info(f"[{job_id[:8]}] ✅ Separation complete, waiting for review")

    except Exception as e:
        logger.error(f"[{job_id[:8]}] ❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = "Separation failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["progress"] = 0


def run_transcription_sync(job_id: str):
    """Phase 2: Transcribe vocals and detect chords."""
    job_dir = OUTPUT_DIR / job_id

    try:
        stems = jobs[job_id].get("_stems", {})
        vocals_path = Path(stems.get("vocals", job_dir / "vocals.wav"))
        accompaniment_path = Path(stems.get("other", job_dir / "other.wav"))

        if not vocals_path.exists():
            vocals_path = job_dir / "original.wav"
        if not accompaniment_path.exists():
            accompaniment_path = job_dir / "original.wav"

        # Step 3: Transcribe vocal melody
        jobs[job_id]["status"] = "transcribing"
        jobs[job_id]["message"] = "Transcribing vocal melody..."
        jobs[job_id]["progress"] = 60

        midi_path = transcribe_vocal_melody(vocals_path, job_dir)

        # Step 4: Detect chords
        jobs[job_id]["status"] = "detecting_chords"
        jobs[job_id]["message"] = "Detecting chord progression..."
        jobs[job_id]["progress"] = 75

        chords = detect_chords(accompaniment_path)

        # Step 5: Generate lead sheet
        jobs[job_id]["status"] = "generating"
        jobs[job_id]["message"] = "Generating lead sheet..."
        jobs[job_id]["progress"] = 90

        abc_notation = midi_to_abc_with_chords(midi_path, chords)

        # Save files
        final_midi_path = job_dir / "vocal_melody.mid"
        shutil.copy(midi_path, final_midi_path)

        abc_path = job_dir / "lead_sheet.abc"
        abc_path.write_text(abc_notation)

        chords_path = job_dir / "chords.txt"
        chord_text = "\n".join([f"{c['time']:.1f}s: {c['chord']}" for c in chords])
        chords_path.write_text(chord_text)

        # Complete
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["message"] = "Lead sheet generated!"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["midi_url"] = f"/api/download/{job_id}/midi"
        jobs[job_id]["abc_notation"] = abc_notation
        jobs[job_id]["chords"] = chords

        logger.info(f"[{job_id[:8]}] ✅ Lead sheet complete!")

    except Exception as e:
        logger.error(f"[{job_id[:8]}] ❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = "Transcription failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["progress"] = 0


async def process_separation(job_id: str, youtube_url: str):
    """Phase 1: Download and separate."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, run_separation_sync, job_id, youtube_url)


async def process_transcription(job_id: str):
    """Phase 2: Transcribe."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, run_transcription_sync, job_id)


@app.post("/api/transcribe", response_model=JobStatus)
async def start_transcription(
    request: TranscriptionRequest, background_tasks: BackgroundTasks
):
    """Start a new transcription job (Phase 1: Download & Separate)."""
    job_id = str(uuid.uuid4())

    logger.info(f"🎬 New job: {job_id[:8]} - {request.youtube_url}")

    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "message": "Job queued...",
        "cached": None,
        "video_title": None,
        "separated_tracks": None,
        "midi_url": None,
        "abc_notation": None,
        "chords": None,
        "error": None,
    }

    background_tasks.add_task(process_separation, job_id, request.youtube_url)

    return JobStatus(**{k: v for k, v in jobs[job_id].items() if not k.startswith("_")})


@app.post("/api/continue/{job_id}", response_model=JobStatus)
async def continue_transcription(job_id: str, background_tasks: BackgroundTasks):
    """Continue with transcription after reviewing separated tracks (Phase 2)."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if jobs[job_id]["status"] != "waiting_review":
        raise HTTPException(status_code=400, detail="Job is not waiting for review")

    logger.info(f"[{job_id[:8]}] Continuing with transcription...")

    jobs[job_id]["status"] = "transcribing"
    jobs[job_id]["message"] = "Starting transcription..."

    background_tasks.add_task(process_transcription, job_id)

    return JobStatus(**{k: v for k, v in jobs[job_id].items() if not k.startswith("_")})


@app.get("/api/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a transcription job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(**{k: v for k, v in jobs[job_id].items() if not k.startswith("_")})


@app.get("/api/audio/{job_id}/{stem_name}")
async def get_audio_stem(job_id: str, stem_name: str):
    """Stream a separated audio stem for preview."""
    audio_path = OUTPUT_DIR / job_id / f"{stem_name}.wav"

    if not audio_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Audio stem '{stem_name}' not found"
        )

    return FileResponse(audio_path, media_type="audio/wav", filename=f"{stem_name}.wav")


@app.get("/api/download/{job_id}/midi")
async def download_midi(job_id: str):
    """Download the generated MIDI file."""
    midi_path = OUTPUT_DIR / job_id / "vocal_melody.mid"

    if not midi_path.exists():
        raise HTTPException(status_code=404, detail="MIDI file not found")

    return FileResponse(midi_path, media_type="audio/midi", filename="vocal_melody.mid")


@app.get("/api/download/{job_id}/abc")
async def download_abc(job_id: str):
    """Download the ABC notation file."""
    abc_path = OUTPUT_DIR / job_id / "lead_sheet.abc"

    if not abc_path.exists():
        raise HTTPException(status_code=404, detail="ABC file not found")

    return FileResponse(abc_path, media_type="text/plain", filename="lead_sheet.abc")


@app.get("/api/download/{job_id}/chords")
async def download_chords(job_id: str):
    """Download the chord progression."""
    chords_path = OUTPUT_DIR / job_id / "chords.txt"

    if not chords_path.exists():
        raise HTTPException(status_code=404, detail="Chords file not found")

    return FileResponse(chords_path, media_type="text/plain", filename="chords.txt")


@app.delete("/api/cache")
async def clear_cache():
    """Clear all cached separations."""
    import shutil

    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(exist_ok=True)

    logger.info("🗑️ Cache cleared")
    return {"status": "ok", "message": "Cache cleared"}


@app.get("/api/cache")
async def list_cache():
    """List all cached videos."""
    cached_videos = []

    if CACHE_DIR.exists():
        for video_dir in CACHE_DIR.iterdir():
            if video_dir.is_dir():
                metadata = get_cached_metadata(video_dir.name)
                cached_videos.append(
                    {
                        "video_id": video_dir.name,
                        "title": metadata.get("title", "Unknown")
                        if metadata
                        else "Unknown",
                    }
                )

    return {"cached_videos": cached_videos, "count": len(cached_videos)}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    logger.info("🎤 Starting Nasum Transcriptor API (Vocal + Chords)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
