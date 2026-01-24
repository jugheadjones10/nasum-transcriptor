"""
Lead Sheet Transcriptor Backend
Converts YouTube audio to lead sheets with melody and chord symbols.

Pipeline (Step-by-step with user review):
1. Download + Separate: Get audio, split vocals/accompaniment
2. Transcribe: Convert vocals to MIDI notes using basic_pitch
3. Detect Chords: Analyze accompaniment for chord progression
4. Generate: Create quantized MusicXML lead sheet
"""

import asyncio
import json
import re
import uuid
from enum import Enum
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
CACHE_DIR = BASE_DIR / "cache"

OUTPUTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# =============================================================================
# Key and BPM are now provided by user input - automatic detection removed
# =============================================================================


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Lead Sheet Transcriptor",
    description="Convert YouTube audio to lead sheets (melody + chords)",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Can't use credentials with allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Models
# =============================================================================


class JobStep(str, Enum):
    """Current step in the pipeline."""

    IDLE = "idle"
    DOWNLOADING = "downloading"
    SEPARATING = "separating"
    SEPARATED = "separated"  # Waiting for user to continue
    TRANSCRIBING = "transcribing"
    TRANSCRIBED = "transcribed"  # Waiting for user to continue
    DETECTING_CHORDS = "detecting_chords"
    CHORDS_DETECTED = "chords_detected"  # Waiting for user to continue
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class MelodyNote(BaseModel):
    """A single note in the melody."""

    pitch: int  # MIDI pitch (60 = C4)
    start_time: float  # seconds
    duration: float  # seconds
    velocity: int = 100


class ChordEvent(BaseModel):
    """A chord at a specific time."""

    time: float  # seconds
    chord: str  # e.g., "Cmaj", "Am7", "G"
    confidence: float = 1.0


class ProcessRequest(BaseModel):
    youtube_url: str
    key: str  # e.g., "C major", "A minor"
    bpm: int  # 20-300


class JobResponse(BaseModel):
    job_id: str
    step: JobStep
    progress: int = 0
    message: str = ""

    # Metadata
    title: Optional[str] = None
    duration: Optional[float] = None  # seconds
    bpm: Optional[float] = None
    key: Optional[str] = None
    time_signature: str = "4/4"

    # Step 1 outputs
    vocals_url: Optional[str] = None
    accompaniment_url: Optional[str] = None
    original_url: Optional[str] = None

    # Step 2 outputs
    melody_notes: Optional[list[MelodyNote]] = None
    melody_midi_url: Optional[str] = None

    # Step 3 outputs
    chords: Optional[list[ChordEvent]] = None

    # Step 4 outputs
    music_xml_url: Optional[str] = None

    error: Optional[str] = None


# In-memory job storage
jobs: dict[str, JobResponse] = {}

# =============================================================================
# Helper Functions
# =============================================================================


def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL."""
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_job_dir(job_id: str) -> Path:
    """Get the output directory for a job."""
    return OUTPUTS_DIR / job_id


def midi_note_to_name(midi_note: int) -> str:
    """Convert MIDI note number to note name."""
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (midi_note // 12) - 1
    note = notes[midi_note % 12]
    return f"{note}{octave}"


# =============================================================================
# Step 1: Download + Separation
# =============================================================================


async def download_audio(
    job_id: str, youtube_url: str, output_dir: Path
) -> tuple[Path, str]:
    """Download audio from YouTube using yt-dlp."""
    jobs[job_id].step = JobStep.DOWNLOADING
    jobs[job_id].message = "Downloading audio from YouTube..."
    jobs[job_id].progress = 5

    output_template = str(output_dir / "original.%(ext)s")

    # First, get the title (use browser cookies to avoid bot detection)
    title_cmd = ["yt-dlp", "--cookies-from-browser", "chrome", "--print", "%(title)s", "--no-playlist", youtube_url]
    title_result = await asyncio.create_subprocess_exec(
        *title_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    title_stdout, _ = await title_result.communicate()
    title = title_stdout.decode().strip().split("\n")[0] if title_stdout else "Unknown"

    print(f"[{job_id}] Title: {title}")
    jobs[job_id].title = title
    jobs[job_id].progress = 8

    async def run_download(format_selector: Optional[str]):
        cmd = [
            "yt-dlp",
            "--cookies-from-browser",
            "chrome",
            "-x",
            "--audio-format",
            "wav",
            "-o",
            output_template,
            "--no-playlist",
            "--no-warnings",
        ]
        if format_selector:
            cmd += ["-f", format_selector]
        cmd.append(youtube_url)

        print(f"[{job_id}] Downloading with: {' '.join(cmd)}")

        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await result.communicate()
        stderr_text = stderr.decode() if stderr else ""

        print(f"[{job_id}] yt-dlp exit code: {result.returncode}")
        if stderr_text:
            print(f"[{job_id}] yt-dlp stderr: {stderr_text[:500]}")

        return result, stderr_text

    async def select_best_format_id() -> Optional[str]:
        info_cmd = [
            "yt-dlp",
            "--cookies-from-browser",
            "chrome",
            "--dump-json",
            "--no-playlist",
            youtube_url,
        ]
        info_result = await asyncio.create_subprocess_exec(
            *info_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        info_stdout, _ = await info_result.communicate()
        if info_result.returncode != 0 or not info_stdout:
            return None

        try:
            info = json.loads(info_stdout.decode().strip().splitlines()[-1])
        except Exception:
            return None

        formats = info.get("formats") or []
        audio_formats = [
            f
            for f in formats
            if f.get("vcodec") == "none" and f.get("acodec") not in (None, "none")
        ]
        if audio_formats:
            best_audio = max(
                audio_formats,
                key=lambda f: f.get("abr") or f.get("tbr") or 0,
            )
            return best_audio.get("format_id")

        if formats:
            best_overall = max(formats, key=lambda f: f.get("tbr") or 0)
            return best_overall.get("format_id")

        return None

    formats_to_try: list[Optional[str]] = ["bestaudio/best", "bestaudio", "best", None]
    result = None
    last_error = ""

    for format_selector in formats_to_try:
        result, stderr_text = await run_download(format_selector)
        if result.returncode == 0:
            break
        last_error = stderr_text
        if "Requested format is not available" not in stderr_text:
            raise Exception(f"yt-dlp failed: {stderr_text}")

    if result is None or result.returncode != 0:
        if "Requested format is not available" in last_error:
            format_id = await select_best_format_id()
            if format_id:
                result, stderr_text = await run_download(format_id)
                if result.returncode != 0:
                    raise Exception(f"yt-dlp failed: {stderr_text}")
            else:
                raise Exception(f"yt-dlp failed: {last_error}")
        else:
            raise Exception(f"yt-dlp failed: {last_error}")

    audio_path = output_dir / "original.wav"

    if not audio_path.exists():
        # Check if there's a webm or other file that didn't get converted
        for ext in ["webm", "opus", "m4a", "mp3"]:
            alt_path = output_dir / f"original.{ext}"
            if alt_path.exists():
                raise Exception(f"Audio downloaded as {ext} but WAV conversion failed")
        raise Exception("Audio file not found after download")

    # Get duration
    y, sr = librosa.load(
        str(audio_path), sr=None, duration=10
    )  # Just load first 10s for duration check
    full_duration = librosa.get_duration(path=str(audio_path))

    jobs[job_id].title = title
    jobs[job_id].duration = full_duration
    jobs[job_id].original_url = f"/api/audio/{job_id}/original"
    jobs[job_id].progress = 15

    return audio_path, title


async def separate_stems(
    job_id: str, audio_path: Path, output_dir: Path
) -> tuple[Path, Path]:
    """Separate audio into vocals and accompaniment using Demucs."""
    jobs[job_id].step = JobStep.SEPARATING
    jobs[
        job_id
    ].message = "Separating vocals and accompaniment (this may take a few minutes)..."
    jobs[job_id].progress = 20

    cmd = [
        "python",
        "-m",
        "demucs",
        "--two-stems",
        "vocals",
        "-n",
        "htdemucs",
        "-o",
        str(output_dir),
        str(audio_path),
    ]

    result = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await result.communicate()

    if result.returncode != 0:
        raise Exception(f"Demucs failed: {stderr.decode()}")

    demucs_output = output_dir / "htdemucs" / "original"
    vocals_path = demucs_output / "vocals.wav"
    accompaniment_path = demucs_output / "no_vocals.wav"

    if not vocals_path.exists() or not accompaniment_path.exists():
        raise Exception("Separated stems not found")

    jobs[job_id].vocals_url = f"/api/audio/{job_id}/vocals"
    jobs[job_id].accompaniment_url = f"/api/audio/{job_id}/accompaniment"
    jobs[job_id].progress = 40

    return vocals_path, accompaniment_path


async def run_step1(job_id: str, youtube_url: str):
    """Execute Step 1: Download and Separate."""
    output_dir = get_job_dir(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        audio_path, title = await download_audio(job_id, youtube_url, output_dir)
        print(f"[{job_id}] Downloaded: {title}")

        vocals_path, accompaniment_path = await separate_stems(
            job_id, audio_path, output_dir
        )
        print(f"[{job_id}] Separated stems")

        # BPM is already set from user input at job creation
        # No automatic detection needed
        print(f"[{job_id}] Using user-provided BPM: {jobs[job_id].bpm}")

        jobs[job_id].step = JobStep.SEPARATED
        jobs[
            job_id
        ].message = "✓ Separation complete. Review the stems and continue when ready."
        jobs[job_id].progress = 40

    except Exception as e:
        print(f"[{job_id}] Step 1 error: {e}")
        jobs[job_id].step = JobStep.FAILED
        jobs[job_id].error = str(e)
        jobs[job_id].message = "Step 1 failed"


# =============================================================================
# Step 2: Transcription (basic_pitch)
# =============================================================================


async def transcribe_melody(
    job_id: str, vocals_path: Path, output_dir: Path
) -> list[MelodyNote]:
    """Transcribe vocals to MIDI notes using basic_pitch."""
    jobs[job_id].step = JobStep.TRANSCRIBING
    jobs[job_id].message = "Transcribing melody from vocals..."
    jobs[job_id].progress = 45

    from basic_pitch.inference import predict

    # Run prediction
    model_output, midi_data, note_events = predict(str(vocals_path))

    # Convert to our format
    notes = []
    for start, end, pitch, velocity, _ in note_events:
        notes.append(
            MelodyNote(
                pitch=int(pitch),
                start_time=float(start),
                duration=float(end - start),
                velocity=int(velocity * 127),
            )
        )

    # Sort by start time
    notes.sort(key=lambda n: n.start_time)

    # Save MIDI file
    midi_path = output_dir / "melody.mid"
    midi_data.write(str(midi_path))

    jobs[job_id].melody_midi_url = f"/api/audio/{job_id}/melody_midi"
    jobs[job_id].progress = 60

    return notes


def run_step2_sync(job_id: str, vocals_path: Path, output_dir: Path):
    """Synchronous transcription using basic_pitch (CPU-bound)."""
    from basic_pitch.inference import predict

    print(f"[{job_id}] Starting basic_pitch prediction on {vocals_path}")
    model_output, midi_data, note_events = predict(str(vocals_path))

    notes = []
    for start, end, pitch, velocity, _ in note_events:
        notes.append(
            MelodyNote(
                pitch=int(pitch),
                start_time=float(start),
                duration=float(end - start),
                velocity=int(velocity * 127),
            )
        )

    notes.sort(key=lambda n: n.start_time)

    midi_path = output_dir / "melody.mid"
    midi_data.write(str(midi_path))
    print(f"[{job_id}] Saved MIDI to {midi_path}")

    return notes


async def run_step2(job_id: str):
    """Execute Step 2: Transcribe melody."""
    output_dir = get_job_dir(job_id)
    vocals_path = output_dir / "htdemucs" / "original" / "vocals.wav"

    try:
        jobs[job_id].step = JobStep.TRANSCRIBING
        jobs[job_id].message = "Transcribing melody from vocals..."
        jobs[job_id].progress = 45

        if not vocals_path.exists():
            raise Exception(f"Vocals file not found: {vocals_path}")

        print(
            f"[{job_id}] Vocals file exists: {vocals_path}, size: {vocals_path.stat().st_size}"
        )

        # Run CPU-bound transcription in executor
        loop = asyncio.get_event_loop()
        notes = await loop.run_in_executor(
            None, run_step2_sync, job_id, vocals_path, output_dir
        )

        jobs[job_id].melody_notes = notes
        jobs[job_id].melody_midi_url = f"/api/audio/{job_id}/melody_midi"

        print(f"[{job_id}] Transcribed {len(notes)} notes")

        jobs[job_id].step = JobStep.TRANSCRIBED
        jobs[
            job_id
        ].message = f"✓ Transcribed {len(notes)} melody notes. Review and continue."
        jobs[job_id].progress = 60

    except Exception as e:
        print(f"[{job_id}] Step 2 error: {e}")
        import traceback

        traceback.print_exc()
        jobs[job_id].step = JobStep.FAILED
        jobs[job_id].error = str(e)
        jobs[job_id].message = "Transcription failed"


# =============================================================================
# Step 3: Chord Detection
# =============================================================================

CHORD_TEMPLATES = {
    "maj": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    "min": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    "7": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    "maj7": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    "min7": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    "dim": [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    "aug": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    "sus4": [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    "sus2": [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def detect_chord_at_frame(chroma: np.ndarray) -> tuple[str, float]:
    """Detect the most likely chord from a chroma vector."""
    best_chord = "N"  # No chord
    best_score = 0.0

    for root in range(12):
        for chord_type, template in CHORD_TEMPLATES.items():
            # Rotate template to match root
            rotated = np.roll(template, root)
            # Compute correlation
            score = np.corrcoef(chroma, rotated)[0, 1]
            if not np.isnan(score) and score > best_score:
                best_score = score
                root_name = NOTE_NAMES[root]
                if chord_type == "maj":
                    best_chord = root_name
                elif chord_type == "min":
                    best_chord = f"{root_name}m"
                else:
                    best_chord = f"{root_name}{chord_type}"

    return best_chord, best_score


def detect_chords_from_audio(audio_path: Path, bpm: float) -> list[ChordEvent]:
    """Detect chords from audio file."""
    y, sr = librosa.load(str(audio_path), sr=22050)

    # Get hop length for ~1 beat resolution
    beat_duration = 60.0 / bpm
    hop_length = int(sr * beat_duration / 2)  # 2 analyses per beat

    # Compute chroma features
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

    chords = []
    prev_chord = None

    for i, frame in enumerate(chroma.T):
        time = i * hop_length / sr
        chord, confidence = detect_chord_at_frame(frame)

        # Only add if chord changed
        if chord != prev_chord and confidence > 0.5:
            chords.append(
                ChordEvent(
                    time=time,
                    chord=chord,
                    confidence=confidence,
                )
            )
            prev_chord = chord

    return chords


## detect_key() function removed - key is now provided by user input


async def run_step3(job_id: str):
    """Execute Step 3: Detect chords."""
    output_dir = get_job_dir(job_id)
    accompaniment_path = output_dir / "htdemucs" / "original" / "no_vocals.wav"

    try:
        jobs[job_id].step = JobStep.DETECTING_CHORDS
        jobs[job_id].message = "Analyzing chord progression..."
        jobs[job_id].progress = 65

        bpm = jobs[job_id].bpm or 120.0

        # Detect chords
        chords = detect_chords_from_audio(accompaniment_path, bpm)

        # Key is already set from user input at job creation
        # No automatic detection needed
        key = jobs[job_id].key or "C major"

        jobs[job_id].chords = chords
        # key is already set, no need to update

        print(f"[{job_id}] Detected {len(chords)} chord changes, using user-provided key: {key}")

        # Save chords to file (include key and bpm for restore)
        chords_data = [
            {"time": c.time, "chord": c.chord, "confidence": c.confidence}
            for c in chords
        ]
        with open(output_dir / "chords.json", "w") as f:
            json.dump({"key": key, "bpm": bpm, "chords": chords_data}, f, indent=2)

        jobs[job_id].step = JobStep.CHORDS_DETECTED
        jobs[
            job_id
        ].message = f"✓ Detected {len(chords)} chords in {key}. Review and continue."
        jobs[job_id].progress = 80

    except Exception as e:
        print(f"[{job_id}] Step 3 error: {e}")
        jobs[job_id].step = JobStep.FAILED
        jobs[job_id].error = str(e)
        jobs[job_id].message = "Chord detection failed"


# =============================================================================
# Step 4: Generate MusicXML
# =============================================================================


def quantize_time(time: float, bpm: float, subdivision: int = 16) -> float:
    """Quantize time to nearest subdivision of a beat."""
    beat_duration = 60.0 / bpm
    subdivision_duration = beat_duration / (subdivision / 4)
    return round(time / subdivision_duration) * subdivision_duration


def quantize_duration(duration: float, bpm: float, min_note: int = 16) -> float:
    """Quantize duration to nearest note value."""
    beat_duration = 60.0 / bpm
    min_duration = beat_duration / (min_note / 4)
    quantized = max(min_duration, round(duration / min_duration) * min_duration)
    return quantized


def snap_to_valid_duration(quarter_length: float) -> float:
    """Snap a duration to a valid music notation value (in quarter notes)."""
    # Valid durations: whole=4, half=2, quarter=1, eighth=0.5, sixteenth=0.25, thirty-second=0.125
    # Also with dots: dotted half=3, dotted quarter=1.5, dotted eighth=0.75
    valid_durations = [4, 3, 2, 1.5, 1, 0.75, 0.5, 0.375, 0.25, 0.1875, 0.125]

    # Find closest valid duration
    closest = min(valid_durations, key=lambda x: abs(x - quarter_length))
    return closest


def generate_musicxml(
    notes: list[MelodyNote],
    chords: list[ChordEvent],
    bpm: float,
    key: str,
    title: str,
    time_sig: str = "4/4",
) -> str:
    """Generate MusicXML lead sheet using music21."""
    from music21 import harmony, metadata, meter, note, stream, tempo
    from music21 import key as m21key

    # Create score
    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = title

    # Create part for melody
    melody_part = stream.Part()
    melody_part.id = "Melody"

    # Add tempo
    melody_part.append(tempo.MetronomeMark(number=bpm))

    # Add time signature
    num, denom = map(int, time_sig.split("/"))
    melody_part.append(meter.TimeSignature(time_sig))

    # Add key signature - STRICTLY use the user-provided key
    print(f"[generate_musicxml] User-provided key: {key}")
    key_name = key.split()[0] if key else "C"
    is_minor = "minor" in key.lower() if key else False
    print(f"[generate_musicxml] Parsed key_name: {key_name}, is_minor: {is_minor}")

    try:
        ks = m21key.Key(key_name, "minor" if is_minor else "major")
        melody_part.append(ks)
        print(f"[generate_musicxml] Created key signature: {ks}")
    except Exception as e:
        print(f"[generate_musicxml] Could not create key signature for {key}: {e}")
        # Fall back to C major only if key creation fails
        try:
            ks = m21key.Key("C", "major")
            melody_part.append(ks)
        except:
            pass

    # Calculate timing
    beat_duration_seconds = 60.0 / bpm
    measure_duration_seconds = beat_duration_seconds * num

    # Convert notes to quarter-length based timing
    quantized_notes = []
    for n in notes:
        # Convert time from seconds to quarter notes
        start_quarters = n.start_time / beat_duration_seconds
        duration_quarters = n.duration / beat_duration_seconds

        # Snap to 16th note grid
        start_quarters = round(start_quarters * 4) / 4
        duration_quarters = snap_to_valid_duration(duration_quarters)

        if duration_quarters > 0:
            quantized_notes.append(
                {
                    "pitch": n.pitch,
                    "start": start_quarters,
                    "duration": duration_quarters,
                }
            )

    # Build chord lookup (time in quarter notes -> chord symbol)
    chord_lookup = {}
    for c in chords:
        q_time = round(c.time / beat_duration_seconds)  # Snap to beats
        chord_lookup[q_time] = c.chord

    # Calculate measures needed
    if quantized_notes:
        max_quarters = max(n["start"] + n["duration"] for n in quantized_notes)
    else:
        max_quarters = 0

    num_measures = int(max_quarters / num) + 1
    num_measures = min(
        num_measures, 64
    )  # Cap at 64 measures for better rendering performance

    for measure_num in range(num_measures):
        m = stream.Measure(number=measure_num + 1)
        measure_start = measure_num * num
        measure_end = measure_start + num

        # Add chord symbol at start of measure
        if measure_start in chord_lookup:
            try:
                cs = harmony.ChordSymbol(chord_lookup[measure_start])
                m.insert(0, cs)
            except:
                pass

        # Get notes in this measure
        measure_notes = [
            n for n in quantized_notes if measure_start <= n["start"] < measure_end
        ]

        # Build measure content
        current_pos = 0.0

        for n in sorted(measure_notes, key=lambda x: x["start"]):
            note_offset = n["start"] - measure_start

            # Add rest if there's a gap
            if note_offset > current_pos + 0.001:
                gap = note_offset - current_pos
                gap_snapped = snap_to_valid_duration(gap)
                if gap_snapped > 0:
                    r = note.Rest(quarterLength=gap_snapped)
                    m.append(r)
                    current_pos += gap_snapped

            # Calculate note duration (clipped to measure boundary)
            note_dur = min(n["duration"], measure_end - n["start"])
            note_dur = snap_to_valid_duration(note_dur)

            if note_dur > 0:
                n_obj = note.Note(n["pitch"], quarterLength=note_dur)
                m.append(n_obj)
                current_pos = note_offset + note_dur

        # Fill remaining with rest
        remaining = num - current_pos
        if remaining > 0.001:
            remaining = snap_to_valid_duration(remaining)
            if remaining > 0:
                r = note.Rest(quarterLength=remaining)
                m.append(r)

        melody_part.append(m)

    score.append(melody_part)

    # Make measures and export
    melody_part.makeMeasures(inPlace=True)

    # Export to MusicXML
    xml_path = score.write("musicxml")
    return xml_path.read_text()


async def run_step4(job_id: str):
    """Execute Step 4: Generate MusicXML."""
    output_dir = get_job_dir(job_id)

    try:
        jobs[job_id].step = JobStep.GENERATING
        jobs[job_id].message = "Generating lead sheet..."
        jobs[job_id].progress = 85

        job = jobs[job_id]

        # Log the user-provided key and BPM
        print(f"[{job_id}] Generating MusicXML with user-provided key: {job.key}, bpm: {job.bpm}")

        # Generate MusicXML - STRICTLY use user-provided key and bpm
        musicxml = generate_musicxml(
            notes=job.melody_notes or [],
            chords=job.chords or [],
            bpm=job.bpm or 120.0,
            key=job.key or "C major",
            title=job.title or "Untitled",
            time_sig=job.time_signature,
        )

        # Save to file
        xml_path = output_dir / "lead_sheet.musicxml"
        xml_path.write_text(musicxml)

        jobs[job_id].music_xml_url = f"/api/musicxml/{job_id}"

        print(f"[{job_id}] Generated MusicXML")

        jobs[job_id].step = JobStep.COMPLETED
        jobs[job_id].message = "✓ Lead sheet complete!"
        jobs[job_id].progress = 100

    except Exception as e:
        print(f"[{job_id}] Step 4 error: {e}")
        jobs[job_id].step = JobStep.FAILED
        jobs[job_id].error = str(e)
        jobs[job_id].message = "MusicXML generation failed"


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/")
async def root():
    return {"message": "Lead Sheet Transcriptor API", "version": "0.2.0"}


# Valid key notes and modes
# Valid key notes (includes both sharps and flats)
VALID_NOTES = ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]
VALID_MODES = ["major", "minor"]


@app.post("/api/process-song", response_model=JobResponse)
async def create_job(request: ProcessRequest, background_tasks: BackgroundTasks):
    """Start processing a YouTube URL (Step 1)."""

    video_id = extract_video_id(request.youtube_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    # Validate key
    key_parts = request.key.split(" ")
    if len(key_parts) != 2:
        raise HTTPException(
            status_code=400,
            detail="Invalid key format: must be '{note} {mode}' (e.g., 'C major')",
        )
    note, mode = key_parts
    if note not in VALID_NOTES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid key note: must be one of {', '.join(VALID_NOTES)}",
        )
    if mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid key mode: must be one of {', '.join(VALID_MODES)}",
        )

    # Validate BPM
    if request.bpm < 20 or request.bpm > 300:
        raise HTTPException(
            status_code=400, detail="Invalid BPM: must be between 20 and 300"
        )

    job_id = str(uuid.uuid4())
    jobs[job_id] = JobResponse(
        job_id=job_id,
        step=JobStep.IDLE,
        progress=0,
        message="Starting download and separation...",
        key=request.key,  # Store user-provided key
        bpm=float(request.bpm),  # Store user-provided BPM
    )

    # Store URL for later steps
    output_dir = get_job_dir(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "url.txt").write_text(request.youtube_url)

    # Start Step 1
    background_tasks.add_task(run_step1, job_id, request.youtube_url)

    return jobs[job_id]


@app.post("/api/job/{job_id}/continue", response_model=JobResponse)
async def continue_job(job_id: str, background_tasks: BackgroundTasks):
    """Continue to the next step after user review."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job.step == JobStep.SEPARATED:
        # Continue to Step 2: Transcription
        background_tasks.add_task(run_step2, job_id)
        job.message = "Starting transcription..."

    elif job.step == JobStep.TRANSCRIBED:
        # Continue to Step 3: Chord Detection
        background_tasks.add_task(run_step3, job_id)
        job.message = "Starting chord detection..."

    elif job.step == JobStep.CHORDS_DETECTED:
        # Continue to Step 4: Generate MusicXML
        background_tasks.add_task(run_step4, job_id)
        job.message = "Starting MusicXML generation..."

    else:
        raise HTTPException(
            status_code=400, detail=f"Cannot continue from step: {job.step}"
        )

    return job


@app.get("/api/job/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get the status of a processing job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/api/audio/{job_id}/{stem}")
async def get_audio(job_id: str, stem: str):
    """Get an audio file (works from cache even if job not in memory)."""
    output_dir = get_job_dir(job_id)

    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    if stem == "vocals":
        audio_path = output_dir / "htdemucs" / "original" / "vocals.wav"
    elif stem == "accompaniment":
        audio_path = output_dir / "htdemucs" / "original" / "no_vocals.wav"
    elif stem == "original":
        audio_path = output_dir / "original.wav"
    elif stem == "melody_midi":
        audio_path = output_dir / "melody.mid"
        if audio_path.exists():
            return FileResponse(audio_path, media_type="audio/midi")
    else:
        raise HTTPException(status_code=400, detail="Invalid stem type")

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(audio_path, media_type="audio/wav")


@app.get("/api/musicxml/{job_id}")
async def get_musicxml(job_id: str):
    """Get the generated MusicXML file."""
    # Allow fetching from cache even if job not in memory
    output_dir = get_job_dir(job_id)
    xml_path = output_dir / "lead_sheet.musicxml"

    if not xml_path.exists():
        raise HTTPException(status_code=404, detail="MusicXML file not found")

    return FileResponse(xml_path, media_type="application/xml")


@app.post("/api/job/{job_id}/restore", response_model=JobResponse)
async def restore_job(job_id: str):
    """Restore a job from cached files on disk."""
    output_dir = get_job_dir(job_id)

    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Job directory not found")

    # Check what files exist
    has_xml = (output_dir / "lead_sheet.musicxml").exists()
    has_chords = (output_dir / "chords.json").exists()
    has_midi = (output_dir / "melody.mid").exists()
    has_vocals = (output_dir / "htdemucs" / "original" / "vocals.wav").exists()
    has_accompaniment = (
        output_dir / "htdemucs" / "original" / "no_vocals.wav"
    ).exists()
    has_original = (output_dir / "original.wav").exists()

    # If we have chords and midi but no XML, regenerate it
    if has_chords and has_midi and not has_xml:
        try:
            # Load chords data (includes key and bpm from user input)
            with open(output_dir / "chords.json") as f:
                chords_data = json.load(f)
                key = chords_data.get("key", "C major")
                bpm = chords_data.get("bpm", 120.0)  # Read BPM from cache
                chords = [ChordEvent(**c) for c in chords_data.get("chords", [])]

            # Load melody notes from MIDI
            import pretty_midi

            midi = pretty_midi.PrettyMIDI(str(output_dir / "melody.mid"))
            notes = []
            for instrument in midi.instruments:
                for n in instrument.notes:
                    notes.append(MelodyNote(
                        pitch=n.pitch,  # MIDI note number
                        start_time=n.start,
                        duration=n.end - n.start,
                        velocity=n.velocity
                    ))
            
            # Get title
            title = "Restored Job"
            url_file = output_dir / "url.txt"
            if url_file.exists():
                title = url_file.read_text().strip()
            
            # Generate MusicXML
            musicxml = generate_musicxml(notes, chords, bpm, key, title)
            xml_path = output_dir / "lead_sheet.musicxml"
            xml_path.write_text(musicxml)
            has_xml = True
            print(f"Regenerated MusicXML for job {job_id}")
        except Exception as e:
            print(f"Failed to regenerate MusicXML: {e}")
    
    # Determine the step based on what's available
    if has_xml:
        step = JobStep.COMPLETED
        progress = 100
        message = "✓ Restored from cache - Lead sheet complete!"
    elif has_chords:
        step = JobStep.CHORDS_DETECTED
        progress = 80
        message = "Restored from cache - Ready to generate lead sheet"
    elif has_midi:
        step = JobStep.TRANSCRIBED
        progress = 60
        message = "Restored from cache - Ready to detect chords"
    elif has_vocals:
        step = JobStep.SEPARATED
        progress = 40
        message = "Restored from cache - Ready to transcribe"
    else:
        raise HTTPException(status_code=404, detail="Not enough cached data to restore")

    # Load chords if available
    chords = None
    key = None
    if has_chords:
        with open(output_dir / "chords.json") as f:
            chords_data = json.load(f)
            key = chords_data.get("key")
            chords = [ChordEvent(**c) for c in chords_data.get("chords", [])]

    # Get title from URL file if available
    title = "Restored Job"
    url_file = output_dir / "url.txt"
    if url_file.exists():
        title = url_file.read_text().strip()

    # Get duration from audio if available
    duration = None
    if has_original:
        duration = librosa.get_duration(path=str(output_dir / "original.wav"))

    # BPM is read from chords.json (stored from user input)
    # If no chords.json, use default
    bpm = None
    if has_chords:
        try:
            with open(output_dir / "chords.json") as f:
                chords_data = json.load(f)
                bpm = chords_data.get("bpm", 120.0)
        except Exception:
            bpm = 120.0

    # Create job response
    jobs[job_id] = JobResponse(
        job_id=job_id,
        step=step,
        progress=progress,
        message=message,
        title=title,
        duration=duration,
        bpm=bpm,
        key=key,
        vocals_url=f"/api/audio/{job_id}/vocals" if has_vocals else None,
        accompaniment_url=f"/api/audio/{job_id}/accompaniment"
        if has_accompaniment
        else None,
        original_url=f"/api/audio/{job_id}/original" if has_original else None,
        melody_midi_url=f"/api/audio/{job_id}/melody_midi" if has_midi else None,
        chords=chords,
        music_xml_url=f"/api/musicxml/{job_id}" if has_xml else None,
    )

    return jobs[job_id]


@app.get("/api/jobs/cached")
async def list_cached_jobs():
    """List all job directories with cached data."""
    cached = []
    for job_dir in OUTPUTS_DIR.iterdir():
        if job_dir.is_dir():
            job_id = job_dir.name
            has_xml = (job_dir / "lead_sheet.musicxml").exists()
            has_vocals = (job_dir / "htdemucs" / "original" / "vocals.wav").exists()

            # Get title
            title = job_id
            url_file = job_dir / "url.txt"
            if url_file.exists():
                title = url_file.read_text().strip()[:50]

            cached.append(
                {
                    "job_id": job_id,
                    "title": title,
                    "has_musicxml": has_xml,
                    "has_stems": has_vocals,
                }
            )
    return cached


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
