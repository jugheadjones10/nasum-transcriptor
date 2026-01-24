# Data Model: User Input for Key and BPM

**Feature**: `20260124-user-input-key`
**Date**: 2026-01-24

## Entities

### ProcessSongRequest (Modified)

Request body for starting a new transcription job.

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| url | string | Yes | Valid YouTube URL | YouTube video URL |
| key | string | Yes | One of 24 valid keys | Musical key (e.g., "C major") |
| bpm | integer | Yes | 20-300 | Tempo in beats per minute |

**Valid Key Values** (24 total):
```
C major, C minor, C# major, C# minor,
D major, D minor, D# major, D# minor,
E major, E minor,
F major, F minor, F# major, F# minor,
G major, G minor, G# major, G# minor,
A major, A minor, A# major, A# minor,
B major, B minor
```

### JobResponse (Modified)

Response containing job status and results.

| Field | Type | Change | Description |
|-------|------|--------|-------------|
| key | string | Now user-provided | Musical key from user input |
| bpm | float | Now user-provided | BPM from user input |

*All other fields remain unchanged*

## State Transitions

No changes to job state machine. Key and BPM are set at job creation and remain constant throughout processing.

```
IDLE → DOWNLOADING → SEPARATING → TRANSCRIBING → COMPLETED
       ↑                                              ↓
       └──────────── (key, bpm set here) ────────────┘
```

## Validation Rules

### Key Validation
- Must be a string
- Must match pattern: `{note} {mode}` where:
  - note ∈ {C, C#, D, D#, E, F, F#, G, G#, A, A#, B}
  - mode ∈ {major, minor}

### BPM Validation
- Must be an integer
- Must be ≥ 20
- Must be ≤ 300

## Backward Compatibility

### Cached Jobs
- Existing cached jobs have key/BPM stored in `chords.json`
- `restore_job()` will read these values from cache
- No migration needed - cached values are preserved

### API Changes
- `/api/process-song` now requires `key` and `bpm` parameters
- Frontend must be updated to send these parameters
- Breaking change for any direct API consumers
