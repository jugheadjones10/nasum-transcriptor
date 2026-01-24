# API Contract: User Input for Key and BPM

**Feature**: `20260124-user-input-key`
**Date**: 2026-01-24

## Modified Endpoints

### POST /api/process-song

Start a new transcription job with user-provided key and BPM.

**Request Body** (JSON):

```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "key": "G major",
  "bpm": 113
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| url | string | Yes | YouTube video URL |
| key | string | Yes | Musical key (e.g., "C major", "A minor") |
| bpm | integer | Yes | Tempo in beats per minute (20-300) |

**Response** (200 OK):

```json
{
  "job_id": "abc123",
  "step": "downloading",
  "progress": 0,
  "message": "Starting download...",
  "key": "G major",
  "bpm": 113
}
```

**Error Responses**:

- `400 Bad Request`: Invalid key or BPM value
  ```json
  {
    "detail": "Invalid key: must be one of C, C#, D, D#, E, F, F#, G, G#, A, A#, B followed by major or minor"
  }
  ```

- `400 Bad Request`: BPM out of range
  ```json
  {
    "detail": "Invalid BPM: must be between 20 and 300"
  }
  ```

## Unchanged Endpoints

The following endpoints remain unchanged:

- `GET /api/job/{job_id}` - Get job status
- `POST /api/job/{job_id}/continue` - Continue to next step
- `GET /api/audio/{job_id}/{stem}` - Download audio files
- `GET /api/musicxml/{job_id}` - Download MusicXML
- `POST /api/job/{job_id}/restore` - Restore cached job
- `GET /api/jobs/cached` - List cached jobs

## Valid Key Values

```
C major    C minor
C# major   C# minor
D major    D minor
D# major   D# minor
E major    E minor
F major    F minor
F# major   F# minor
G major    G minor
G# major   G# minor
A major    A minor
A# major   A# minor
B major    B minor
```

## Frontend Integration

The frontend must update its API call to include key and BPM:

```typescript
// Before
const response = await fetch('/api/process-song', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ url: youtubeUrl })
});

// After
const response = await fetch('/api/process-song', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    url: youtubeUrl,
    key: `${selectedNote} ${selectedMode}`,  // e.g., "G major"
    bpm: parseInt(bpmInput)                   // e.g., 113
  })
});
```
