# Quickstart: User Input for Key and BPM

**Feature**: `20260124-user-input-key`
**Date**: 2026-01-24

## Prerequisites

- Backend running: `cd backend && uv run python main.py`
- Frontend running: `cd frontend && npm run dev`

## Testing the Feature

### 1. Open the Application

Navigate to `http://localhost:5173` in your browser.

### 2. Enter Song Details

1. **YouTube URL**: Paste a YouTube video URL
2. **Key Selection**:
   - Select the root note (C, C#, D, etc.)
   - Select the mode (major or minor)
3. **BPM Input**:
   - Enter a number between 20-300
   - The field only accepts numeric input

### 3. Start Processing

Click "Extract" to begin processing. The key and BPM you entered will be used for the lead sheet.

### 4. Verify Results

After processing completes:
1. View the generated sheet music
2. Verify the key signature matches your selection
3. Verify the tempo marking matches your BPM input

## Test Cases

### Test Case 1: Key Selection

| Input | Expected Result |
|-------|-----------------|
| Key: G major, BPM: 120 | Lead sheet shows 1 sharp (F#), tempo = 120 |
| Key: F major, BPM: 90 | Lead sheet shows 1 flat (Bb), tempo = 90 |
| Key: A minor, BPM: 140 | Lead sheet shows no sharps/flats, tempo = 140 |

### Test Case 2: BPM Validation

| Input | Expected Result |
|-------|-----------------|
| BPM: 120 | Accepted |
| BPM: abc | Rejected (non-numeric) |
| BPM: -10 | Rejected (below minimum) |
| BPM: 500 | Rejected (above maximum) |

### Test Case 3: Required Fields

| Scenario | Expected Result |
|----------|-----------------|
| No key selected | Form submission blocked |
| No BPM entered | Form submission blocked |
| All fields filled | Form submits successfully |

## Troubleshooting

### "Invalid key" error
- Ensure you selected both a root note AND a mode (major/minor)
- Key format must be: `{note} {mode}` (e.g., "C major")

### "Invalid BPM" error
- Ensure BPM is a number between 20 and 300
- Remove any non-numeric characters

### Sheet music shows wrong key
- Verify the key you selected matches the song
- Check that the backend received the correct key parameter
