# Research: User Input for Key and BPM

**Feature**: `20260124-user-input-key`
**Date**: 2026-01-24

## Research Questions

### Q1: What are the standard musical key representations?

**Decision**: Use standard note names with sharps only (no flats in dropdown)

**Rationale**:
- Sharps are more common in music notation software
- Simplifies dropdown to exactly 12 options
- Backend can handle conversion if needed for display

**Alternatives Considered**:
- Include both sharps and flats (C#/Db) - rejected as overly complex for dropdown
- Use only flats - rejected as less common convention

**Key Values**:
```
C, C#, D, D#, E, F, F#, G, G#, A, A#, B
```

### Q2: What BPM range should be supported?

**Decision**: 20-300 BPM range with integer values

**Rationale**:
- 20 BPM covers extremely slow pieces (largo)
- 300 BPM covers extremely fast pieces (prestissimo)
- Integer values are sufficient for practical use
- Matches industry standard tempo ranges

**Alternatives Considered**:
- 40-200 BPM - rejected as too restrictive for edge cases
- Decimal BPM (120.5) - rejected as unnecessary precision

### Q3: How should key/BPM be passed to the API?

**Decision**: Pass as separate query parameters in POST request body

**Rationale**:
- Consistent with existing API pattern
- Easy to validate on backend
- Clear separation of parameters

**API Format**:
```json
{
  "url": "https://youtube.com/...",
  "key": "C major",
  "bpm": 120
}
```

### Q4: Should essentia dependency be removed?

**Decision**: Yes, remove essentia dependency

**Rationale**:
- No longer needed for key/beat detection
- Reduces installation complexity
- Faster dependency resolution
- Smaller deployment footprint

**Alternatives Considered**:
- Keep for potential future use - rejected as YAGNI (You Aren't Gonna Need It)

## Resolved Clarifications

All technical questions resolved. No NEEDS CLARIFICATION items remain.
