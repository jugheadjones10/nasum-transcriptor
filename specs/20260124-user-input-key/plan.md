# Implementation Plan: User Input for Key and BPM

**Feature**: `20260124-user-input-key` | **Date**: 2026-01-24 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/20260124-user-input-key/spec.md`

## Summary

Replace automatic key and BPM detection with user-provided inputs. Add frontend UI components for key selection (12 notes × major/minor) and numeric BPM input. Modify backend API to accept these parameters and remove automatic detection code. This simplifies the processing pipeline and gives users direct control over musical parameters.

## Technical Context

**Language/Version**: Python 3.11 (backend), TypeScript 5.x (frontend)
**Primary Dependencies**: FastAPI, React 19, Vite, Tailwind CSS, music21
**Storage**: File-based job persistence (outputs/ directory)
**Testing**: Manual testing (visual UI), integration tests for API
**Target Platform**: Web application (localhost development)
**Project Type**: web (frontend + backend)
**Performance Goals**: Faster processing by removing detection steps
**Constraints**: Must maintain existing pipeline architecture
**Scale/Scope**: Single-user local application

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Pipeline-First Architecture | ✅ PASS | User inputs key/BPM at start, flows through pipeline unchanged |
| II. User-Controlled Progression | ✅ PASS | User provides inputs before processing starts |
| III. Separation of Concerns | ✅ PASS | Frontend handles UI, backend handles processing via REST API |
| IV. Graceful Degradation | ✅ PASS | Validation prevents invalid inputs; cached jobs still work |
| V. Simplicity Over Configurability | ✅ PASS | Key/BPM are essential inputs, not optional configuration |

**Gate Result**: PASS - All principles satisfied

## Project Structure

### Documentation (this feature)

```
specs/20260124-user-input-key/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── contracts/           # Phase 1 output
    └── api.md           # API contract changes
```

### Source Code (repository root)

```
backend/
├── main.py              # FastAPI app - modify API endpoints
└── pyproject.toml       # Dependencies - remove essentia

frontend/
└── src/
    └── App.tsx          # React app - add key/BPM inputs
```

**Structure Decision**: Web application with separate frontend and backend. Changes affect both layers but maintain existing architecture.

## Implementation Approach

### Phase 1: Frontend UI Changes

**File**: `frontend/src/App.tsx`

1. Add state for key and BPM:
   ```typescript
   const [selectedKey, setSelectedKey] = useState<string>("C");
   const [selectedMode, setSelectedMode] = useState<string>("major");
   const [bpm, setBpm] = useState<number>(120);
   ```

2. Add key selection dropdowns:
   - Root note: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
   - Mode: major, minor

3. Add BPM input:
   - `<input type="number" min="20" max="300" />`

4. Update form submission to include key and BPM in API request

### Phase 2: Backend API Changes

**File**: `backend/main.py`

1. Modify `/api/process-song` endpoint:
   - Add `key: str` parameter (e.g., "C major")
   - Add `bpm: int` parameter
   - Store in job immediately instead of detecting later

2. Remove automatic detection:
   - Remove `detect_key()` function
   - Remove `detect_beats_and_tempo()` function
   - Remove calls in `run_step1()` and `run_step3()`

3. Update `restore_job()`:
   - Read key/BPM from cached job data
   - Don't recalculate

### Phase 3: Cleanup

**File**: `backend/pyproject.toml`

1. Remove `essentia` dependency (no longer needed for detection)

## Files to Modify

| File | Action | Changes |
|------|--------|---------|
| `frontend/src/App.tsx` | Modify | Add key selector, mode selector, BPM input |
| `backend/main.py` | Modify | Add API params, remove detection functions |
| `backend/pyproject.toml` | Modify | Remove essentia dependency |

## Complexity Tracking

*No violations - feature simplifies the codebase by removing detection code*
