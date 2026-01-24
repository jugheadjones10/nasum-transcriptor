# Tasks: User Input for Key and BPM

**Input**: Design documents from `/specs/20260124-user-input-key/`
**Prerequisites**: plan.md (required), spec.md (required), data-model.md, contracts/api.md

**Tests**: Not explicitly requested - no test tasks included.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions
- **Web app**: `backend/`, `frontend/src/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: No setup needed - modifying existing project

*This feature modifies an existing codebase. No new project initialization required.*

**Checkpoint**: Ready to proceed to user story implementation

---

## Phase 2: User Story 1+2 - Key and BPM Input (Priority: P1) 🎯 MVP

**Goal**: Add frontend UI for key selection (12 notes × major/minor) and numeric BPM input, update backend API to accept these parameters

**Independent Test**: Enter a YouTube URL, select "G major" key and "120" BPM, process the song, verify the generated MusicXML has G major key signature and 120 BPM tempo marking

*Note: US1 (Key Selection) and US2 (BPM Input) are combined because they modify the same files and are both P1 priority.*

### Implementation

- [x] T001 [P] [US1+2] Add key and BPM state variables in `frontend/src/App.tsx`
  - Add `selectedNote` state (default: "C")
  - Add `selectedMode` state (default: "major")
  - Add `bpm` state (default: 120)

- [x] T002 [P] [US1+2] Create key selection dropdown component in `frontend/src/App.tsx`
  - Root note dropdown: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
  - Mode dropdown: major, minor
  - Style with Tailwind CSS to match existing UI

- [x] T003 [P] [US1+2] Create BPM numeric input in `frontend/src/App.tsx`
  - Input type="number"
  - min="20" max="300"
  - Placeholder "120"
  - Style with Tailwind CSS to match existing UI

- [x] T004 [US1+2] Update form submission to include key and BPM in `frontend/src/App.tsx`
  - Modify fetch call to `/api/process-song`
  - Add `key: \`${selectedNote} ${selectedMode}\`` to request body
  - Add `bpm: parseInt(bpm)` to request body
  - Add validation: require both key and BPM before submission

- [x] T005 [US1+2] Add key and bpm parameters to `/api/process-song` endpoint in `backend/main.py`
  - Create Pydantic model for request body with url, key, bpm fields
  - Add key validation: must be one of 24 valid keys
  - Add bpm validation: must be integer 20-300
  - Store key and bpm in job immediately at creation

- [x] T006 [US1+2] Update `run_step1()` to use user-provided BPM in `backend/main.py`
  - Remove call to `detect_beats_and_tempo()`
  - Use `jobs[job_id].bpm` (already set from user input)

- [x] T007 [US1+2] Update `run_step3()` to use user-provided key in `backend/main.py`
  - Remove call to `detect_key()`
  - Use `jobs[job_id].key` (already set from user input)

**Checkpoint**: At this point, users can input key and BPM, and the lead sheet uses those values

---

## Phase 3: User Story 3 - Remove Automatic Detection (Priority: P2)

**Goal**: Remove automatic key and BPM detection code, clean up dependencies

**Independent Test**: Process a song and verify no detection functions are called, processing completes faster

### Implementation

- [x] T008 [US3] Remove `detect_key()` function from `backend/main.py`
  - Delete the function definition
  - Delete any imports only used by this function

- [x] T009 [US3] Remove `detect_beats_and_tempo()` function from `backend/main.py`
  - Delete the function definition
  - Delete the `get_*_processor()` helper functions
  - Delete `_madmom_processors` cache variable

- [x] T010 [US3] Update `restore_job()` to not recalculate key/BPM in `backend/main.py`
  - Remove calls to `detect_beats_and_tempo()` in restore logic
  - Read key/BPM from cached job data (chords.json) instead

- [x] T011 [US3] Remove essentia dependency from `backend/pyproject.toml`
  - Remove `"essentia"` from dependencies list
  - Run `uv sync` to update lock file

**Checkpoint**: Automatic detection code removed, application still works with user-provided values

---

## Phase 4: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and cleanup

- [x] T012 Validate end-to-end flow per `quickstart.md`
  - Test key selection with all 12 notes
  - Test major and minor modes
  - Test BPM validation (reject non-numeric, out of range)
  - Verify MusicXML output matches inputs

- [x] T013 Test backward compatibility with cached jobs
  - Restore a previously cached job
  - Verify key/BPM values load from cache correctly

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: Skipped - no setup needed
- **Phase 2 (US1+2)**: Can start immediately - MVP implementation
- **Phase 3 (US3)**: Depends on Phase 2 completion - cleanup after main features work
- **Phase 4 (Polish)**: Depends on Phase 3 completion - final validation

### Task Dependencies Within Phase 2

```
T001, T002, T003 (parallel - different concerns)
    ↓
T004 (depends on T001-T003 - integrates UI components)
    ↓
T005 (backend API - can start after T004 or in parallel)
    ↓
T006, T007 (sequential in same file - use user-provided values)
```

### Parallel Opportunities

**Phase 2 - Frontend tasks can run in parallel:**
```
T001: Add state variables
T002: Create key dropdown
T003: Create BPM input
```

**Phase 3 - Backend cleanup tasks are sequential (same file):**
```
T008 → T009 → T010 → T011
```

---

## Implementation Strategy

### MVP First (Phase 2 Only)

1. Complete T001-T007 (User Story 1+2)
2. **STOP and VALIDATE**: Test with a real YouTube video
3. Verify key signature and tempo in generated MusicXML
4. Demo if ready

### Full Implementation

1. Complete Phase 2 (MVP)
2. Complete Phase 3 (Cleanup)
3. Complete Phase 4 (Validation)
4. Ready for production use

---

## Notes

- T001, T002, T003 can run in parallel (different UI components)
- T005-T007 are sequential (same file, dependent changes)
- T008-T010 are sequential (same file, removing related code)
- No test tasks included - tests not explicitly requested
- Commit after each task or logical group
- Stop at any checkpoint to validate independently
