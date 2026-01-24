# Feature Specification: User Input for Key and BPM

**Feature**: `20260124-user-input-key`
**Created**: 2026-01-24
**Status**: Draft
**Input**: User description: "Change of plan, the user will input the key and the bpm so we don't need to calculate for that. but the Transcribed lead sheet should be according to the key that user inputed, there should be an input that allows all the keys and whether its a major or a minor. The bpm input box should only allow numbers."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Key Selection Input (Priority: P1)

As a musician using Nasum Transcriptor, I want to manually select the musical key (including all 12 notes and major/minor mode) for my transcription, so that the generated lead sheet uses the correct key signature that I know is accurate for my song.

**Why this priority**: This is the core requirement - removing automatic key detection and replacing it with user input. The transcribed lead sheet must be according to the key that the user inputs.

**Technical Implementation**:

**Frontend Changes** (`frontend/src/App.tsx`):
- Add a key selection UI component with:
  - A dropdown/select for the root note: C, C#/Db, D, D#/Eb, E, F, F#/Gb, G, G#/Ab, A, A#/Bb, B
  - A dropdown/select for the mode: major, minor
- Display this input before the user starts processing or during the initial form
- The selected key should be sent to the backend with the processing request

**Backend Changes** (`backend/main.py`):
- Modify the `/api/process-song` endpoint to accept `key` parameter (e.g., "C major", "A minor")
- Remove the automatic key detection call (`detect_key()` function)
- Pass the user-provided key directly to `generate_musicxml()`
- Update `JobResponse` model if needed to store user-provided key

**Key Values to Support**:
All 12 chromatic notes × 2 modes = 24 possible keys:
- C major, C minor
- C#/Db major, C#/Db minor
- D major, D minor
- D#/Eb major, D#/Eb minor
- E major, E minor
- F major, F minor
- F#/Gb major, F#/Gb minor
- G major, G minor
- G#/Ab major, G#/Ab minor
- A major, A minor
- A#/Bb major, A#/Bb minor
- B major, B minor

**Independent Test**: Can be fully tested by selecting different keys in the UI and verifying the generated MusicXML contains the correct key signature.

**Acceptance Scenarios**:

1. **Given** the user is on the transcription form, **When** they open the key selector, **Then** they see all 12 root notes available for selection
2. **Given** the user has selected a root note, **When** they select the mode, **Then** they can choose between "major" and "minor"
3. **Given** the user selects "G major", **When** the lead sheet is generated, **Then** the MusicXML contains a G major key signature (1 sharp)

---

### User Story 2 - BPM Number Input (Priority: P1)

As a musician using Nasum Transcriptor, I want to manually enter the BPM (tempo) as a number, so that the generated lead sheet has the correct tempo marking and note quantization.

**Why this priority**: This is equally critical - the BPM input box should only allow numbers, ensuring valid tempo values for accurate note timing.

**Technical Implementation**:

**Frontend Changes** (`frontend/src/App.tsx`):
- Add a BPM input field with:
  - Input type="number" to only allow numeric input
  - Minimum value constraint (e.g., 20 BPM)
  - Maximum value constraint (e.g., 300 BPM)
  - Placeholder text showing expected format (e.g., "120")
- Validate that the input is a valid number before submission
- Display this input alongside the key selection

**Backend Changes** (`backend/main.py`):
- Modify the `/api/process-song` endpoint to accept `bpm` parameter (number)
- Remove the automatic BPM detection call (`detect_beats_and_tempo()` function)
- Pass the user-provided BPM directly to chord detection and MusicXML generation
- Update `JobResponse` model to use user-provided BPM

**Validation Rules**:
- Must be a positive number
- Must be numeric only (no letters or special characters)
- Reasonable range: 20-300 BPM (industry standard tempo range)

**Independent Test**: Can be fully tested by entering various BPM values and verifying the generated MusicXML contains the correct tempo marking.

**Acceptance Scenarios**:

1. **Given** the user is on the transcription form, **When** they see the BPM input field, **Then** it only accepts numeric input
2. **Given** the user tries to type letters in the BPM field, **When** they press keys, **Then** non-numeric characters are not entered
3. **Given** the user enters "120" as BPM, **When** the lead sheet is generated, **Then** the MusicXML contains tempo marking of 120 BPM

---

### User Story 3 - Remove Automatic Detection (Priority: P2)

As a developer, I need to remove the automatic key and BPM detection code since users will now provide these values manually, simplifying the processing pipeline.

**Why this priority**: This is a cleanup task that follows from the main features. It removes unnecessary computation and the essentia dependency.

**Technical Implementation**:

**Backend Changes** (`backend/main.py`):
- Remove or comment out `detect_key()` function
- Remove or comment out `detect_beats_and_tempo()` function
- Remove calls to these functions in `run_step1()` and `run_step3()`
- Update `restore_job()` to not recalculate key/BPM

**Dependency Changes** (`backend/pyproject.toml`):
- Optionally remove `essentia` dependency if no longer needed for other features

**Independent Test**: Can be verified by confirming the application works without calling detection functions and that processing time is reduced.

**Acceptance Scenarios**:

1. **Given** the user submits a song for processing, **When** step 1 completes, **Then** no automatic BPM detection is performed
2. **Given** the user proceeds to step 3, **When** chord detection runs, **Then** no automatic key detection is performed

---

### Edge Cases

- What happens when user enters BPM outside valid range (e.g., 0, -10, 500)?
  - Show validation error, prevent form submission
- What happens when user doesn't select a key?
  - Require key selection before processing can start
- What happens when user enters decimal BPM (e.g., 120.5)?
  - Accept and round to nearest integer, or accept decimals if supported
- What happens when restoring a cached job that was processed with old automatic detection?
  - Use the stored key/BPM values from the cache

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a dropdown/select input for selecting the root note of the key (all 12 chromatic notes)
- **FR-002**: System MUST provide a dropdown/select input for selecting the mode (major or minor)
- **FR-003**: System MUST provide a numeric input field for BPM that only allows numbers
- **FR-004**: System MUST validate BPM input to ensure it is a positive number
- **FR-005**: System MUST use the user-provided key when generating the MusicXML lead sheet
- **FR-006**: System MUST use the user-provided BPM for tempo marking and note quantization
- **FR-007**: System MUST NOT perform automatic key detection
- **FR-008**: System MUST NOT perform automatic BPM/beat detection
- **FR-009**: System MUST require both key and BPM inputs before starting transcription

### Key Entities

- **Key**: Musical key consisting of root note (C, C#, D, etc.) and mode (major/minor)
- **BPM**: Tempo in beats per minute, numeric value typically 20-300

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can select any of the 24 standard musical keys (12 notes × 2 modes)
- **SC-002**: Users can only enter numeric values in the BPM field
- **SC-003**: Generated lead sheets display the exact key signature selected by the user
- **SC-004**: Generated lead sheets display the exact tempo (BPM) entered by the user
- **SC-005**: Processing time is reduced by eliminating automatic detection steps
