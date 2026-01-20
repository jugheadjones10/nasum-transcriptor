import { useState, useEffect, useRef, useCallback } from 'react';
import { OpenSheetMusicDisplay } from 'opensheetmusicdisplay';
import { 
  Music, Loader2, Play, Pause, Download, AlertCircle, CheckCircle2, 
  Mic2, Guitar, ChevronRight, Piano, FileMusic, Waves, Volume2 
} from 'lucide-react';

// =============================================================================
// Types
// =============================================================================

type JobStep = 
  | 'idle'
  | 'downloading'
  | 'separating'
  | 'separated'
  | 'transcribing'
  | 'transcribed'
  | 'detecting_chords'
  | 'chords_detected'
  | 'generating'
  | 'completed'
  | 'failed';

interface MelodyNote {
  pitch: number;
  start_time: number;
  duration: number;
  velocity: number;
}

interface ChordEvent {
  time: number;
  chord: string;
  confidence: number;
}

interface Job {
  job_id: string;
  step: JobStep;
  progress: number;
  message: string;
  title?: string;
  duration?: number;
  bpm?: number;
  key?: string;
  time_signature?: string;
  vocals_url?: string;
  accompaniment_url?: string;
  original_url?: string;
  melody_notes?: MelodyNote[];
  melody_midi_url?: string;
  chords?: ChordEvent[];
  music_xml_url?: string;
  error?: string;
}

const API_BASE = 'http://localhost:8000';

// =============================================================================
// Step Configuration
// =============================================================================

const STEPS = [
  { id: 'separate', label: 'Separate', icon: Waves, waitStep: 'separated' },
  { id: 'transcribe', label: 'Transcribe', icon: FileMusic, waitStep: 'transcribed' },
  { id: 'chords', label: 'Chords', icon: Piano, waitStep: 'chords_detected' },
  { id: 'generate', label: 'Lead Sheet', icon: Music, waitStep: 'completed' },
] as const;

function getStepIndex(step: JobStep): number {
  if (['idle', 'downloading', 'separating'].includes(step)) return 0;
  if (['separated', 'transcribing'].includes(step)) return 1;
  if (['transcribed', 'detecting_chords'].includes(step)) return 2;
  if (['chords_detected', 'generating'].includes(step)) return 3;
  if (step === 'completed') return 4;
  return -1;
}

function isWaitingForUser(step: JobStep): boolean {
  return ['separated', 'transcribed', 'chords_detected'].includes(step);
}

// =============================================================================
// Utility Functions
// =============================================================================

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

function midiToNoteName(midi: number): string {
  const octave = Math.floor(midi / 12) - 1;
  const note = NOTE_NAMES[midi % 12];
  return `${note}${octave}`;
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// =============================================================================
// Components
// =============================================================================

function AudioPlayer({ label, url, icon: Icon }: { label: string; url: string; icon: typeof Mic2 }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleTimeUpdate = () => {
    if (!audioRef.current) return;
    const pct = (audioRef.current.currentTime / audioRef.current.duration) * 100;
    setProgress(pct);
    setCurrentTime(audioRef.current.currentTime);
  };

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration);
    }
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!audioRef.current) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    audioRef.current.currentTime = pct * audioRef.current.duration;
  };

  return (
    <div className="p-4 bg-[var(--color-surface)] rounded-xl border border-[var(--color-border)]">
      <div className="flex items-center gap-3 mb-3">
        <button
          onClick={togglePlay}
          className="w-10 h-10 flex items-center justify-center rounded-full bg-[var(--color-accent)] text-[var(--color-bg)] hover:bg-[var(--color-accent-hover)] transition-colors shrink-0"
        >
          {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
        </button>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <Icon className="w-4 h-4 text-[var(--color-accent)]" />
            <span className="font-medium text-sm">{label}</span>
          </div>
          <div className="text-xs text-[var(--color-text-muted)]">
            {formatTime(currentTime)} / {formatTime(duration)}
          </div>
        </div>
      </div>
      <div 
        className="h-2 bg-[var(--color-surface-elevated)] rounded-full overflow-hidden cursor-pointer"
        onClick={handleSeek}
      >
        <div 
          className="h-full bg-[var(--color-accent)] transition-all duration-100"
          style={{ width: `${progress}%` }}
        />
      </div>
      <audio
        ref={audioRef}
        src={url}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={() => setIsPlaying(false)}
      />
    </div>
  );
}

function StepIndicator({ currentStep, isProcessing }: { currentStep: JobStep; isProcessing: boolean }) {
  const stepIndex = getStepIndex(currentStep);
  
  return (
    <div className="flex items-center justify-between mb-8">
      {STEPS.map((step, i) => {
        const Icon = step.icon;
        const isActive = i === stepIndex || (i === stepIndex - 1 && isWaitingForUser(currentStep));
        const isComplete = i < stepIndex || (i === stepIndex && currentStep === 'completed');
        const isWaiting = step.waitStep === currentStep;
        
        return (
          <div key={step.id} className="flex items-center">
            <div className="flex flex-col items-center">
              <div 
                className={`w-12 h-12 rounded-xl flex items-center justify-center transition-all ${
                  isComplete 
                    ? 'bg-green-500/20 text-green-400' 
                    : isWaiting
                    ? 'bg-[var(--color-accent)] text-[var(--color-bg)] ring-4 ring-[var(--color-accent)]/30'
                    : isActive
                    ? 'bg-[var(--color-accent)]/20 text-[var(--color-accent)]'
                    : 'bg-[var(--color-surface)] text-[var(--color-text-muted)]'
                }`}
              >
                {isComplete ? (
                  <CheckCircle2 className="w-6 h-6" />
                ) : isActive && isProcessing && !isWaiting ? (
                  <Loader2 className="w-6 h-6 animate-spin" />
                ) : (
                  <Icon className="w-6 h-6" />
                )}
              </div>
              <span className={`text-xs mt-2 font-medium ${
                isComplete || isWaiting ? 'text-[var(--color-text)]' : 'text-[var(--color-text-muted)]'
              }`}>
                {step.label}
              </span>
            </div>
            {i < STEPS.length - 1 && (
              <ChevronRight className={`w-5 h-5 mx-3 ${
                i < stepIndex ? 'text-green-400' : 'text-[var(--color-border)]'
              }`} />
            )}
          </div>
        );
      })}
    </div>
  );
}

function MelodyPreview({ notes, bpm }: { notes: MelodyNote[]; bpm: number }) {
  const [showAll, setShowAll] = useState(false);
  const displayNotes = showAll ? notes : notes.slice(0, 20);

  return (
    <div className="p-4 bg-[var(--color-surface)] rounded-xl border border-[var(--color-border)]">
      <h4 className="font-medium mb-3 flex items-center gap-2">
        <FileMusic className="w-4 h-4 text-[var(--color-accent)]" />
        Transcribed Melody ({notes.length} notes)
      </h4>
      <div className="flex flex-wrap gap-1.5 max-h-40 overflow-y-auto">
        {displayNotes.map((note, i) => (
          <div
            key={i}
            className="px-2 py-1 bg-[var(--color-surface-elevated)] rounded text-xs font-mono"
            title={`Start: ${note.start_time.toFixed(2)}s, Duration: ${note.duration.toFixed(2)}s`}
          >
            {midiToNoteName(note.pitch)}
          </div>
        ))}
        {!showAll && notes.length > 20 && (
          <button
            onClick={() => setShowAll(true)}
            className="px-2 py-1 bg-[var(--color-accent)]/20 text-[var(--color-accent)] rounded text-xs hover:bg-[var(--color-accent)]/30"
          >
            +{notes.length - 20} more
        </button>
        )}
      </div>
    </div>
  );
}

function ChordsPreview({ chords, duration }: { chords: ChordEvent[]; duration: number }) {
  return (
    <div className="p-4 bg-[var(--color-surface)] rounded-xl border border-[var(--color-border)]">
      <h4 className="font-medium mb-3 flex items-center gap-2">
        <Piano className="w-4 h-4 text-[var(--color-accent)]" />
        Chord Progression ({chords.length} changes)
      </h4>
      <div className="relative h-16 bg-[var(--color-surface-elevated)] rounded-lg overflow-hidden">
        {chords.map((chord, i) => {
          const nextChord = chords[i + 1];
          const startPct = (chord.time / duration) * 100;
          const endPct = nextChord ? (nextChord.time / duration) * 100 : 100;
          const width = endPct - startPct;
          
          return (
            <div
              key={i}
              className="absolute top-0 h-full flex items-center justify-center border-r border-[var(--color-border)]"
              style={{ left: `${startPct}%`, width: `${width}%` }}
            >
              <span className={`text-sm font-bold px-1 truncate ${
                width < 5 ? 'text-[8px]' : width < 10 ? 'text-xs' : ''
              }`}>
                {chord.chord}
              </span>
            </div>
          );
        })}
      </div>
      <div className="flex justify-between text-xs text-[var(--color-text-muted)] mt-1">
        <span>0:00</span>
        <span>{formatTime(duration)}</span>
      </div>
    </div>
  );
}

function SheetMusicRenderer({ musicXmlUrl }: { musicXmlUrl: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const osmdRef = useRef<OpenSheetMusicDisplay | null>(null);
  const [status, setStatus] = useState<'loading' | 'rendering' | 'done' | 'error'>('loading');
  const [error, setError] = useState<string | null>(null);
  const [xmlData, setXmlData] = useState<string | null>(null);

  // Step 1: Fetch the XML data
  useEffect(() => {
    let cancelled = false;

    const fetchXml = async () => {
      setStatus('loading');
      setError(null);
      
      try {
        const response = await fetch(musicXmlUrl);
        if (!response.ok) throw new Error('Failed to fetch MusicXML');
        const xml = await response.text();
        
        if (cancelled) return;
        setXmlData(xml);
        setStatus('rendering');
      } catch (err) {
        if (cancelled) return;
        console.error('Load error:', err);
        setError(err instanceof Error ? err.message : 'Failed to load sheet music');
        setStatus('error');
      }
    };

    fetchXml();
    
    return () => {
      cancelled = true;
    };
  }, [musicXmlUrl]);

  // Step 2: Render when container is ready and XML is loaded
  useEffect(() => {
    if (status !== 'rendering' || !xmlData || !containerRef.current) return;

    let cancelled = false;

    const renderSheet = async () => {
      try {
        // Create OSMD with optimized settings for large scores
        if (!osmdRef.current) {
          osmdRef.current = new OpenSheetMusicDisplay(containerRef.current!, {
            autoResize: true,
            drawTitle: true,
            drawComposer: false,
            drawingParameters: "compacttight",
            drawPartNames: false,
            drawMeasureNumbers: true,
            drawMetronomeMarks: false,
          });
        }
        
        await osmdRef.current.load(xmlData);
        
        if (cancelled) return;
        
        // Use requestAnimationFrame to allow UI to update before heavy render
        requestAnimationFrame(() => {
          if (cancelled || !osmdRef.current) return;
          try {
            osmdRef.current.render();
            setStatus('done');
          } catch (renderErr) {
            console.error('Render error:', renderErr);
            setError('Failed to render sheet music');
            setStatus('error');
          }
        });
      } catch (err) {
        if (cancelled) return;
        console.error('Render error:', err);
        setError(err instanceof Error ? err.message : 'Failed to render sheet music');
        setStatus('error');
      }
    };

    renderSheet();
    
    return () => {
      cancelled = true;
    };
  }, [status, xmlData]);

  return (
    <div className="relative">
      {/* OSMD container - must be visible for OSMD to calculate widths */}
      <div 
        ref={containerRef} 
        className="osmd-container bg-white rounded-xl p-4 text-black overflow-x-auto"
        style={{ minHeight: '300px' }}
      />

      {/* Loading/rendering overlay - shown on top */}
      {(status === 'loading' || status === 'rendering') && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-[var(--color-surface)] rounded-xl gap-3 z-10">
          <Loader2 className="w-8 h-8 animate-spin text-[var(--color-accent)]" />
          <span className="text-sm text-[var(--color-text-muted)]">
            {status === 'loading' ? 'Fetching sheet music...' : 'Rendering notation (this may take a moment)...'}
          </span>
        </div>
      )}

      {/* Error state */}
      {status === 'error' && (
        <div className="absolute inset-0 flex items-center justify-center bg-[var(--color-surface)] rounded-xl text-red-400 z-10">
          <AlertCircle className="w-6 h-6 mr-2" />
          {error}
        </div>
      )}
    </div>
  );
}

function ContinueButton({ onClick, loading, label }: { onClick: () => void; loading: boolean; label: string }) {
  return (
    <button
      onClick={onClick}
      disabled={loading}
      className="w-full py-4 bg-[var(--color-accent)] text-[var(--color-bg)] font-semibold rounded-xl hover:bg-[var(--color-accent-hover)] disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2 text-lg"
    >
      {loading ? (
        <>
          <Loader2 className="w-5 h-5 animate-spin" />
          Processing...
        </>
      ) : (
        <>
          {label}
          <ChevronRight className="w-5 h-5" />
        </>
      )}
    </button>
  );
}

// =============================================================================
// Main App
// =============================================================================

interface CachedJob {
  job_id: string;
  title: string;
  has_musicxml: boolean;
  has_stems: boolean;
}

export default function App() {
  const [url, setUrl] = useState('');
  const [job, setJob] = useState<Job | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isContinuing, setIsContinuing] = useState(false);
  const [cachedJobs, setCachedJobs] = useState<CachedJob[]>([]);
  const [showCached, setShowCached] = useState(false);
  const [isRestoring, setIsRestoring] = useState(false);

  const isProcessing = job && !['separated', 'transcribed', 'chords_detected', 'completed', 'failed'].includes(job.step);

  // Load cached jobs on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/jobs/cached`)
      .then(res => res.json())
      .then(data => setCachedJobs(data))
      .catch(console.error);
  }, []);

  const handleRestore = async (jobId: string) => {
    setIsRestoring(true);
    try {
      const response = await fetch(`${API_BASE}/api/job/${jobId}/restore`, { method: 'POST' });
      if (!response.ok) throw new Error('Failed to restore job');
      const data: Job = await response.json();
      setJob(data);
      setShowCached(false);
    } catch (error) {
      console.error('Restore error:', error);
    } finally {
      setIsRestoring(false);
    }
  };

  // Poll for job status
  const pollJob = useCallback(async (jobId: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/job/${jobId}`);
      if (!response.ok) throw new Error('Failed to fetch job status');
      const data: Job = await response.json();
      setJob(data);
      
      // Continue polling if processing
      if (!['separated', 'transcribed', 'chords_detected', 'completed', 'failed'].includes(data.step)) {
        setTimeout(() => pollJob(jobId), 1000);
      } else {
        setIsContinuing(false);
      }
    } catch (error) {
      console.error('Polling error:', error);
    }
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim() || isSubmitting) return;

    setIsSubmitting(true);
    setJob(null);

    try {
      const response = await fetch(`${API_BASE}/api/process-song`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ youtube_url: url }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to start processing');
      }

      const data: Job = await response.json();
      setJob(data);
      pollJob(data.job_id);
    } catch (error) {
      setJob({
        job_id: '',
        step: 'failed',
        progress: 0,
        message: '',
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleContinue = async () => {
    if (!job || isContinuing) return;
    
    setIsContinuing(true);
    
    try {
      const response = await fetch(`${API_BASE}/api/job/${job.job_id}/continue`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error('Failed to continue processing');
      }

      pollJob(job.job_id);
    } catch (error) {
      console.error('Continue error:', error);
      setIsContinuing(false);
    }
  };

  const getContinueLabel = () => {
    if (!job) return '';
    switch (job.step) {
      case 'separated': return 'Continue to Transcribe Melody';
      case 'transcribed': return 'Continue to Detect Chords';
      case 'chords_detected': return 'Generate Lead Sheet';
      default: return 'Continue';
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-[var(--color-border)] bg-[var(--color-surface)]/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[var(--color-accent)] to-purple-600 flex items-center justify-center">
            <Music className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-semibold" style={{ fontFamily: 'var(--font-display)' }}>
              Lead Sheet Transcriptor
            </h1>
            <p className="text-xs text-[var(--color-text-muted)]">
              YouTube → Melody + Chords
            </p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-4xl mx-auto px-6 py-8 w-full">
        {/* Input Section */}
        <section className="mb-8">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="youtube-url" className="block text-sm font-medium mb-2">
                YouTube URL
              </label>
              <div className="flex gap-3">
                <input
                  id="youtube-url"
                  type="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://www.youtube.com/watch?v=..."
                  className="flex-1 px-4 py-3 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-xl text-[var(--color-text)] placeholder:text-[var(--color-text-muted)] focus:outline-none focus:ring-2 focus:ring-[var(--color-accent)] focus:border-transparent transition-all"
                  disabled={isSubmitting || (job && isProcessing)}
                />
                <button
                  type="submit"
                  disabled={!url.trim() || isSubmitting || (job && isProcessing)}
                  className="px-6 py-3 bg-[var(--color-accent)] text-[var(--color-bg)] font-semibold rounded-xl hover:bg-[var(--color-accent-hover)] disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2"
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Starting
                    </>
                  ) : (
                    'Start'
                  )}
                </button>
              </div>
            </div>
          </form>
        </section>

        {/* Job Status */}
        {job && (
          <section className="space-y-6">
            {/* Step Indicator */}
            <StepIndicator currentStep={job.step} isProcessing={!!isProcessing} />
            
            {/* Title & Info */}
            <div className="p-5 bg-[var(--color-surface)] rounded-2xl border border-[var(--color-border)]">
              {job.title && (
                <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  {job.step === 'completed' && <CheckCircle2 className="w-5 h-5 text-green-500" />}
                  {job.step === 'failed' && <AlertCircle className="w-5 h-5 text-red-500" />}
                  {isProcessing && <Loader2 className="w-5 h-5 animate-spin text-[var(--color-accent)]" />}
                  {isWaitingForUser(job.step) && <CheckCircle2 className="w-5 h-5 text-[var(--color-accent)]" />}
                  {job.title}
                </h2>
              )}
              
              <p className="text-[var(--color-text-muted)] text-sm mb-4">{job.message}</p>
              
              {/* Progress bar */}
              <div className="h-2 bg-[var(--color-surface-elevated)] rounded-full overflow-hidden">
                <div 
                  className={`h-full rounded-full transition-all duration-500 ${
                    job.step === 'failed' 
                      ? 'bg-red-500' 
                      : job.step === 'completed'
                      ? 'bg-green-500'
                      : 'bg-gradient-to-r from-[var(--color-accent)] to-purple-400'
                  }`}
                  style={{ width: `${job.progress}%` }}
                />
              </div>
              
              {job.error && (
                <div className="mt-4 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-sm">
                  {job.error}
                </div>
              )}

              {/* Musical Info */}
              {(job.bpm || job.key || job.duration) && (
                <div className="mt-4 flex flex-wrap gap-3 text-sm">
                  {job.bpm && (
                    <div className="px-3 py-1.5 bg-[var(--color-surface-elevated)] rounded-lg">
                      <span className="text-[var(--color-text-muted)]">BPM:</span>{' '}
                      <span className="font-medium">{Math.round(job.bpm)}</span>
                    </div>
                  )}
                  {job.key && (
                    <div className="px-3 py-1.5 bg-[var(--color-surface-elevated)] rounded-lg">
                      <span className="text-[var(--color-text-muted)]">Key:</span>{' '}
                      <span className="font-medium">{job.key}</span>
                    </div>
                  )}
                  {job.time_signature && (
                    <div className="px-3 py-1.5 bg-[var(--color-surface-elevated)] rounded-lg">
                      <span className="text-[var(--color-text-muted)]">Time:</span>{' '}
                      <span className="font-medium">{job.time_signature}</span>
                    </div>
                  )}
                  {job.duration && (
                    <div className="px-3 py-1.5 bg-[var(--color-surface-elevated)] rounded-lg">
                      <span className="text-[var(--color-text-muted)]">Duration:</span>{' '}
                      <span className="font-medium">{formatTime(job.duration)}</span>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Step 1: Audio Players */}
            {(job.vocals_url || job.accompaniment_url) && (
              <div className="space-y-4">
                <h3 className="font-semibold flex items-center gap-2">
                  <Waves className="w-5 h-5 text-[var(--color-accent)]" />
                  Separated Audio
                </h3>
                <div className="grid gap-4 md:grid-cols-2">
                  {job.vocals_url && (
                    <AudioPlayer
                      label="Vocals (Melody Source)"
                      url={`${API_BASE}${job.vocals_url}`}
                      icon={Mic2}
                    />
                  )}
                  {job.accompaniment_url && (
                    <AudioPlayer
                      label="Accompaniment (Chord Source)"
                      url={`${API_BASE}${job.accompaniment_url}`}
                      icon={Guitar}
                    />
                  )}
                </div>
              </div>
            )}

            {/* Step 2: Melody Preview */}
            {job.melody_notes && job.melody_notes.length > 0 && (
              <div className="space-y-4">
                <h3 className="font-semibold flex items-center gap-2">
                  <FileMusic className="w-5 h-5 text-[var(--color-accent)]" />
                  Melody Transcription
                </h3>
                <MelodyPreview notes={job.melody_notes} bpm={job.bpm || 120} />
                {job.melody_midi_url && (
                  <a
                    href={`${API_BASE}${job.melody_midi_url}`}
                    download="melody.mid"
                    className="inline-flex items-center gap-2 px-4 py-2 text-sm bg-[var(--color-surface)] hover:bg-[var(--color-surface-elevated)] border border-[var(--color-border)] rounded-lg transition-colors"
                  >
                    <Download className="w-4 h-4" />
                    Download MIDI
                  </a>
                )}
              </div>
            )}

            {/* Step 3: Chords Preview */}
            {job.chords && job.chords.length > 0 && job.duration && (
              <div className="space-y-4">
                <h3 className="font-semibold flex items-center gap-2">
                  <Piano className="w-5 h-5 text-[var(--color-accent)]" />
                  Chord Progression
                </h3>
                <ChordsPreview chords={job.chords} duration={job.duration} />
              </div>
            )}

            {/* Continue Button */}
            {isWaitingForUser(job.step) && (
              <ContinueButton 
                onClick={handleContinue} 
                loading={isContinuing} 
                label={getContinueLabel()} 
              />
            )}

            {/* Step 4: Sheet Music */}
            {job.music_xml_url && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold flex items-center gap-2" style={{ fontFamily: 'var(--font-display)' }}>
                    <Music className="w-5 h-5 text-[var(--color-accent)]" />
                    Lead Sheet
                  </h3>
                  <a
                    href={`${API_BASE}${job.music_xml_url}`}
                    download="lead_sheet.musicxml"
                    className="flex items-center gap-2 px-4 py-2 text-sm bg-[var(--color-surface-elevated)] hover:bg-[var(--color-border)] rounded-lg transition-colors"
                  >
                    <Download className="w-4 h-4" />
                    Download MusicXML
                  </a>
                </div>
                <SheetMusicRenderer musicXmlUrl={`${API_BASE}${job.music_xml_url}`} />
              </div>
            )}
          </section>
        )}

        {/* Empty State */}
        {!job && (
          <div className="text-center py-16">
            <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-[var(--color-surface)] flex items-center justify-center">
              <Music className="w-10 h-10 text-[var(--color-text-muted)]" />
            </div>
            <h2 className="text-xl font-semibold mb-2" style={{ fontFamily: 'var(--font-display)' }}>
              Paste a YouTube URL to get started
            </h2>
            <p className="text-[var(--color-text-muted)] max-w-md mx-auto mb-8">
              We'll extract the vocals, detect the chords, and generate a 
              professional lead sheet with melody notation and chord symbols.
            </p>
            <div className="flex justify-center gap-6 text-sm text-[var(--color-text-muted)]">
              {STEPS.map((step) => {
                const Icon = step.icon;
                return (
                  <div key={step.id} className="flex items-center gap-2">
                    <Icon className="w-4 h-4" />
                    {step.label}
                  </div>
                );
              })}
            </div>

            {/* Cached Jobs */}
            {cachedJobs.length > 0 && (
              <div className="mt-12">
                <button
                  onClick={() => setShowCached(!showCached)}
                  className="text-sm text-[var(--color-accent)] hover:text-[var(--color-accent-hover)] transition-colors"
                >
                  {showCached ? 'Hide' : 'Show'} {cachedJobs.length} cached job{cachedJobs.length > 1 ? 's' : ''}
                </button>
                
                {showCached && (
                  <div className="mt-4 max-w-xl mx-auto space-y-2">
                    {cachedJobs.map((cached) => (
                      <div
                        key={cached.job_id}
                        className="flex items-center justify-between p-3 bg-[var(--color-surface)] rounded-lg border border-[var(--color-border)]"
                      >
                        <div className="flex items-center gap-3 overflow-hidden">
                          {cached.has_musicxml ? (
                            <FileMusic className="w-5 h-5 text-green-400 shrink-0" />
                          ) : (
                            <Volume2 className="w-5 h-5 text-[var(--color-accent)] shrink-0" />
                          )}
                          <span className="text-sm truncate" title={cached.title}>
                            {cached.title.replace(/https?:\/\/(www\.)?(youtube\.com|youtu\.be)\/(watch\?v=)?/g, '')}
                          </span>
                        </div>
                        <button
                          onClick={() => handleRestore(cached.job_id)}
                          disabled={isRestoring}
                          className="px-3 py-1 text-sm bg-[var(--color-accent)]/20 text-[var(--color-accent)] rounded hover:bg-[var(--color-accent)]/30 transition-colors disabled:opacity-50 shrink-0"
                        >
                          {isRestoring ? 'Loading...' : 'Restore'}
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-[var(--color-border)] mt-auto">
        <div className="max-w-4xl mx-auto px-6 py-4 text-center text-sm text-[var(--color-text-muted)]">
          Powered by Demucs, Basic Pitch, and music21
        </div>
      </footer>
    </div>
  );
}
