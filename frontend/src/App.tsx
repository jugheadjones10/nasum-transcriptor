import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Music, Mic2, Download, Loader2, CheckCircle2, AlertCircle, Sparkles, FileMusic, ListMusic, Play, Pause, Volume2, ArrowRight } from 'lucide-react'
import abcjs from 'abcjs'
import './App.css'

interface SeparatedTrack {
  name: string
  url: string
}

interface Chord {
  time: number
  chord: string
  confidence: number
}

interface JobStatus {
  job_id: string
  status: string
  progress: number
  message: string
  cached: boolean | null
  video_title: string | null
  separated_tracks: SeparatedTrack[] | null
  midi_url: string | null
  abc_notation: string | null
  chords: Chord[] | null
  error: string | null
}

const API_BASE = 'http://localhost:8000'

function AudioPlayer({ track }: { track: SeparatedTrack }) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [progress, setProgress] = useState(0)

  const togglePlay = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause()
      } else {
        // Pause all other audio elements first
        document.querySelectorAll('audio').forEach(audio => {
          if (audio !== audioRef.current) {
            audio.pause()
          }
        })
        audioRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      const pct = (audioRef.current.currentTime / audioRef.current.duration) * 100
      setProgress(pct || 0)
    }
  }

  const handleEnded = () => {
    setIsPlaying(false)
    setProgress(0)
  }

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    if (audioRef.current) {
      const rect = e.currentTarget.getBoundingClientRect()
      const pct = (e.clientX - rect.left) / rect.width
      audioRef.current.currentTime = pct * audioRef.current.duration
    }
  }

  return (
    <div className="audio-player">
      <audio
        ref={audioRef}
        src={`${API_BASE}${track.url}`}
        onTimeUpdate={handleTimeUpdate}
        onEnded={handleEnded}
        onPause={() => setIsPlaying(false)}
        onPlay={() => setIsPlaying(true)}
      />
      <button className="play-btn" onClick={togglePlay}>
        {isPlaying ? <Pause size={18} /> : <Play size={18} />}
      </button>
      <div className="track-info">
        <span className="track-name">{track.name}</span>
        <div className="progress-track" onClick={handleSeek}>
          <div className="progress-track-fill" style={{ width: `${progress}%` }} />
        </div>
      </div>
      <Volume2 size={16} className="volume-icon" />
    </div>
  )
}

function App() {
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [currentJob, setCurrentJob] = useState<JobStatus | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isContinuing, setIsContinuing] = useState(false)
  const sheetMusicRef = useRef<HTMLDivElement>(null)

  // Poll for job status
  useEffect(() => {
    if (!currentJob || currentJob.status === 'completed' || currentJob.status === 'failed' || currentJob.status === 'waiting_review') {
      return
    }

    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE}/api/status/${currentJob.job_id}`)
        const data = await response.json()
        setCurrentJob(data)
      } catch (error) {
        console.error('Error polling status:', error)
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [currentJob])

  // Render sheet music when ABC notation is available
  useEffect(() => {
    if (currentJob?.abc_notation && sheetMusicRef.current) {
      sheetMusicRef.current.innerHTML = ''
      abcjs.renderAbc(sheetMusicRef.current, currentJob.abc_notation, {
        responsive: 'resize',
        add_classes: true,
        staffwidth: 800,
        wrap: {
          minSpacing: 1.5,
          maxSpacing: 2.5,
          preferredMeasuresPerLine: 4
        }
      })
    }
  }, [currentJob?.abc_notation])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!youtubeUrl.trim()) return

    setIsSubmitting(true)
    try {
      const response = await fetch(`${API_BASE}/api/transcribe`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ youtube_url: youtubeUrl }),
      })

      if (!response.ok) {
        throw new Error('Failed to start transcription')
      }

      const data = await response.json()
      setCurrentJob(data)
    } catch (error) {
      console.error('Error starting transcription:', error)
      alert('Failed to start transcription. Make sure the backend is running.')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleContinue = async () => {
    if (!currentJob) return

    setIsContinuing(true)
    try {
      const response = await fetch(`${API_BASE}/api/continue/${currentJob.job_id}`, {
        method: 'POST',
      })

      if (!response.ok) {
        throw new Error('Failed to continue transcription')
      }

      const data = await response.json()
      setCurrentJob(data)
    } catch (error) {
      console.error('Error continuing:', error)
      alert('Failed to continue. Please try again.')
    } finally {
      setIsContinuing(false)
    }
  }

  const getStatusIcon = () => {
    switch (currentJob?.status) {
      case 'completed':
        return <CheckCircle2 className="status-icon success" />
      case 'failed':
        return <AlertCircle className="status-icon error" />
      case 'waiting_review':
        return <Volume2 className="status-icon accent" />
      default:
        return <Loader2 className="status-icon spinning" />
    }
  }

  const resetJob = () => {
    // Stop all audio
    document.querySelectorAll('audio').forEach(audio => audio.pause())
    setCurrentJob(null)
    setYoutubeUrl('')
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="app">
      {/* Animated background */}
      <div className="bg-gradient" />
      <div className="bg-pattern" />
      
      {/* Floating musical notes */}
      <div className="floating-notes">
        {[...Array(6)].map((_, i) => (
          <motion.div
            key={i}
            className="floating-note"
            initial={{ y: '100vh', x: `${15 + i * 15}vw`, opacity: 0 }}
            animate={{
              y: '-10vh',
              opacity: [0, 0.3, 0.3, 0],
            }}
            transition={{
              duration: 15 + i * 2,
              repeat: Infinity,
              delay: i * 2,
              ease: 'linear',
            }}
          >
            ♪
          </motion.div>
        ))}
      </div>

      <motion.header
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
      >
        <div className="logo">
          <Mic2 className="logo-icon" />
          <h1>Nasum Transcriptor</h1>
        </div>
        <p className="tagline">Vocal melody + chord transcription from YouTube</p>
      </motion.header>

      <main>
        <AnimatePresence mode="wait">
          {!currentJob ? (
            <motion.section
              key="input"
              className="input-section"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              transition={{ duration: 0.4 }}
            >
              <div className="card glass">
                <div className="card-header">
                  <Sparkles className="card-icon" />
                  <h2>Create Lead Sheet</h2>
                </div>
                
                <form onSubmit={handleSubmit}>
                  <div className="input-group">
                    <label htmlFor="youtube-url">YouTube URL</label>
                    <div className="input-wrapper">
                      <Music className="input-icon" />
                      <input
                        id="youtube-url"
                        type="url"
                        placeholder="https://youtube.com/watch?v=..."
                        value={youtubeUrl}
                        onChange={(e) => setYoutubeUrl(e.target.value)}
                        required
                      />
                    </div>
                  </div>

                  <motion.button
                    type="submit"
                    className="btn primary"
                    disabled={isSubmitting || !youtubeUrl.trim()}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    {isSubmitting ? (
                      <>
                        <Loader2 className="btn-icon spinning" />
                        Starting...
                      </>
                    ) : (
                      <>
                        <Mic2 className="btn-icon" />
                        Separate & Transcribe
                      </>
                    )}
                  </motion.button>
                </form>

                <div className="info-box">
                  <h3>How it works</h3>
                  <ol>
                    <li>Paste a YouTube link (CCM, worship, etc.)</li>
                    <li>AI separates vocals, drums, bass, and other instruments</li>
                    <li><strong>Preview each track</strong> before transcribing</li>
                    <li>Vocal melody is transcribed + chords detected</li>
                    <li>Download lead sheet with melody + chords</li>
                  </ol>
                </div>
              </div>
            </motion.section>
          ) : currentJob.status === 'waiting_review' ? (
            <motion.section
              key="review"
              className="review-section"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              transition={{ duration: 0.4 }}
            >
              <div className="card glass wide">
                <div className="card-header">
                  <Volume2 className="card-icon" />
                  <h2>Preview Separated Tracks</h2>
                  {currentJob.cached && (
                    <span className="cache-badge">⚡ Cached</span>
                  )}
                </div>

                {currentJob.video_title && (
                  <p className="video-title">🎵 {currentJob.video_title}</p>
                )}

                <p className="review-instructions">
                  Listen to each separated track below. Make sure the <strong>Vocals</strong> track 
                  sounds correct before continuing with transcription.
                </p>

                <div className="tracks-container">
                  {currentJob.separated_tracks?.map((track, idx) => (
                    <AudioPlayer key={idx} track={track} />
                  ))}
      </div>

                <div className="review-actions">
                  <motion.button
                    onClick={handleContinue}
                    className="btn primary"
                    disabled={isContinuing}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    {isContinuing ? (
                      <>
                        <Loader2 className="btn-icon spinning" />
                        Processing...
                      </>
                    ) : (
                      <>
                        Continue with Transcription
                        <ArrowRight className="btn-icon" />
                      </>
                    )}
                  </motion.button>

                  <motion.button
                    onClick={resetJob}
                    className="btn outline"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    Start Over
                  </motion.button>
                </div>
              </div>
            </motion.section>
          ) : (
            <motion.section
              key="progress"
              className="progress-section"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              transition={{ duration: 0.4 }}
            >
              <div className="card glass wide">
                <div className="card-header">
                  {getStatusIcon()}
                  <h2>
                    {currentJob.status === 'completed'
                      ? 'Lead Sheet Ready!'
                      : currentJob.status === 'failed'
                      ? 'Transcription Failed'
                      : 'Processing...'}
                  </h2>
                </div>

                <div className="progress-container">
                  <div className="progress-bar">
                    <motion.div
                      className="progress-fill"
                      initial={{ width: 0 }}
                      animate={{ width: `${currentJob.progress}%` }}
                      transition={{ duration: 0.5 }}
                    />
                  </div>
                  <span className="progress-text">{currentJob.progress}%</span>
                </div>

                <p className="status-message">{currentJob.message}</p>

                {currentJob.error && (
                  <div className="error-box">
                    <AlertCircle />
                    <p>{currentJob.error}</p>
                  </div>
                )}

                {currentJob.status === 'completed' && (
                  <motion.div
                    className="results"
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.2 }}
                  >
                    {/* Chord Progression */}
                    {currentJob.chords && currentJob.chords.length > 0 && (
                      <div className="chord-section">
                        <h3>🎸 Chord Progression</h3>
                        <div className="chord-list">
                          {currentJob.chords.slice(0, 24).map((chord, idx) => (
                            <div key={idx} className="chord-item">
                              <span className="chord-time">{formatTime(chord.time)}</span>
                              <span className="chord-name">{chord.chord}</span>
                            </div>
                          ))}
                          {currentJob.chords.length > 24 && (
                            <div className="chord-item more">
                              +{currentJob.chords.length - 24} more...
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Sheet Music */}
                    <h3>🎼 Vocal Melody with Chords</h3>
                    <div className="sheet-music-container" ref={sheetMusicRef} />

                    <div className="download-buttons">
                      <a
                        href={`${API_BASE}${currentJob.midi_url}`}
                        download
                        className="btn secondary"
                      >
                        <Download className="btn-icon" />
                        Vocal MIDI
                      </a>
                      <a
                        href={`${API_BASE}/api/download/${currentJob.job_id}/abc`}
                        download
                        className="btn secondary"
                      >
                        <FileMusic className="btn-icon" />
                        Lead Sheet (ABC)
                      </a>
                      <a
                        href={`${API_BASE}/api/download/${currentJob.job_id}/chords`}
                        download
                        className="btn secondary"
                      >
                        <ListMusic className="btn-icon" />
                        Chords Only
                      </a>
                    </div>
                  </motion.div>
                )}

                <motion.button
                  onClick={resetJob}
                  className="btn outline"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  {currentJob.status === 'completed' || currentJob.status === 'failed'
                    ? 'Transcribe Another'
                    : 'Cancel'}
                </motion.button>
              </div>
            </motion.section>
          )}
        </AnimatePresence>
      </main>

      <footer>
        <p>AI-powered vocal & chord transcription for worship teams</p>
      </footer>
    </div>
  )
}

export default App
