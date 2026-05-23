import { useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || ''

function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async () => {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim() }),
      })
      if (!res.ok) {
        const detail = await res.json().catch(() => null)
        throw new Error(detail?.detail || `Server responded ${res.status}`)
      }
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Failed to reach the API')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleSubmit()
    }
  }

  return (
    <div className="app">
      {/* ── Navbar ────────────────────────────────── */}
      <nav className="navbar">
        <div className="nav-brand">
          Shadwal<span className="accent">Singh</span>
        </div>
        <ul className="nav-links">
          <li><a href="#classify">Classify</a></li>
          <li><a href="https://github.com/shadwalsr" target="_blank" rel="noopener noreferrer">GitHub</a></li>
          <li><a href="mailto:shadwalsr@gmail.com">Contact</a></li>
        </ul>
      </nav>

      {/* ── Hero ──────────────────────────────────── */}
      <main className="hero" id="classify">
        <div className="hero-content">
          <h1 className="hero-title">
            SpamZ: A SMS
            <br />
            Spam Classifier
          </h1>
          <p className="hero-subtitle">
            Paste any SMS message below and get an instant classification — 
            powered by a fine-tuned DistilBERT model served via ONNX Runtime.
          </p>

          {/* ── Classifier Card ──────────────────── */}
          <div className="classifier-card">
            <div className="card-label">Classify a message</div>
            <textarea
              id="sms-input"
              className="text-input"
              placeholder="Paste an SMS message here…"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
            />
            <button
              id="submit-btn"
              className="submit-btn"
              onClick={handleSubmit}
              disabled={loading || !text.trim()}
            >
              {loading && <span className="spinner" />}
              {loading ? 'Classifying…' : 'Classify'}
            </button>

            {/* ── Error ──────────────────────────── */}
            {error && (
              <div className="error" id="error-message">{error}</div>
            )}

            {/* ── Result ─────────────────────────── */}
            {result && (
              <div className="result" id="result-card">
                {/* Label */}
                <div className="result-row">
                  <span className="result-label">Label</span>
                  <span className={`badge badge--${result.label}`}>
                    <span className="badge-dot" />
                    {result.label}
                  </span>
                </div>

                {/* Confidence */}
                <div className="result-row">
                  <span className="result-label">Confidence</span>
                  <div className="confidence-bar-wrapper">
                    <div className="confidence-bar">
                      <div
                        className={`confidence-fill confidence-fill--${result.label}`}
                        style={{ width: `${(result.confidence * 100).toFixed(0)}%` }}
                      />
                    </div>
                    <span className="confidence-value">
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Latency */}
                <div className="result-row">
                  <span className="result-label">Latency</span>
                  <span className="latency">{result.latency_ms.toFixed(1)} ms</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ── Geometric Shapes ────────────────────── */}
        <div className="hero-visual" aria-hidden="true">
          <div className="shape shape--purple-lg" />
          <div className="shape shape--teal" />
          <div className="shape shape--coral-lg" />
          <div className="shape shape--coral-sm" />
          <div className="shape shape--purple-rect" />

          {/* Dot grids */}
          <div className="dot-grid dot-grid--top">
            {Array.from({ length: 12 }).map((_, i) => (
              <span className="dot" key={`dt-${i}`} />
            ))}
          </div>
          <div className="dot-grid dot-grid--bottom">
            {Array.from({ length: 8 }).map((_, i) => (
              <span className="dot" key={`db-${i}`} />
            ))}
          </div>
        </div>
      </main>

      {/* ── Footer ────────────────────────────────── */}
      <footer className="footer">
        © {new Date().getFullYear()} Shadwal Singh · SpamZ — SMS Spam Classifier
      </footer>
    </div>
  )
}

export default App
