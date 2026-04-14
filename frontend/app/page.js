'use client'
import { useState, useEffect } from 'react'

const FIELD_LABELS = {
  invoice_number: 'Invoice number',
  invoice_date:   'Invoice date',
  due_date:       'Due date',
  issuer:         'Issuer',
  recipient:      'Recipient',
  total:          'Total amount'
}

const CATEGORY_STYLES = {
  invoice:  { color: '#a3e635', colorLight: '#3f6212', label: 'Invoice',  description: 'Field extraction triggered' },
  contract: { color: '#86efac', colorLight: '#16a34a', label: 'Contract', description: 'No extraction for contracts' },
  email:    { color: '#4ade80', colorLight: '#15803d', label: 'Email',    description: 'No extraction for emails' },
  news:     { color: '#34d399', colorLight: '#047857', label: 'News',     description: 'No extraction for news' },
}

// ─── CONFIGURATION ────────────────────────────────────────────────────────────
// When the backend is ready, change this to 'false' to use the real API
const USE_MOCK = true

// The URL of the Python backend — change this if the backend runs on a different port
const API_URL = 'http://localhost:8000/predict'

// Mock data that simulates what the backend would return
// This is used when USE_MOCK is true
const MOCK_RESULT = {
  category: 'invoice',
  confidence: 97,
  fields: {
    invoice_number: 'INV-2024-0342',
    invoice_date:   '2024-03-15',
    due_date:       '2024-04-15',
    issuer:         'TechSolutions S.L.',
    recipient:      'Global Retail Corp.',
    total:          '14,850.00'
  }
}
// ─────────────────────────────────────────────────────────────────────────────

export default function Home() {
  const [step, setStep] = useState('upload')
  const [result, setResult] = useState(null)
  const [file, setFile] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [tick, setTick] = useState(0)
  const [dark, setDark] = useState(true)
  const [barWidth, setBarWidth] = useState(0)

  // error state — stores a message if something goes wrong
  const [error, setError] = useState(null)

  const t = {
    bg:         dark ? '#0a0a0a'  : '#f5f5f5',
    bgDot:      dark ? '#ffffff08': '#00000008',
    surface:    dark ? '#0f0f0f'  : '#ffffff',
    surfaceAlt: dark ? '#111111'  : '#f0f0f0',
    border:     dark ? '#1f1f1f'  : '#e0e0e0',
    borderSub:  dark ? '#2a2a2a'  : '#d4d4d4',
    text:       dark ? '#ffffff'  : '#0a0a0a',
    textMuted:  dark ? '#555555'  : '#888888',
    textDim:    dark ? '#333333'  : '#bbbbbb',
    accentText: dark ? '#a3e635'  : '#3f6212',
  }

  useEffect(() => {
    if (step !== 'processing') return
    const interval = setInterval(() => setTick(n => n + 1), 600)

    async function classify() {
      try {
        let data

        if (USE_MOCK) {
          // ── MOCK MODE ────────────────────────────────────────────────────
          await new Promise(resolve => setTimeout(resolve, 2500))
          data = MOCK_RESULT
          // ─────────────────────────────────────────────────────────────────
        } else {
          // ── REAL API MODE ─────────────────────────────────────────────────
          const formData = new FormData()
          formData.append('file', file)

          const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
          })

          if (!response.ok) {
            throw new Error(`Backend error: ${response.status}`)
          }

          data = await response.json()
          // ─────────────────────────────────────────────────────────────────
        }

        setResult(data)
        setStep('result')

      } catch (err) {
        console.error('Classification failed:', err)
        setError('Something went wrong. Please try again.')
        setStep('upload')
      }
    }

    classify()
    return () => clearInterval(interval)
  }, [step])

  useEffect(() => {
    if (step === 'result') {
      setTimeout(() => setBarWidth(result?.confidence ?? 0), 100)
    } else {
      setBarWidth(0)
    }
  }, [step, result])

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped?.type === 'application/pdf') setFile(dropped)
  }

  const cat = result ? (CATEGORY_STYLES[result.category] || CATEGORY_STYLES.invoice) : null
  const catColor = cat ? (dark ? cat.color : cat.colorLight) : '#a3e635'
  const dots = ['Extracting text', 'Running classifier', 'Extracting fields']

  // pills match CATEGORY_STYLES colors exactly
  const pills = [
    { label: 'Invoice',  color: dark ? '#a3e635' : '#3f6212' },
    { label: 'Contract', color: dark ? '#86efac' : '#16a34a' },
    { label: 'Email',    color: dark ? '#4ade80' : '#15803d' },
    { label: 'News',     color: dark ? '#34d399' : '#047857' },
  ]

  return (
    <div style={{
      minHeight: '100vh',
      backgroundColor: t.bg,
      backgroundImage: `radial-gradient(circle, ${t.bgDot} 1px, transparent 1px)`,
      backgroundSize: '28px 28px',
      color: t.text,
      fontFamily: 'system-ui, sans-serif',
      display: 'flex',
      flexDirection: 'column',
      transition: 'background-color 0.3s, color 0.3s'
    }}>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg) } }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to   { opacity: 1; transform: translateY(0px); }
        }
        .screen { animation: fadeIn 0.4s ease; }
      `}</style>

      {/* Header */}
      <header style={{
        borderBottom: `1px solid ${t.border}`,
        padding: '1rem 2rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        backgroundColor: dark ? 'rgba(10,10,10,0.85)' : 'rgba(245,245,245,0.85)',
        backdropFilter: 'blur(10px)',
        position: 'sticky',
        top: 0,
        zIndex: 10
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '28px', height: '28px',
            backgroundColor: t.accentText,
            borderRadius: '6px',
            display: 'flex', alignItems: 'center', justifyContent: 'center'
          }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={dark ? '#0a0a0a' : '#fff'} strokeWidth="2.5">
              <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
            </svg>
          </div>
          <span style={{ fontWeight: 700, fontSize: '14px', letterSpacing: '-0.02em', color: t.text }}>DocClass</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '12px', color: t.textMuted, fontFamily: 'monospace' }}>
            IE University · Statistical Learning
          </span>
          <button
            onClick={() => setDark(d => !d)}
            style={{
              width: '44px', height: '24px',
              borderRadius: '999px',
              border: `1px solid ${t.borderSub}`,
              backgroundColor: dark ? '#1a1a1a' : '#e0e0e0',
              cursor: 'pointer',
              position: 'relative',
              transition: 'all 0.2s',
              padding: 0
            }}
          >
            <div style={{
              width: '16px', height: '16px',
              borderRadius: '50%',
              backgroundColor: t.accentText,
              position: 'absolute',
              top: '3px',
              left: dark ? '3px' : '21px',
              transition: 'left 0.2s'
            }}/>
          </button>
          <span style={{ fontSize: '11px', color: t.textMuted, fontFamily: 'monospace' }}>
            {dark ? 'Dark' : 'Light'}
          </span>
        </div>
      </header>

      <main style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '3rem 2rem' }}>

        {/* UPLOAD SCREEN */}
        {step === 'upload' && (
          <div className="screen" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4rem', maxWidth: '900px', width: '100%', alignItems: 'center' }}>
            <div>
              <h1 style={{ fontSize: '52px', fontWeight: 800, lineHeight: 1.05, letterSpacing: '-0.03em', marginBottom: '0.5rem', color: t.text }}>
                Classify<br/><span style={{ color: t.accentText }}>Documents.</span>
              </h1>
              <p style={{ fontSize: '16px', fontWeight: 600, color: t.textMuted, marginBottom: '1rem', letterSpacing: '-0.01em' }}>
                Invoice, Contract, Email or News?
              </p>
              <p style={{ color: t.textMuted, fontSize: '14px', lineHeight: 1.7 }}>
                Upload a PDF and our pipeline will classify it into one of four categories and extract structured data from invoices automatically.
              </p>

              {/* Error message — only shows if something went wrong */}
              {error && (
                <div style={{
                  marginTop: '1rem',
                  padding: '10px 14px',
                  borderRadius: '10px',
                  backgroundColor: 'rgba(248,113,113,0.08)',
                  border: '1px solid rgba(248,113,113,0.2)',
                  fontSize: '13px',
                  color: '#f87171',
                  fontFamily: 'monospace'
                }}>
                  {error}
                </div>
              )}
            </div>

            <div>
              <div
                onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-input').click()}
                style={{
                  border: `2px dashed ${dragOver ? t.accentText : file ? t.accentText + '66' : t.borderSub}`,
                  borderRadius: '20px',
                  padding: '3rem 2rem',
                  textAlign: 'center',
                  cursor: 'pointer',
                  backgroundColor: dragOver ? 'rgba(163,230,53,0.04)' : t.surfaceAlt,
                  transition: 'all 0.2s',
                  marginBottom: '1rem'
                }}
              >
                <div style={{
                  width: '48px', height: '48px',
                  backgroundColor: t.surface,
                  border: `1px solid ${t.border}`,
                  borderRadius: '12px',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  margin: '0 auto 1rem'
                }}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={file ? t.accentText : t.textMuted} strokeWidth="1.5">
                    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                  </svg>
                </div>
                {file
                  ? <p style={{ fontSize: '13px', fontFamily: 'monospace', color: t.accentText, wordBreak: 'break-all' }}>{file.name}</p>
                  : <>
                      <p style={{ fontSize: '14px', color: t.textMuted, marginBottom: '4px' }}>Drop PDF here or click to browse</p>
                      <p style={{ fontSize: '12px', color: t.textDim }}>PDF only</p>
                    </>
                }
                <input id="file-input" type="file" accept=".pdf" style={{ display: 'none' }}
                  onChange={(e) => {
                    setFile(e.target.files[0])
                    setError(null)
                  }} />
              </div>

              {/* Category pills — each is a different shade of green */}
              <div style={{ display: 'flex', gap: '8px', marginBottom: '12px', flexWrap: 'wrap' }}>
                {pills.map(c => (
                  <span key={c.label} style={{
                    fontSize: '11px',
                    padding: '3px 10px',
                    borderRadius: '999px',
                    border: `1px solid ${c.color}44`,
                    color: c.color,
                    fontFamily: 'monospace',
                    backgroundColor: `${c.color}11`
                  }}>{c.label}</span>
                ))}
              </div>

              <button
                onClick={() => { if (file) setStep('processing') }}
                style={{
                  width: '100%',
                  padding: '14px',
                  borderRadius: '12px',
                  border: 'none',
                  backgroundColor: file ? t.accentText : t.surfaceAlt,
                  color: file ? '#0a0a0a' : t.textDim,
                  fontWeight: 700,
                  fontSize: '14px',
                  cursor: file ? 'pointer' : 'not-allowed',
                  transition: 'all 0.2s',
                  letterSpacing: '-0.01em'
                }}
              >
                {file ? 'Run classification →' : 'Select a file first'}
              </button>
            </div>
          </div>
        )}

        {/* PROCESSING SCREEN */}
        {step === 'processing' && (
          <div className="screen" style={{ textAlign: 'center', maxWidth: '400px' }}>
            <div style={{
              width: '56px', height: '56px',
              border: `2px solid ${t.border}`,
              borderTop: `2px solid ${t.accentText}`,
              borderRadius: '50%',
              margin: '0 auto 2rem',
              animation: 'spin 0.8s linear infinite'
            }}/>
            <p style={{ fontFamily: 'monospace', color: t.accentText, fontSize: '13px', marginBottom: '6px' }}>
              {dots[tick % dots.length]}…
            </p>
            <p style={{ fontFamily: 'monospace', color: t.textDim, fontSize: '12px' }}>{file?.name}</p>
          </div>
        )}

        {/* RESULT SCREEN */}
        {step === 'result' && result && cat && (
          <div className="screen" style={{ maxWidth: '700px', width: '100%' }}>
            <div style={{
              padding: '2rem',
              borderRadius: '20px',
              border: `1px solid ${t.border}`,
              backgroundColor: t.surface,
              marginBottom: '1rem'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
                <div>
                  <p style={{ fontSize: '11px', fontFamily: 'monospace', color: t.textDim, marginBottom: '8px', letterSpacing: '0.1em' }}>
                    CLASSIFICATION RESULT
                  </p>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                    <h1 style={{ fontSize: '64px', fontWeight: 800, letterSpacing: '-0.04em', color: catColor, lineHeight: 1 }}>
                      {cat.label}
                    </h1>
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      padding: '6px 12px',
                      borderRadius: '999px',
                      border: `1px solid ${catColor}44`,
                      backgroundColor: `${catColor}11`
                    }}>
                      <span style={{ fontSize: '16px', fontWeight: 700, color: catColor, letterSpacing: '-0.01em' }}>
                        {result.confidence}%
                      </span>
                      <span style={{ fontSize: '11px', fontFamily: 'monospace', color: t.textMuted }}>
                        confidence
                      </span>
                    </div>
                  </div>
                  <p style={{ fontSize: '12px', fontFamily: 'monospace', color: t.textDim, marginTop: '6px' }}>
                    {file?.name}
                  </p>
                </div>
                <button
                  onClick={() => { setStep('upload'); setResult(null); setFile(null) }}
                  style={{
                    fontSize: '12px',
                    fontFamily: 'monospace',
                    color: t.textMuted,
                    backgroundColor: 'transparent',
                    border: `1px solid ${t.borderSub}`,
                    borderRadius: '8px',
                    padding: '8px 14px',
                    cursor: 'pointer'
                  }}
                >
                  ← New document
                </button>
              </div>

              <div style={{ height: '2px', backgroundColor: t.surfaceAlt, borderRadius: '999px', overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width: `${barWidth}%`,
                  backgroundColor: catColor,
                  borderRadius: '999px',
                  transition: 'width 1s ease'
                }}/>
              </div>
            </div>

            {result.category === 'invoice' && (
              <div style={{
                borderRadius: '20px',
                border: `1px solid ${t.border}`,
                backgroundColor: t.surface,
                overflow: 'hidden'
              }}>
                <div style={{
                  padding: '1rem 1.5rem',
                  borderBottom: `1px solid ${t.border}`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between'
                }}>
                  <p style={{ fontSize: '11px', fontFamily: 'monospace', color: t.textDim, letterSpacing: '0.1em' }}>
                    EXTRACTED FIELDS
                  </p>
                  <span style={{
                    fontSize: '11px',
                    fontFamily: 'monospace',
                    color: t.accentText,
                    backgroundColor: dark ? 'rgba(163,230,53,0.08)' : 'rgba(63,98,18,0.08)',
                    padding: '2px 8px',
                    borderRadius: '999px',
                    border: `1px solid ${dark ? 'rgba(163,230,53,0.15)' : 'rgba(63,98,18,0.2)'}`
                  }}>
                    6 fields
                  </span>
                </div>
                {Object.entries(result.fields).map(([key, value], i) => (
                  <div key={key} style={{
                    padding: '1rem 1.5rem',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    borderBottom: i < 5 ? `1px solid ${t.border}` : 'none',
                  }}>
                    <span style={{ fontSize: '13px', color: t.textMuted }}>{FIELD_LABELS[key]}</span>
                    <span style={{
                      fontFamily: 'monospace',
                      fontSize: '13px',
                      color: value ? t.text : t.textDim,
                      fontStyle: value ? 'normal' : 'italic'
                    }}>
                      {value ?? 'not found'}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {result.category !== 'invoice' && (
              <div style={{
                padding: '2rem',
                borderRadius: '20px',
                border: `1px solid ${catColor}22`,
                backgroundColor: `${catColor}08`,
                textAlign: 'center'
              }}>
                <p style={{ fontSize: '13px', fontFamily: 'monospace', color: catColor }}>
                  {cat.description}
                </p>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}