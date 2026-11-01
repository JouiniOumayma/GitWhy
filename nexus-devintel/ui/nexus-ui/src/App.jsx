import { useState, useRef } from 'react'
import { StatCard } from './StatCard'
import './App.css'

// ── Mini composants inline ────────────────────────────────────────────────────

function RouteBadge({ route }) {
  const label = route === 'change_impact' ? '⚡ Change Impact' : '🔍 Root Cause'
  const cls = `RouteBadge RouteBadge--${route}`
  return <span className={cls}>{label}</span>
}

function ConfidenceGauge({ confidence, band, escalated }) {
  const pct = Math.round((confidence ?? 0) * 100)
  const barClass = pct >= 70 ? 'success' : pct >= 40 ? 'warning' : 'danger'
  return (
    <div className="ConfidenceGauge">
      <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)', marginBottom: 4 }}>
        Confiance — <strong style={{ color: 'var(--text-primary)' }}>{band ?? ''}</strong>
        {escalated && <span style={{ color: 'var(--warning)', marginLeft: 6 }}>⚠ escaladé</span>}
      </p>
      <div className="gauge-container">
        <div className={`gauge-bar ${barClass}`} style={{ width: `${pct}%` }} />
      </div>
      <p className="gauge-value">{pct}%</p>
    </div>
  )
}

function EscalationBanner() {
  return (
    <div className="EscalationBanner">
      <p>⚠️ Réponse escaladée — confiance insuffisante, vérification humaine recommandée.</p>
    </div>
  )
}

function SourceList({ citations }) {
  return (
    <div className="SourceList">
      <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)', marginBottom: 6, fontWeight: 600 }}>
        Sources
      </p>
      <ul>
        {citations.map((c, i) => (
          <li key={i}>
            <span style={{ color: 'var(--accent)' }}>#{i + 1}</span>
            {c.url ? <a href={c.url} target="_blank" rel="noreferrer">{c.text || c.url}</a> : <span>{c.text}</span>}
          </li>
        ))}
      </ul>
    </div>
  )
}

function AgentTrace({ trace }) {
  if (!trace || trace.length === 0) return null
  return (
    <div className="AgentTrace">
      <details open>
        <summary>Trace de l'agent ({trace.length} étapes)</summary>
        <ul>
          {trace.map((t, i) => (
            <li key={i}>
              <span style={{ color: 'var(--success)' }}>✓</span>
              <strong>{t.label}</strong>
              {t.detail && <span style={{ color: 'var(--text-muted)' }}> — {t.detail}</span>}
            </li>
          ))}
        </ul>
      </details>
    </div>
  )
}

// ── Exemples de questions ─────────────────────────────────────────────────────

const EXAMPLES = [
  { id: '1', question: "Que se passe-t-il si je change httpie/cli::httpie/context.py ?", type: 'change_impact' },
  { id: '2', question: "Quel est l'impact du retrait de httpie/context.py ?", type: 'change_impact' },
  { id: '3', question: "Pourquoi l'incident 1583 s'est-il produit ?", type: 'root_cause' },
  { id: '4', question: "Quelle est la cause racine de l'incident 1583 ?", type: 'root_cause' },
]

// ── Composant principal ───────────────────────────────────────────────────────

export default function App() {
  const [loading, setLoading] = useState(false)
  const [answer, setAnswer] = useState(null)
  const [stats, setStats] = useState(null)
  const [trace, setTrace] = useState([])
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  const submit = async (question) => {
    const q = (question ?? inputRef.current?.value ?? '').trim()
    if (!q) return

    setLoading(true)
    setError(null)
    setAnswer(null)
    setTrace([])

    try {
      // Stats
      const statsResp = await fetch('/api/stats')
      if (statsResp.ok) setStats(await statsResp.json())

      // Réponse
      const resp = await fetch('/api/answer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      })

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}))
        throw new Error(err.detail || `Erreur HTTP ${resp.status}`)
      }

      const data = await resp.json()
      setAnswer(data)

      // Trace simplifiée
      const isImpact = (data.intent ?? '').includes('impact')
      setTrace([
        { step: 'investigate', label: isImpact ? 'Investigation' : 'Enquête', detail: '' },
        { step: 'draft', label: isImpact ? 'Rédaction' : 'Analyse', detail: '' },
        { step: 'critic', label: 'Critique', detail: data.status ?? '' },
      ])
    } catch (err) {
      setError(err.message)
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleKey = (e) => { if (e.key === 'Enter') submit() }

  return (
    <div className="app">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="app-header">
        <div className="header-left">
          <h1>NEXUS-DevIntel</h1>
          <p className="header-subtitle">Agentic GraphRAG</p>
        </div>
        <div className="header-right">
          <span className="repo-badge">httpie/cli</span>
          <span className="connection-status">
            <span className="dot" />
            Connecté
          </span>
        </div>
      </header>

      {/* ── Stats ──────────────────────────────────────────────────────────── */}
      {stats && (
        <section className="stats-bar">
          <div className="stats-grid">
            <StatCard title="Fichiers" value={stats.files ?? 0} icon="📄" accent="accent" />
            <StatCard title="Commits" value={stats.commits ?? 0} icon="🔀" accent="accent" />
            <StatCard title="PRs" value={stats.prs ?? 0} icon="🔃" accent="accent" />
            <StatCard title="Incidents" value={stats.incidents ?? 0} icon="🐛" accent="accent" />
            <StatCard title="Nœuds" value={stats.graph_nodes ?? 0} icon="⬡" accent="accent" />
            <StatCard title="Vecteurs" value={stats.vector_chunks ?? 0} icon="⚡" accent="accent" />
          </div>
        </section>
      )}

      {/* ── Zone de question ───────────────────────────────────────────────── */}
      <section className="question-section">
        <div className="input-wrapper">
          <input
            ref={inputRef}
            type="text"
            id="question-input"
            className="question-input"
            placeholder={loading ? 'Analyse en cours…' : 'Posez votre question sur le code…'}
            onKeyDown={handleKey}
            disabled={loading}
          />
          <button
            type="button"
            id="analyze-btn"
            className="analyze-btn"
            disabled={loading}
            onClick={() => submit()}
          >
            {loading ? 'Analyse…' : 'Analyser'}
          </button>
        </div>

        {!loading && (
          <div className="example-chips">
            <h3>Questions d'exemple</h3>
            {EXAMPLES.map((ex) => (
              <button
                key={ex.id}
                className="example-chip"
                onClick={() => submit(ex.question)}
                disabled={loading}
              >
                {ex.question.length > 50 ? ex.question.substring(0, 50) + '…' : ex.question}
              </button>
            ))}
          </div>
        )}
      </section>

      {/* ── Loading ────────────────────────────────────────────────────────── */}
      {loading && (
        <section className="loading-state">
          <p>Analyse en cours…</p>
          <div className="spinner" />
        </section>
      )}

      {/* ── Erreur ─────────────────────────────────────────────────────────── */}
      {error && !answer && (
        <section className="error-state">
          <p>❌ {error}</p>
          <button onClick={() => setError(null)} className="retry-btn">Réessayer</button>
        </section>
      )}

      {/* ── État vide ──────────────────────────────────────────────────────── */}
      {!loading && !answer && !error && (
        <section className="empty-state">
          <p>Saisissez une question ou cliquez sur un exemple pour commencer.</p>
        </section>
      )}

      {/* ── Résultat ───────────────────────────────────────────────────────── */}
      {answer && (
        <section className="result-section">
          <div className="result-grid">

            {/* Colonne principale */}
            <div className="result-main">
              {answer.intent && <RouteBadge route={answer.intent} />}
              <div className="answer-card">
                <h2>Réponse</h2>
                <p className="answer-text">{answer.answer_text}</p>

                {answer.evidence_path?.hops?.length > 0 && (
                  <div className="evidence-path">
                    <h3>EvidencePath</h3>
                    <pre className="path-text">
                      {answer.evidence_path.hops.map((h, i) =>
                        `${i + 1}. [${h.node_kind}] ${h.node_id} (${h.role})${h.rationale ? ' — ' + h.rationale : ''}`
                      ).join('\n')}
                    </pre>
                  </div>
                )}
              </div>
            </div>

            {/* Colonne latérale */}
            <div className="result-sidebar">
              <ConfidenceGauge
                confidence={answer.confidence_score}
                band={answer.confidence_band}
                escalated={answer.escalated}
              />
              {answer.escalated && <EscalationBanner />}
              {answer.citations?.length > 0 && <SourceList citations={answer.citations} />}
              <AgentTrace trace={trace} />
            </div>
          </div>

          {/* Démo sécurité */}
          <div className="security-demo">
            <button
              className="security-btn"
              onClick={() => alert('Action bloquée : écriture non autorisée dans Neo4j / Postgres / GitHub')}
            >
              🔒 Tester une action interdite
            </button>
            {answer.escalated && (
              <p className="escalade-notes">
                ⚠️ Réponse escaladée — confiance insuffisante, vérification humaine recommandée
              </p>
            )}
          </div>
        </section>
      )}
    </div>
  )
}