import { useState, useRef, useEffect } from 'react'
import { StatCard } from './StatCard'
import GraphView from './GraphView'
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

// ── Ingestion : libellés d'étapes ────────────────────────────────────────────

const STEP_LABELS = {
  tree: 'Arborescence GitHub',
  imports: 'Imports (AST)',
  graph: 'Écriture Neo4j',
  history: 'Historique git',
  pulls: 'Pull requests',
  vectors: 'Chunks + embeddings',
  verify: 'Vérification',
}

const JOB_LABELS = {
  queued: 'en attente',
  running: 'en cours',
  completed: 'terminé',
  failed: 'échec',
}


// ── Composant principal ───────────────────────────────────────────────────────

export default function App({ repoId = null } = {}) {
  const [loading, setLoading] = useState(false)
  const [answer, setAnswer] = useState(null)
  const [stats, setStats] = useState(null)
  const [trace, setTrace] = useState([])
  const [error, setError] = useState(null)
  const inputRef = useRef(null)
  // Dépôt visé : champ editable, mémorisé, sinon celui du serveur.
  const [repo, setRepo] = useState(
    () => (typeof localStorage !== 'undefined' && localStorage.getItem('nexus.repo')) || repoId || '',
  )
  const [job, setJob] = useState(null)
  const [ingesting, setIngesting] = useState(false)
  // Dernière question posée : permet de la rejouer sur un autre dépôt.
  const lastQuestion = useRef('')
  // Fichiers les plus structurants du dépôt choisi → questions d'exemple pertinentes.
  const [exampleFiles, setExampleFiles] = useState([])
  const targetRepo = (repo || repoId || stats?.repository || '').trim()

  useEffect(() => {
    const r = targetRepo
    if (!r) { setExampleFiles([]); return }
    let cancelled = false
    fetch(`/api/top-files?repo_id=${encodeURIComponent(r)}&limit=5`)
      .then((res) => (res.ok ? res.json() : { files: [] }))
      .then((d) => { if (!cancelled) setExampleFiles(Array.isArray(d?.files) ? d.files : []) })
      .catch(() => { if (!cancelled) setExampleFiles([]) })
    return () => { cancelled = true }
  }, [targetRepo])

  const loadStats = async (forRepo) => {
    const wanted = (forRepo ?? targetRepo).trim()
    const query = wanted ? `?repo_id=${encodeURIComponent(wanted)}` : ''
    try {
      const resp = await fetch(`/api/stats${query}`)
      if (resp.ok) setStats(await resp.json())
    } catch (err) {
      console.error(err)
    }
  }

  useEffect(() => { loadStats() }, [])  // eslint-disable-line react-hooks/exhaustive-deps

  const commitRepo = () => {
    const cleaned = repo.trim()
    setRepo(cleaned)
    if (typeof localStorage !== 'undefined') {
      if (cleaned) localStorage.setItem('nexus.repo', cleaned)
      else localStorage.removeItem('nexus.repo')
    }
    loadStats(cleaned)
  }

  // Ingestion d'un dépôt depuis son lien, suivie étape par étape.
  const startIngest = async () => {
    const raw = repo.trim()
    if (!raw || ingesting) return
    setIngesting(true)
    setError(null)
    setAnswer(null)
    setJob(null)
    try {
      const resp = await fetch('/api/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repository: raw }),
      })
      const created = await resp.json().catch(() => ({}))
      if (!resp.ok) throw new Error(created.detail || `Erreur HTTP ${resp.status}`)
      // Le POST ne renvoie que l'accusé (job_id, statut) : on complète les
      // champs que le panneau lit, sinon le premier rendu casse.
      setJob({ steps: [], result: null, error: null, ...created })
      // Forme canonique (`owner/name`) : c'est sous cette clé que le graphe
      // est écrit, et c'est elle qu'il faut interroger — même si l'utilisateur
      // a collé un lien complet.
      if (created.repository) {
        setRepo(created.repository)
        if (typeof localStorage !== 'undefined') {
          localStorage.setItem('nexus.repo', created.repository)
        }
      }

      let current = created
      while (current.status === 'queued' || current.status === 'running') {
        await new Promise((resolve) => setTimeout(resolve, 1500))
        const poll = await fetch(`/api/ingest/${created.job_id}`)
        if (!poll.ok) break
        current = await poll.json()
        setJob(current)
      }

      if (current.status === 'completed') {
        const name = current.repository || raw
        setRepo(name)
        if (typeof localStorage !== 'undefined') localStorage.setItem('nexus.repo', name)
        await loadStats(name)
      } else if (current.status === 'failed') {
        setError(current.error || 'ingestion échouée')
      } else {
        setError('suivi du job interrompu : recharge la page pour vérifier')
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setIngesting(false)
    }
  }

  const submit = async (question, overrideRepo) => {
    const q = (question ?? inputRef.current?.value ?? '').trim()
    if (!q) return
    lastQuestion.current = q
    const repoForQuery = (overrideRepo ?? targetRepo).trim()

    setLoading(true)
    setError(null)
    setAnswer(null)
    setTrace([])

    try {
      // Stats du dépôt interrogé
      await loadStats(repoForQuery)

      // Réponse
      const resp = await fetch('/api/answer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, repo_id: repoForQuery || undefined }),
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

  // Le fichier demandé vit dans un autre dépôt : on bascule et on rejoue.
  const analyseOtherRepo = async (hintRepo) => {
    setRepo(hintRepo)
    if (typeof localStorage !== 'undefined') localStorage.setItem('nexus.repo', hintRepo)
    await loadStats(hintRepo)
    await submit(lastQuestion.current, hintRepo)
  }

  return (
    <div className="app">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="app-header">
        <div className="header-left">
          <h1>NEXUS-DevIntel</h1>
          <p className="header-subtitle">Agentic GraphRAG</p>
        </div>
        <div className="header-right">
          <input
            type="text"
            id="repo-input"
            className="repo-input"
            value={repo}
            onChange={(e) => setRepo(e.target.value)}
            onBlur={commitRepo}
            onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); commitRepo() } }}
            placeholder="lien GitHub ou owner/name"
            title="Colle un lien de dépôt, puis clique sur Ingérer"
          />
          <button
            type="button"
            id="ingest-btn"
            className="ingest-btn"
            onClick={startIngest}
            disabled={ingesting || !repo.trim()}
            title="Télécharger le dépôt, construire le graphe et indexer le code"
          >
            {ingesting ? 'Ingestion…' : 'Ingérer'}
          </button>
          <span className="connection-status">
            <span className="dot" />
            Connecté
          </span>
        </div>
      </header>

      {/* ── Progression de l'ingestion ─────────────────────────────────────── */}
      {job && (
        <section className="ingest-panel">
          <div className="ingest-head">
            <strong>{job.repository}</strong>
            <span className={`ingest-status ingest-status--${job.status}`}>
              {JOB_LABELS[job.status] ?? job.status}
            </span>
          </div>
          <ul className="ingest-steps">
            {(job.steps ?? []).map((s) => (
              <li key={s.name}>
                <span className="ingest-mark">
                  {s.status === 'done' ? '✓' : s.status === 'failed' ? '✗' : '⏳'}
                </span>
                <strong>{STEP_LABELS[s.name] ?? s.name}</strong>
                <span className="ingest-detail">{s.detail}</span>
              </li>
            ))}
          </ul>
          {job.result && (
            <p className="ingest-summary">
              {job.result.files} fichiers · {job.result.imports} IMPORTS ·{' '}
              {job.result.history?.commits ?? 0} commits · {job.result.pulls?.prs ?? 0} PRs ·{' '}
              {job.result.vectors?.chunks ?? 0} chunks · {job.result.duration_s}s
            </p>
          )}
          {job.error && <p className="ingest-error">{job.error}</p>}
        </section>
      )}

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

        {!loading && exampleFiles.length > 0 && (
          <div className="example-chips">
            <h3>Questions d'exemple pour {targetRepo}</h3>
            {exampleFiles.map((f) => {
              const q = `Que se casse-t-il si je modifie ${f.path} ?`
              return (
                <button
                  key={f.path}
                  className="example-chip"
                  onClick={() => submit(q)}
                  disabled={loading}
                  title={`${f.dependents} fichier(s) en dépendent`}
                >
                  {q.length > 60 ? q.substring(0, 60) + '…' : q}
                </button>
              )
            })}
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

                {answer.repo_hint?.repositories?.length > 0 && (
                  <div className="repo-hint">
                    <p>
                      <code>{answer.repo_hint.missing.join(', ')}</code> n'existe pas dans{' '}
                      <strong>{targetRepo || 'ce dépôt'}</strong>, mais il est ingéré ailleurs :
                    </p>
                    <div className="repo-hint-actions">
                      {answer.repo_hint.repositories.map((r) => (
                        <button
                          key={r.repo_id}
                          className="repo-hint-btn"
                          onClick={() => analyseOtherRepo(r.repo_id)}
                          disabled={loading}
                        >
                          Analyser {r.repo_id}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {(() => {
                  const hop = (answer.evidence_path?.hops ?? []).find(
                    (h) => h?.node_kind === 'File' && typeof h.node_id === 'string',
                  )
                  return hop ? <GraphView repoId={targetRepo} focus={hop.node_id} /> : null
                })()}

                {answer.evidence_path?.hops?.length > 0 && (
                  <details className="evidence-path">
                    <summary>
                      Détails techniques — les {answer.evidence_path.hops.length} preuves
                      utilisées par l'analyse
                    </summary>
                    <pre className="path-text">
                      {answer.evidence_path.hops.map((h, i) =>
                        `${i + 1}. [${h.node_kind}] ${h.node_id} (${h.role})${h.rationale ? ' — ' + h.rationale : ''}`
                      ).join('\n')}
                    </pre>
                  </details>
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

        </section>
      )}
    </div>
  )
}