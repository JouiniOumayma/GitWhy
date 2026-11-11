import { useEffect, useMemo, useState } from 'react'

/**
 * Explique et dessine le voisinage réel d'un fichier dans le graphe Neo4j.
 *
 * Trois blocs, dans l'ordre de lecture d'un non-informaticien :
 *   1. un résumé en français courant, calculé à partir des compteurs du graphe ;
 *   2. la liste des fichiers concernés, chacun avec sa raison d'être là ;
 *   3. le schéma (disposition fixe : impactés à gauche, dépendances à droite,
 *      historique en bas — une flèche va de l'utilisateur vers le fichier utilisé).
 *
 * Aucune donnée n'est inventée : tout vient de ``/api/graph``.
 */

const KIND_META = {
  focus: { color: 'var(--accent)', radius: 15 },
  importer: { color: 'var(--warning)', radius: 8 },
  dependency: { color: 'var(--info)', radius: 8 },
  commit: { color: 'var(--text-secondary)', radius: 6 },
  pr: { color: 'var(--accent-hover)', radius: 6 },
}

const WIDTH = 980
const HEIGHT = 560
const CENTER = { x: WIDTH / 2, y: 250 }
const LIST_LIMIT = 10

const MONTHS = ['janv.', 'févr.', 'mars', 'avr.', 'mai', 'juin',
  'juil.', 'août', 'sept.', 'oct.', 'nov.', 'déc.']

const shortName = (path) => String(path || '').split('/').pop() || ''
const folderOf = (path) => String(path || '').split('/').slice(0, -1).join('/')
const shortText = (text, max = 26) =>
  text && text.length > max ? `${text.slice(0, max - 1)}…` : text || ''

/** ``2026-11-04`` → ``4 nov. 2026`` (sans fuseau, donc sans décalage). */
function frDate(value) {
  const [year, month, day] = String(value || '').split('-')
  const index = Number(month) - 1
  if (!year || !day || !MONTHS[index]) return value || ''
  return `${Number(day)} ${MONTHS[index]} ${year}`
}

/** Le risque, en une phrase, déduit du seul nombre de fichiers impactés. */
function riskSentence(importers) {
  if (importers === 0) {
    return "Aucun autre fichier ne s'appuie dessus : le modifier ne casse rien d'autre dans le code analysé."
  }
  if (importers <= 2) {
    return "Peu de fichiers en dépendent : une erreur ici n'aurait que peu de conséquences."
  }
  if (importers <= 7) {
    return "Plusieurs fichiers en dépendent : une erreur ici peut casser plusieurs fonctionnalités."
  }
  return "C'est un fichier central : beaucoup d'autres s'appuient dessus, une erreur ici peut casser une grande partie du projet."
}

function layout(data) {
  const positions = new Map()
  positions.set(data.focus.id, CENTER)

  const placeOnArc = (list, centerAngle, span, radius) => {
    list.forEach((node, index) => {
      const ratio = list.length === 1 ? 0 : index / (list.length - 1) - 0.5
      const angle = centerAngle + ratio * span
      positions.set(node.id, {
        x: CENTER.x + radius * Math.cos(angle),
        y: CENTER.y + radius * Math.sin(angle),
      })
    })
  }

  const importers = data.nodes.filter((n) => n.kind === 'importer')
  const dependencies = data.nodes.filter((n) => n.kind === 'dependency')
  const history = [
    ...data.nodes.filter((n) => n.kind === 'commit'),
    ...data.nodes.filter((n) => n.kind === 'pr'),
  ]

  placeOnArc(importers, Math.PI, Math.PI * 0.82, 280)
  placeOnArc(dependencies, 0, Math.PI * 0.82, 280)

  if (history.length > 0) {
    const step = Math.min(78, 800 / Math.max(history.length, 1))
    const start = CENTER.x - (step * (history.length - 1)) / 2
    history.forEach((node, index) => {
      positions.set(node.id, { x: start + index * step, y: 500 })
    })
  }

  return positions
}

function FileRow({ node, focusName, role }) {
  const name = shortName(node.path || node.label)
  const folder = folderOf(node.path || '')
  const sentence =
    role === 'importer'
      ? `Ce fichier utilise ${focusName}. Si ${focusName} change, ce fichier peut arrêter de fonctionner.`
      : `${focusName} s'appuie sur ce fichier. Si ce fichier change, ${focusName} peut arrêter de fonctionner.`
  const dot = role === 'importer' ? KIND_META.importer.color : KIND_META.dependency.color

  return (
    <li className="impact-row">
      <span className="impact-dot" style={{ background: dot }} aria-hidden="true" />
      <div className="impact-body">
        <p className="impact-name">
          {name}
          {folder && <span className="impact-folder"> · dossier {folder}</span>}
        </p>
        <p className="impact-why">{sentence}</p>
      </div>
    </li>
  )
}

function HistoryRow({ node }) {
  // Commit : ``label`` = message, ``meta`` = date. PR : ``label`` = « PR #123 »,
  // ``meta`` = titre. Les deux cas sont donc inversés.
  const isPr = node.kind === 'pr'
  const title = isPr ? node.meta : node.label
  const when = isPr ? node.label : frDate(node.meta)
  return (
    <li className="impact-row">
      <span
        className="impact-dot"
        style={{ background: isPr ? KIND_META.pr.color : KIND_META.commit.color }}
        aria-hidden
      />
      <div className="impact-body">
        <p className="impact-name">
          {when}
          <span className="impact-folder"> · {isPr ? 'proposition de modification' : 'changement'}</span>
        </p>
        {title && <p className="impact-why">{shortText(title, 110)}</p>}
      </div>
    </li>
  )
}

export default function GraphView({ repoId, focus: initialFocus }) {
  const [focus, setFocus] = useState(initialFocus)
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => setFocus(initialFocus), [initialFocus])

  useEffect(() => {
    if (!focus || !repoId) return undefined
    const controller = new AbortController()
    setLoading(true)
    setError(null)

    const nodeId = focus.includes('::') ? focus : `${repoId}::${focus}`
    const params = new URLSearchParams({ repo_id: repoId, node_id: nodeId, limit: '45' })

    fetch(`/api/graph?${params}`, { signal: controller.signal })
      .then(async (response) => {
        const payload = await response.json().catch(() => ({}))
        if (!response.ok) throw new Error(payload.detail || `Erreur HTTP ${response.status}`)
        return payload
      })
      .then((payload) => setData(payload))
      .catch((err) => {
        if (err.name !== 'AbortError') setError(err.message)
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })

    return () => controller.abort()
  }, [focus, repoId])

  const groups = useMemo(() => {
    if (!data) return null
    return {
      importers: data.nodes.filter((n) => n.kind === 'importer'),
      dependencies: data.nodes.filter((n) => n.kind === 'dependency'),
      commits: data.nodes.filter((n) => n.kind === 'commit'),
      prs: data.nodes.filter((n) => n.kind === 'pr'),
    }
  }, [data])

  if (!repoId || !initialFocus) return null

  if (error) {
    return (
      <div className="graph-panel">
        <h3>Impact dans le code</h3>
        <p className="graph-hint">⚠️ {error}</p>
      </div>
    )
  }

  if (!data || !groups) {
    return (
      <div className="graph-panel">
        <h3>Impact dans le code</h3>
        <p className="graph-hint">{loading ? 'Lecture du graphe…' : 'Aucune donnée.'}</p>
      </div>
    )
  }

  const positions = layout(data)
  const counts = data.counts || {}
  const isRoot = data.focus.id === initialFocus
  const focusName = shortName(data.focus.path)
  const importers = counts.importers ?? 0
  const dependencies = counts.dependencies ?? 0
  const commits = counts.commits ?? 0
  const prs = counts.prs ?? 0

  const summaryParts = [
    `${focusName} est utilisé par ${importers} ${importers > 1 ? 'autres fichiers' : 'autre fichier'}.`,
    dependencies > 0
      ? `Il s'appuie lui-même sur ${dependencies} ${dependencies > 1 ? 'fichiers' : 'fichier'}.`
      : "Il ne s'appuie sur aucun autre fichier du projet.",
  ]
  if (commits > 0 || prs > 0) {
    summaryParts.push(
      `Dans l'historique récent, ${commits} ${commits > 1 ? 'changements le concernent' : 'changement le concerne'}` +
        (prs > 0 ? ` et ${prs} ${prs > 1 ? 'propositions de modification y sont liées' : 'proposition de modification y est liée'}.` : '.'),
    )
  }

  return (
    <div className="graph-panel">
      <div className="graph-head">
        <h3>
          Que se passe-t-il si tu modifies <span className="graph-focus-name">{focusName}</span> ?
        </h3>
        <div className="graph-head-right">
          <code className="graph-focus-path">{data.focus.path}</code>
          {!isRoot && (
            <button type="button" className="graph-reset" onClick={() => setFocus(initialFocus)}>
              ↺ Revenir à {shortName(initialFocus)}
            </button>
          )}
        </div>
      </div>

      <p className="graph-plain">
        <span className="graph-plain-label">En clair</span>
        {summaryParts.join(' ')} {riskSentence(importers)}
      </p>

      {importers > 0 && (
        <section className="impact-group">
          <h4>
            Fichiers qui peuvent casser <span className="impact-count">{importers}</span>
          </h4>
          <p className="impact-legend">Ils utilisent {focusName} : ce sont les premiers touchés.</p>
          <ul className="impact-list">
            {groups.importers.slice(0, LIST_LIMIT).map((node) => (
              <FileRow key={node.id} node={node} focusName={focusName} role="importer" />
            ))}
          </ul>
          {groups.importers.length > LIST_LIMIT && (
            <p className="impact-more">
              + {groups.importers.length - LIST_LIMIT} autre(s) fichier(s) visible(s) sur le schéma.
            </p>
          )}
        </section>
      )}

      {dependencies > 0 && (
        <section className="impact-group">
          <h4>
            Fichiers dont {focusName} dépend <span className="impact-count">{dependencies}</span>
          </h4>
          <p className="impact-legend">
            {focusName} a besoin d'eux pour fonctionner : s'ils changent, {focusName} peut casser.
          </p>
          <ul className="impact-list">
            {groups.dependencies.slice(0, LIST_LIMIT).map((node) => (
              <FileRow key={node.id} node={node} focusName={focusName} role="dependency" />
            ))}
          </ul>
          {groups.dependencies.length > LIST_LIMIT && (
            <p className="impact-more">
              + {groups.dependencies.length - LIST_LIMIT} autre(s) fichier(s) visible(s) sur le schéma.
            </p>
          )}
        </section>
      )}

      {(commits > 0 || prs > 0) && (
        <section className="impact-group">
          <h4>
            Historique récent <span className="impact-count">{commits + prs}</span>
          </h4>
          <p className="impact-legend">
            Les derniers changements qui ont touché {focusName}, et les propositions de modification associées.
          </p>
          <ul className="impact-list">
            {groups.commits.slice(0, 4).map((node) => (
              <HistoryRow key={node.id} node={node} />
            ))}
            {groups.prs.slice(0, 3).map((node) => (
              <HistoryRow key={node.id} node={node} />
            ))}
          </ul>
        </section>
      )}

      <details className="graph-details" open>
        <summary>Voir le schéma des liens</summary>

        <ol className="graph-howto">
          <li>La pastille au centre est le fichier que tu veux modifier.</li>
          <li>À gauche, en orange : les fichiers qui l'utilisent (ils peuvent casser).</li>
          <li>À droite, en bleu : les fichiers dont il a besoin. En bas : les changements récents.</li>
        </ol>

        <svg
          className="graph-svg"
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          role="img"
          aria-label={`Liens du fichier ${data.focus.path}`}
        >
          <defs>
            <marker
              id="graph-arrow"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--text-muted)" />
            </marker>
          </defs>

          <text className="graph-zone" x={26} y={34}>UTILISÉ PAR ({importers})</text>
          <text className="graph-zone" x={WIDTH - 26} y={34} textAnchor="end">
            A BESOIN DE ({dependencies})
          </text>
          {(commits + prs) > 0 && (
            <text className="graph-zone" x={WIDTH / 2} y={HEIGHT - 24} textAnchor="middle">
              CHANGEMENTS RÉCENTS ({commits} · {prs} PROPOSITIONS)
            </text>
          )}

          <line
            x1={WIDTH / 2}
            y1={62}
            x2={WIDTH / 2}
            y2={HEIGHT - 48}
            stroke="var(--border)"
            strokeDasharray="4 6"
            strokeOpacity="0.5"
          />

          {data.edges.map((edge, index) => {
            const from = positions.get(edge.source)
            const to = positions.get(edge.target)
            if (!from || !to) return null
            const color =
              edge.type === 'IMPORTS'
                ? edge.target === data.focus.id
                  ? 'var(--warning)'
                  : 'var(--info)'
                : edge.type === 'MERGED_INTO'
                  ? 'var(--accent-hover)'
                  : 'var(--border)'
            return (
              <line
                key={`${edge.source}-${edge.target}-${edge.type}-${index}`}
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                stroke={color}
                strokeWidth={edge.type === 'IMPORTS' ? 1.6 : 1.1}
                strokeOpacity={edge.type === 'IMPORTS' ? 0.75 : 0.5}
                markerEnd="url(#graph-arrow)"
              />
            )
          })}

          {data.nodes.map((node) => {
            const position = positions.get(node.id)
            if (!position) return null
            const meta = KIND_META[node.kind] || KIND_META.commit
            const isFocus = node.kind === 'focus'
            const isFile = node.kind === 'importer' || node.kind === 'dependency'
            const anchor = isFocus
              ? 'middle'
              : position.x < CENTER.x - 40 ? 'end' : position.x > CENTER.x + 40 ? 'start' : 'middle'
            const dx = anchor === 'end' ? -16 : anchor === 'start' ? 16 : 0
            const dy = isFocus ? -24 : position.y > 460 ? 22 : 5
            const label = isFocus || isFile ? shortName(node.path || node.label) : shortText(node.label, 22)

            return (
              <g
                key={node.id}
                className={`graph-node${isFile || isFocus ? ' graph-node--clickable' : ''}`}
                onClick={() => (isFile || isFocus) && node.path && setFocus(`${repoId}::${node.path}`)}
              >
                <title>{`${node.label}${node.meta ? `\n${node.meta}` : ''}`}</title>
                <circle
                  cx={position.x}
                  cy={position.y}
                  r={meta.radius}
                  fill={meta.color}
                  fillOpacity={isFocus ? 0.95 : 0.8}
                  stroke={isFocus ? 'var(--text-primary)' : 'var(--bg)'}
                  strokeWidth={isFocus ? 2 : 1}
                />
                <text
                  className={`graph-label${isFocus ? ' graph-label--focus' : ''}`}
                  x={position.x + dx}
                  y={position.y + dy}
                  textAnchor={anchor}
                >
                  {label}
                </text>
              </g>
            )
          })}
        </svg>

        <div className="graph-legend">
          {Object.entries({
            focus: 1,
            importer: importers,
            dependency: dependencies,
            commit: commits,
            pr: prs,
          }).map(([kind, count]) => (
            <span key={kind} className="graph-legend-item">
              <span className="graph-dot" style={{ background: KIND_META[kind].color }} />
              {kind === 'focus' ? 'Fichier analysé' : null}
              {kind === 'importer' ? 'Utilisé par' : null}
              {kind === 'dependency' ? 'A besoin de' : null}
              {kind === 'commit' ? 'Changements' : null}
              {kind === 'pr' ? 'Propositions (PR)' : null}
              {' · '}
              {count}
            </span>
          ))}
        </div>

        <p className="graph-hint">
          Une flèche va toujours du fichier qui utilise vers le fichier utilisé. Clique sur un
          fichier pour explorer ses propres liens.
        </p>
      </details>
    </div>
  )
}
