import { useEffect, useState } from 'react'
import { AlertTriangle, Bot, ChevronDown, X } from 'lucide-react'
import * as api from '../api'

type Props = { anomaly: api.Anomaly; onClose: () => void }
const labels: Record<string, string> = { temporal_score: 'Temporal', multivariate_score: 'Multivariate', isolation_score: 'Isolation forest', spatial_score: 'Spatial', rule_qc_score: 'Rule QC', gru_score: 'GRU sequence', persistence_score: 'Persistence', drift_score: 'Drift', stuck_score: 'Stuck sensor', communication_score: 'Communication', data_quality_score: 'Data quality' }

export default function AnomalyInvestigation({ anomaly, onClose }: Props) {
  const [record, setRecord] = useState(anomaly)
  const [spatial, setSpatial] = useState<api.SpatialContext | null>(null)
  const [explanation, setExplanation] = useState<api.Explanation | null>(null)
  const [loading, setLoading] = useState(true)
  const [explainLoading, setExplainLoading] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => {
    let active = true
    setRecord(anomaly); setSpatial(null); setExplanation(null); setLoading(true); setError('')
    Promise.all([api.getAnomaly(anomaly.id), api.getSpatialContext(anomaly.id)]).then(([detail, context]) => { if (active) { setRecord(detail); setSpatial(context) } }).catch(() => { if (active) setError('Some investigation evidence could not be loaded.') }).finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [anomaly.id])
  const explain = async () => {
    setExplainLoading(true); setError('')
    try { setExplanation(await api.explainAnomaly(record.id)) } catch { setError('Explanation service is unavailable.') } finally { setExplainLoading(false) }
  }
  const evidence = record.evidence_detail || {}
  return <div className="drawer-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}><aside className="ops-drawer investigation-drawer" aria-label="Anomaly investigation">
    <div className="drawer-head"><div><p className="eyebrow">PREDICTION {record.id} / INVESTIGATION</p><h2>{record.station_id}</h2><span>{new Date(record.timestamp).toLocaleString()}</span></div><button className="icon-button" onClick={onClose} aria-label="Close investigation"><X size={18} /></button></div>
    <div className="decision-banner"><span className={record.is_anomaly ? 'decision-indicator alert' : 'decision-indicator'} /><div><small>Pipeline classification</small><strong>{record.classification.replaceAll('_', ' ')}</strong></div><div className="decision-confidence"><strong>{Math.round(record.confidence * 100)}%</strong><small>Decision confidence</small></div></div>
    <p className="calibration-note">Confidence is the pipeline's decision score, not a guarantee of physical fault probability.</p>
    {error && <div className="inline-error">{error}</div>}
    <section className="drawer-section"><div className="panel-heading"><div><p className="eyebrow">DETECTOR OUTPUT</p><h3>Evidence strength</h3></div><span className="quiet-count">0–1 normalized scores</span></div><div className="evidence-bars">{Object.entries(labels).map(([key, label]) => { const value = evidence[key]; const availability = record.evidence_availability?.[key.replace('_score', '')] || record.evidence_availability?.[key]; const available = typeof value === 'number' && availability?.available !== false; return <div className={`evidence-row ${available ? '' : 'unavailable'}`} key={key}><div className="evidence-row-head"><span>{label}</span><strong>{available ? value.toFixed(3) : 'Not available'}</strong></div>{available ? <div className="score-track"><i style={{ width: `${Math.max(0, Math.min(value! * 100, 100))}%` }} /></div> : <small title={availability?.reason || 'Required detector context was not available.'}>{availability?.reason || 'Required detector context was not available.'}</small>}</div> })}</div>
    </section>
    <section className="drawer-section"><div className="panel-heading"><div><p className="eyebrow">NEIGHBOR COMPARISON</p><h3>Spatial context</h3></div><span className={`status-pill ${spatial?.spatial_status === 'sufficient' ? 'good' : 'warn'}`}>{spatial?.spatial_status || (loading ? 'loading' : 'unavailable')}</span></div>{spatial?.spatial_status === 'insufficient' && <div className="spatial-notice"><AlertTriangle size={16} /><span>{spatial.explanation} Usable neighbors: {spatial.usable_neighbors_count}; supporting: {spatial.supporting_count}; contradicting: {spatial.contradicting_count}.</span></div>}{spatial && <div className="spatial-table-wrap"><table className="spatial-table"><thead><tr><th>Station</th><th>Distance</th><th>Temp °C</th><th>Pressure hPa</th><th>RH %</th><th>Relation</th></tr></thead><tbody><tr className="target-row"><td>{spatial.target.name}<small>Target</small></td><td>—</td><td>{fmt(spatial.target.temperature_c)}</td><td>{fmt(spatial.target.pressure_hpa)}</td><td>{fmt(spatial.target.relative_humidity_pct)}</td><td>Target</td></tr>{spatial.neighbors.map((neighbor) => <tr key={neighbor.station_id}><td>{neighbor.name}<small>{neighbor.station_id}</small></td><td>{neighbor.distance_km} km</td><td>{fmt(neighbor.temperature_c)}</td><td>{fmt(neighbor.pressure_hpa)}</td><td>{fmt(neighbor.relative_humidity_pct)}</td><td>{neighbor.data_available ? neighbor.relation : 'Unavailable'}</td></tr>)}</tbody></table></div>}</section>
    <section className="drawer-section"><div className="panel-heading"><div><p className="eyebrow">OPERATOR ACTION</p><h3>Diagnosis and recommendation</h3></div></div><div className="decision-trace"><div><small>Probable root cause</small><strong>{record.root_cause || 'Inconclusive from current evidence'}</strong></div><div><small>Recommended next step</small><strong>{record.recommended_action || 'Continue monitoring; review when additional evidence is available.'}</strong></div>{record.explanation_facts?.map((fact) => <p key={fact}>{fact}</p>)}</div><button className="command-button" onClick={explain} disabled={explainLoading}><Bot size={16} />{explainLoading ? 'Preparing explanation…' : 'Generate grounded explanation'}</button>{explanation && <div className="assistant-result"><small>{explanation.fallback ? 'Evidence-based fallback' : explanation.provider} · prediction {explanation.prediction_id}</small><strong>{explanation.summary}</strong><p>{explanation.explanation}</p></div>}</section>
    <details className="raw-evidence"><summary>Raw structured evidence <ChevronDown size={15} /></summary><pre>{JSON.stringify(record.evidence, null, 2)}</pre></details>
  </aside></div>
}

function fmt(value: number | null) { return value == null ? '—' : value.toFixed(1) }
