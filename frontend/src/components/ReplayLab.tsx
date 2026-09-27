import { useEffect, useState } from 'react'
import { Activity, CircleStop, Pause, Play, RotateCcw } from 'lucide-react'
import type { ReplayAction, ReplayConfig, ReplayStart, ReplayStatus } from '../api'
import * as api from '../api'

type Props = {
  config: ReplayConfig | null
  status: ReplayStatus | null
  onAction: (action: ReplayAction, body?: ReplayStart) => void
  onInvestigate: (predictionId: number) => void
}

export default function ReplayLab({ config, status, onAction, onInvestigate }: Props) {
  const [speed, setSpeed] = useState(5)
  const [station, setStation] = useState('')
  const [observation, setObservation] = useState<api.Observation | null>(null)

  useEffect(() => {
    if (status?.speed && config?.speeds.includes(status.speed)) setSpeed(status.speed)
  }, [status?.speed, config?.speeds])

  useEffect(() => {
    let active = true
    setObservation(null)
    if (!status?.current_station || !status.current_timestamp) return () => { active = false }
    api.getReplayObservation(status.current_station, status.current_timestamp)
      .then((row) => { if (active) setObservation(row) })
      .catch(() => { if (active) setObservation(null) })
    return () => { active = false }
  }, [status?.current_station, status?.current_timestamp])

  const progress = Math.max(0, Math.min(status?.progress_pct || 0, 100))
  const predictionId = Number(status?.current_prediction?.prediction_id)
  const fileName = config ? config.source.split(/[\\/]/).pop() : null
  const weatherValues = [
    ['temperature_c', 'Temperature °C'],
    ['pressure_hpa', 'Pressure hPa'],
    ['relative_humidity_pct', 'Relative humidity %'],
  ] as const

  return <div className="replay-page">
    <div className="page-title-row">
      <div><p className="eyebrow">CONTROL ROOM / HISTORICAL SIMULATION</p><h1>Replay laboratory</h1><p className="page-subtitle">Chronological observations scored through the production ML pipeline.</p></div>
      <span className={`status-pill ${status?.running ? 'good' : status?.state === 'error' ? 'warn' : ''}`}>{status?.state || 'unknown'}</span>
    </div>
    <div className="replay-layout">
      <section className="panel replay-control-panel">
        <div className="panel-heading"><div><p className="eyebrow">REPLAY CONTROL</p><h2>Historical stream</h2></div><Activity size={19} /></div>
        <label className="field-label">Station scope<select value={station} onChange={(event) => setStation(event.target.value)} disabled={status?.running || status?.paused}><option value="">All stations</option>{config?.stations.map((id) => <option key={id} value={id}>{id}</option>)}</select></label>
        <label className="field-label">Playback speed<select value={speed} onChange={(event) => setSpeed(Number(event.target.value))} disabled={status?.running || status?.paused}>{(config?.speeds || [0.5, 1, 2, 5, 10, 20, 50]).map((value) => <option key={value} value={value}>{value}×</option>)}</select></label>
        <div className="control-actions">
          <button className="command-button" onClick={() => onAction('start', { speed, station_id: station || undefined })} disabled={status?.running || status?.paused}><Play size={16} /> Start replay</button>
          <button className="icon-command" title="Pause replay" onClick={() => onAction('pause')} disabled={!status?.running}><Pause size={17} /></button>
          <button className="icon-command" title="Resume replay" onClick={() => onAction('resume')} disabled={!status?.paused}><Play size={17} /></button>
          <button className="icon-command" title="Stop replay" onClick={() => onAction('stop')} disabled={!status?.running && !status?.paused}><CircleStop size={17} /></button>
          <button className="icon-command" title="Reset replay" onClick={() => onAction('reset')} disabled={status?.running || status?.paused}><RotateCcw size={17} /></button>
        </div>
        <div className="progress-block"><div className="progress-label"><span>Dataset progress</span><strong>{progress.toFixed(1)}%</strong></div><div className="progress-track"><i style={{ width: `${progress}%` }} /></div><small>{(status?.processed_observations || 0).toLocaleString()} / {(status?.total_observations || 0).toLocaleString()} observations</small></div>
        <div className="replay-meta"><span>Dataset</span><strong>{fileName || 'Waiting for replay configuration'}</strong><span>Time range</span><strong>{config?.min_timestamp ? new Date(config.min_timestamp).toLocaleDateString() : '—'} – {config?.max_timestamp ? new Date(config.max_timestamp).toLocaleDateString() : '—'}</strong></div>
      </section>
      <section className="panel replay-live-panel">
        <div className="panel-heading"><div><p className="eyebrow">LATEST INFERENCE</p><h2>Incoming observation</h2></div><span className="quiet-count">{status?.current_station || 'Awaiting stream'}</span></div>
        {status?.current_prediction ? <>
          <div className="live-prediction"><div><small>Station / time</small><strong>{status.current_station || '—'}</strong><span>{status.current_timestamp ? new Date(status.current_timestamp).toLocaleString() : '—'}</span></div><div><small>ML classification</small><strong>{String(status.current_prediction.classification || '—').replaceAll('_', ' ')}</strong><span>{Math.round(Number(status.current_prediction.confidence || 0) * 100)}% decision confidence</span></div></div>
          <div className="replay-observation-values">{weatherValues.map(([key, label]) => <div key={key}><small>{label}</small><strong>{observation?.[key] == null ? 'Loading observation…' : observation[key]}</strong></div>)}</div>
          {predictionId > 0 && <button className="text-button" onClick={() => onInvestigate(predictionId)}>Investigate latest prediction <span>→</span></button>}
        </> : <div className="empty-state">Start a replay to watch scored observations arrive here.</div>}
        {status?.error && <div className="inline-error">{status.error}</div>}
        <div className="replay-counter-row"><span>Anomalies during this replay</span><strong>{status?.anomaly_count ?? 0}</strong></div>
      </section>
    </div>
  </div>
}
