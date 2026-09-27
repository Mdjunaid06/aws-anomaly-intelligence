import { useEffect, useState } from 'react'
import { Activity, AlertTriangle, X } from 'lucide-react'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import * as api from '../api'

type Props = { station: api.Station; onClose: () => void; onInvestigate: (predictionId: number) => void }
type WeatherKey = 'temperature_c' | 'pressure_hpa' | 'relative_humidity_pct'
const ranges: api.HistoryRange[] = ['6h', '24h', '7d', '30d']
const trends: { key: WeatherKey; label: string; unit: string; color: string }[] = [
  { key: 'temperature_c', label: 'Temperature', unit: '°C', color: '#ed896b' },
  { key: 'pressure_hpa', label: 'Pressure', unit: 'hPa', color: '#78b9d3' },
  { key: 'relative_humidity_pct', label: 'Relative humidity', unit: '%', color: '#5bd6be' },
]
const format = (value: string) => new Date(value).toLocaleString()

export default function StationDrawer({ station, onClose, onInvestigate }: Props) {
  const [range, setRange] = useState<api.HistoryRange>('24h')
  const [history, setHistory] = useState<api.StationHistory | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let active = true
    setHistory(null); setError('')
    api.getStationHistory(station.station_id, range).then((result) => { if (active) setHistory(result) }).catch(() => { if (active) setError('History could not be loaded.') })
    return () => { active = false }
  }, [station.station_id, range])
  const last = history?.observations[history.observations.length - 1]
  return <div className="drawer-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}><aside className="ops-drawer" aria-label={`Station ${station.station_id} details`}>
    <div className="drawer-head"><div><p className="eyebrow">STATION DETAIL</p><h2>{station.name || station.station_id}</h2><span>{station.station_id} · {station.latitude.toFixed(3)}, {station.longitude.toFixed(3)}</span></div><button className="icon-button" onClick={onClose} aria-label="Close station details"><X size={18} /></button></div>
    <div className="station-status-line"><span className={`table-state ${station.latest_anomaly ? 'warn' : 'good'}`} />{station.health_state.replaceAll('_', ' ')}<span className="status-divider" />Health {station.health_score == null ? 'not reported' : `${Math.round(station.health_score * 100)}%`}</div>
    <div className="station-latest-grid"><Metric label="Temperature" value={last?.temperature_c == null ? '—' : `${last.temperature_c.toFixed(1)} °C`} /><Metric label="Pressure" value={last?.pressure_hpa == null ? '—' : `${last.pressure_hpa.toFixed(1)} hPa`} /><Metric label="Humidity" value={last?.relative_humidity_pct == null ? '—' : `${last.relative_humidity_pct.toFixed(1)}%`} /></div>
    <section className="drawer-section"><div className="panel-heading"><div><p className="eyebrow">OBSERVATION HISTORY</p><h3>Weather trends</h3></div><div className="segmented">{ranges.map((item) => <button key={item} className={range === item ? 'selected' : ''} onClick={() => setRange(item)}>{item}</button>)}</div></div>
      {error ? <div className="empty-state">{error}</div> : !history ? <div className="empty-state">Loading station history…</div> : history.observations.length ? <div className="trend-stack">{trends.map((trend) => <TrendLine key={trend.key} data={history.observations} dataKey={trend.key} label={trend.label} unit={trend.unit} color={trend.color} />)}</div> : <div className="empty-state"><Activity size={20} />No observations in this time range.</div>}
    </section>
    <section className="drawer-section"><div className="panel-heading"><div><p className="eyebrow">QUALITY HISTORY</p><h3>Anomaly timeline</h3></div><span className="quiet-count">{history?.anomaly_points ?? 0} in range</span></div><div className="timeline-list">{history?.observations.filter((item) => item.is_anomaly).map((item) => <div className="timeline-item" key={item.observation_id}><AlertTriangle size={16} /><div><strong>{item.classification.replaceAll('_', ' ')}</strong><small>{format(item.timestamp)} · {item.confidence == null ? 'confidence unavailable' : `${Math.round(item.confidence * 100)}% confidence`}</small></div><button className="text-button" onClick={() => item.prediction_id && onInvestigate(item.prediction_id)} disabled={!item.prediction_id}>Investigate</button></div>) || null}{history && !history.observations.some((item) => item.is_anomaly) && <p className="empty-inline">No anomalous observations recorded in this range.</p>}</div></section>
  </aside></div>
}

function TrendLine({ data, dataKey, label, unit, color }: { data: api.StationHistoryPoint[]; dataKey: WeatherKey; label: string; unit: string; color: string }) {
  return <div className="trend-item"><div className="trend-label"><i style={{ background: color }} /><strong>{label}</strong><span>{unit}</span></div><ResponsiveContainer width="100%" height={104}><LineChart data={data} margin={{ top: 4, right: 10, left: -22, bottom: 0 }}><CartesianGrid strokeDasharray="3 3" stroke="#29393a" /><XAxis dataKey="timestamp" tickFormatter={(value) => new Date(value).toLocaleDateString(undefined, { day: 'numeric', month: 'short' })} minTickGap={28} /><YAxis unit={unit} width={48} /><Tooltip labelFormatter={(value) => format(String(value))} formatter={(value) => [value == null ? 'Not available' : `${value} ${unit}`, label]} /><Line type="monotone" dataKey={dataKey} name={label} stroke={color} strokeWidth={2} dot={false} connectNulls /></LineChart></ResponsiveContainer></div>
}

function Metric({ label, value }: { label: string; value: string }) { return <div className="station-metric"><span>{label}</span><strong>{value}</strong></div> }
