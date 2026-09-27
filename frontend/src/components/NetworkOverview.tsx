import { useMemo } from 'react'
import { Activity, AlertTriangle, ArrowUpRight, CheckCircle2, MapPin, ShieldAlert } from 'lucide-react'
import { CircleMarker, MapContainer, Popup, TileLayer } from 'react-leaflet'
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { Anomaly, HealthList, Station, SystemStatus } from '../api'

type Props = { stations: Station[]; anomalies: Anomaly[]; health: HealthList | null; status: SystemStatus | null; onStation: (station: Station) => void; onAnomaly: (anomaly: Anomaly) => void }

const tone = (value?: string | null) => /watch|degrad|fault|critical|failed/i.test(value || '') ? 'warn' : /healthy|operational/i.test(value || '') ? 'good' : 'neutral'

export default function NetworkOverview({ stations, anomalies, health, status, onStation, onAnomaly }: Props) {
  const chartRows = useMemo(() => anomalies.slice(0, 20).slice().reverse().map((item) => ({ label: new Date(item.timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }), confidence: Math.round(item.confidence * 100) })), [anomalies])
  const watchCount = health?.items.filter((item) => /watch|degrad|failed|critical/i.test(item.overall_health)).length || 0
  const healthyCount = health?.items.filter((item) => /healthy|operational/i.test(item.overall_health)).length || 0
  return <div className="overview-page">
    <div className="page-title-row"><div><p className="eyebrow">NETWORK / LIVE OPERATIONS</p><h1>Station overview</h1><p className="page-subtitle">Weather observations, quality signals, and station condition.</p></div><span className="data-stamp"><span className="signal-dot online" /> Refreshes every 5 seconds</span></div>
    <section className="kpi-strip" aria-label="Network summary">
      <article className="kpi"><MapPin /><span>Stations monitored</span><strong>{stations.length}</strong></article>
      <article className="kpi alert-kpi"><AlertTriangle /><span>Active anomalies</span><strong>{status?.database.total_anomalies ?? anomalies.filter((a) => a.is_anomaly).length}</strong></article>
      <article className="kpi watch-kpi"><ShieldAlert /><span>Stations on watch</span><strong>{watchCount}</strong></article>
      <article className="kpi good-kpi"><CheckCircle2 /><span>Healthy / operational</span><strong>{healthyCount}</strong></article>
      <article className="kpi"><ActivityIcon /><span>Observations stored</span><strong>{(status?.database.total_observations ?? 0).toLocaleString()}</strong></article>
    </section>
    <div className="overview-grid">
      <section className="panel network-panel"><div className="panel-heading"><div><p className="eyebrow">GEOGRAPHIC CONTEXT</p><h2>Station network</h2></div><span className="quiet-count">{stations.length} locations</span></div>
        <div className="ops-map-wrap"><MapContainer center={[18.2, 74.2]} zoom={7} scrollWheelZoom className="ops-map"><TileLayer attribution="&copy; OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />{stations.filter((s) => Number.isFinite(s.latitude) && Number.isFinite(s.longitude)).map((station) => <CircleMarker key={station.station_id} center={[station.latitude, station.longitude]} radius={station.latest_anomaly ? 9 : 7} pathOptions={{ color: station.latest_anomaly ? '#c94f48' : tone(station.health_state) === 'warn' ? '#c38a31' : '#168c83', fillColor: station.latest_anomaly ? '#c94f48' : tone(station.health_state) === 'warn' ? '#c38a31' : '#168c83', fillOpacity: .9, weight: 2 }} eventHandlers={{ click: () => onStation(station) }}><Popup><strong>{station.name}</strong><br />{station.station_id}<br />{station.health_state}{station.latest_timestamp ? <><br />{new Date(station.latest_timestamp).toLocaleString()}</> : null}</Popup></CircleMarker>)}</MapContainer></div>
        <div className="map-legend"><span><i className="legend-dot teal" />Operational</span><span><i className="legend-dot amber" />Watch</span><span><i className="legend-dot red" />Latest anomaly</span></div>
      </section>
      <section className="panel chart-panel"><div className="panel-heading"><div><p className="eyebrow">MODEL OUTPUT</p><h2>Decision confidence</h2></div><span className="quiet-count">Recent stored records</span></div>
        {chartRows.length ? <ResponsiveContainer width="100%" height={250}><AreaChart data={chartRows} margin={{ top: 12, right: 12, left: -18, bottom: 0 }}><defs><linearGradient id="opsConfidence" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#d16d50" stopOpacity={.34} /><stop offset="100%" stopColor="#d16d50" stopOpacity={.02} /></linearGradient></defs><XAxis dataKey="label" tickLine={false} axisLine={false} /><YAxis domain={[0, 100]} unit="%" tickLine={false} axisLine={false} /><Tooltip formatter={(value) => [`${value}%`, 'Confidence']} /><Area type="monotone" dataKey="confidence" stroke="#bf5f46" strokeWidth={2} fill="url(#opsConfidence)" /></AreaChart></ResponsiveContainer> : <div className="empty-state">No stored predictions yet.</div>}
      </section>
    </div>
    <div className="overview-grid lower-overview">
      <section className="panel"><div className="panel-heading"><div><p className="eyebrow">DECISION BASIS</p><h2>Multi-signal assessment</h2></div></div><p className="explain-copy">A single unusual value is not enough to call a sensor faulty. The pipeline combines station history, multivariate consistency, quality checks, and nearby-station evidence when valid observations are available.</p><div className="signal-tags"><span>Temporal</span><span>Multivariate</span><span>Spatial</span><span>Rule QC</span><span>Evidence fusion</span></div><div className="differentiation"><strong>SIH differentiation</strong><p>Spatial coverage and temporal alignment affect confidence; missing neighbor readings are not counted as disagreement.</p></div></section>
      <section className="panel alert-table-panel"><div className="panel-heading"><div><p className="eyebrow">INVESTIGATION QUEUE</p><h2>Recent anomalies</h2></div><span className="quiet-count">{anomalies.length} records</span></div>{anomalies.length ? <div className="alert-table">{anomalies.slice(0, 6).map((item) => <button className="alert-table-row" key={item.id} onClick={() => onAnomaly(item)}><span className={`table-state ${tone(item.classification)}`} /><span className="alert-row-copy"><strong>{item.station_id}</strong><small>{item.classification.replaceAll('_', ' ')}</small></span><span className="alert-row-time">{new Date(item.timestamp).toLocaleString()}</span><strong className="alert-percent">{Math.round(item.confidence * 100)}%</strong><ArrowUpRight size={16} /></button>)}</div> : <div className="empty-state">No anomaly records are available.</div>}</section>
    </div>
  </div>
}

function ActivityIcon() { return <Activity /> }
