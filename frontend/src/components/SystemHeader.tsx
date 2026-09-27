import { Activity, Bot, Database, Satellite } from 'lucide-react'
import type { SystemStatus } from '../api'

type Props = { status: SystemStatus | null }

function Signal({ label, value, healthy }: { label: string; value: string; healthy: boolean }) {
  return <div className="system-signal"><span className={`signal-dot ${healthy ? 'online' : 'offline'}`} /><span className="signal-label">{label}</span><strong>{value}</strong></div>
}

export default function SystemHeader({ status }: Props) {
  return <header className="ops-header">
    <a className="ops-brand" href="#overview" aria-label="AWS Anomaly Intelligence overview">
      <span className="brand-mark"><Satellite size={19} /></span>
      <span><strong>AWS Anomaly Intelligence</strong><small>Weather operations / Pune network</small></span>
    </a>
    <div className="system-signals" aria-label="System status">
      <Signal label={status?.database.dialect || 'Database'} value={status?.database.connected ? 'Connected' : 'Unavailable'} healthy={Boolean(status?.database.connected)} />
      <Signal label="ML pipeline" value={status?.ml_pipeline.status || 'Checking'} healthy={status?.ml_pipeline.status === 'online'} />
      <Signal label="Replay" value={status?.replay.state || 'Unknown'} healthy={status?.replay.state !== 'error'} />
      <Signal label="GenAI" value={status?.genai.configured ? status.genai.provider || 'Ready' : 'Fallback'} healthy={Boolean(status?.genai.configured)} />
    </div>
    <span className="header-version"><Activity size={14} /> {status?.environment || 'connecting'}</span>
  </header>
}
