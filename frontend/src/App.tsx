import { useEffect, useState } from 'react'
import { Activity, Bot, Map, Play } from 'lucide-react'
import * as api from './api'
import AnomalyInvestigation from './components/AnomalyInvestigation'
import NetworkOverview from './components/NetworkOverview'
import OperatorAssistant from './components/OperatorAssistant'
import ReplayLab from './components/ReplayLab'
import StationDrawer from './components/StationDrawer'
import SystemHeader from './components/SystemHeader'

type View = 'overview' | 'replay' | 'assistant'

export default function App() {
  const [view, setView] = useState<View>('overview')
  const [stations, setStations] = useState<api.Station[]>([])
  const [health, setHealth] = useState<api.HealthList | null>(null)
  const [anomalies, setAnomalies] = useState<api.Anomaly[]>([])
  const [status, setStatus] = useState<api.SystemStatus | null>(null)
  const [replayConfig, setReplayConfig] = useState<api.ReplayConfig | null>(null)
  const [replay, setReplay] = useState<api.ReplayStatus | null>(null)
  const [selectedStation, setSelectedStation] = useState<api.Station | null>(null)
  const [selectedAnomaly, setSelectedAnomaly] = useState<api.Anomaly | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    const refresh = async () => {
      try {
        const [system, stationResponse, healthResponse, anomalyResponse, replayStatus, config] = await Promise.all([
          api.getSystemStatus(), api.getStations(), api.getHealth(), api.getAnomalies(), api.getReplayStatus(), api.getReplayConfig(),
        ])
        if (!active) return
        setStatus(system); setStations(stationResponse.stations); setHealth(healthResponse)
        setAnomalies(anomalyResponse.items); setReplay(replayStatus); setReplayConfig(config); setError('')
      } catch (cause) {
        if (active) setError(cause instanceof Error ? cause.message : 'Backend data is temporarily unavailable.')
      }
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), 5000)
    return () => { active = false; window.clearInterval(timer) }
  }, [])

  const investigate = async (predictionId: number) => {
    try { setSelectedAnomaly(await api.getAnomaly(predictionId)); setSelectedStation(null) }
    catch { setError(`Could not load prediction ${predictionId}.`) }
  }
  const selectStation = (station: api.Station) => { setSelectedAnomaly(null); setSelectedStation(station) }
  const runReplay = async (action: api.ReplayAction, body?: api.ReplayStart) => {
    try { setReplay(await api.replayAction(action, body)); setError('') }
    catch (cause) { setError(cause instanceof Error ? cause.message : `Replay ${action} failed.`) }
  }

  return <div className="operations-app">
    <SystemHeader status={status} />
    <div className="app-frame">
      <aside className="app-sidebar">
        <div className="sidebar-section-label">MONITORING</div>
        <button className={`side-link ${view === 'overview' ? 'active' : ''}`} onClick={() => setView('overview')}><Map size={17} />Network overview</button>
        <button className={`side-link ${view === 'replay' ? 'active' : ''}`} onClick={() => setView('replay')}><Play size={17} />Replay laboratory</button>
        <button className={`side-link ${view === 'assistant' ? 'active' : ''}`} onClick={() => setView('assistant')}><Bot size={17} />Operator assistant</button>
        <div className="sidebar-bottom"><div className="sidebar-section-label">PIPELINE</div><div className="sidebar-fact"><span>Detector models</span><strong>{status?.ml_pipeline.models_loaded.length ?? '—'}</strong></div><div className="sidebar-fact"><span>Source observations</span><strong>{(status?.database.total_observations ?? 0).toLocaleString()}</strong></div><div className="sidebar-footnote"><Activity size={14} />Evidence-led station monitoring</div></div>
      </aside>
      <main className="app-main">
        {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => setError('')}>Dismiss</button></div>}
        {view === 'overview' && <NetworkOverview stations={stations} anomalies={anomalies} health={health} status={status} onStation={selectStation} onAnomaly={(anomaly) => { setSelectedStation(null); setSelectedAnomaly(anomaly) }} />}
        {view === 'replay' && <ReplayLab config={replayConfig} status={replay} onAction={runReplay} onInvestigate={(id) => void investigate(id)} />}
        {view === 'assistant' && <OperatorAssistant selectedPredictionId={selectedAnomaly?.id} />}
      </main>
    </div>
    {selectedStation && <StationDrawer station={selectedStation} onClose={() => setSelectedStation(null)} onInvestigate={(id) => void investigate(id)} />}
    {selectedAnomaly && <AnomalyInvestigation key={selectedAnomaly.id} anomaly={selectedAnomaly} onClose={() => setSelectedAnomaly(null)} />}
  </div>
}
