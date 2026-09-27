import axios from 'axios'

const api = axios.create({ baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000' })

export const getSystemStatus = () => api.get<SystemStatus>('/system/status').then((r) => r.data)
export const getStations = () => api.get<{ stations: Station[] }>('/stations/').then((r) => r.data)
export const getStationHistory = (stationId: string, range: HistoryRange) => api.get<StationHistory>(`/stations/${stationId}/history`, { params: { range } }).then((r) => r.data)
export const getReplayObservation = (stationId: string, timestamp: string) => api.get<ObservationList>(`/observations/station/${encodeURIComponent(stationId)}`, { params: { start_time: timestamp, end_time: timestamp, limit: 1 } }).then((r) => r.data.items[0] || null)
export const getHealth = () => api.get<HealthList>('/health/').then((r) => r.data)
export const getAnomalies = () => api.get<AnomalyList>('/anomalies/').then((r) => r.data)
export const getAnomaly = (id: number) => api.get<Anomaly>(`/anomalies/${id}`).then((r) => r.data)
export const getSpatialContext = (id: number) => api.get<SpatialContext>(`/anomalies/${id}/spatial-context`).then((r) => r.data)
export const getReplayConfig = () => api.get<ReplayConfig>('/replay/config').then((r) => r.data)
export const getReplayStatus = () => api.get<ReplayStatus>('/replay/status').then((r) => r.data)
export const replayAction = (action: ReplayAction, body?: ReplayStart) => api.post<ReplayStatus>(`/replay/${action}`, body).then((r) => r.data)
export const explainAnomaly = (id: number) => api.post<Explanation>('/assistant/explain', { prediction_id: id }).then((r) => r.data)
export const askAssistant = (question: string, predictionId?: number) => api.post<Assistant>('/assistant/chat', { question, prediction_id: predictionId }).then((r) => r.data)

export type HistoryRange = '6h' | '24h' | '7d' | '30d'
export type ReplayAction = 'start' | 'pause' | 'resume' | 'stop' | 'reset'
export interface Station { station_id: string; name: string; latitude: number; longitude: number; elevation_m: number; health_state: string; health_score: number | null; latest_timestamp: string | null; latest_anomaly: boolean }
export interface Health { station_id: string; overall_health: string; health_score: number | null; health_metrics?: Record<string, unknown> }
export interface HealthList { total: number; operational: number; degraded: number; failed: number; items: Health[] }
export interface EvidenceAvailability { available: boolean; status?: string; reason?: string | null }
export interface Anomaly { id: number; observation_id: number; station_id: string; timestamp: string; is_anomaly: number; confidence: number; classification: string; evidence_detail: Record<string, number | null>; evidence?: Record<string, unknown> | null; evidence_availability?: Record<string, EvidenceAvailability> | null; root_cause?: string | null; recommended_action?: string | null; explanation_facts?: string[] | null; supporting_stations?: string[] | null; contradicting_stations?: string[] | null; affected_stations?: string[] | null }
export interface AnomalyList { total: number; anomaly_count: number; items: Anomaly[] }
export interface StationHistoryPoint { observation_id: number; prediction_id: number | null; timestamp: string; temperature_c: number | null; pressure_hpa: number | null; relative_humidity_pct: number | null; is_anomaly: boolean; classification: string; confidence: number | null; root_cause: string | null }
export interface StationHistory { station_id: string; name: string; range: HistoryRange; total_points: number; anomaly_points: number; observations: StationHistoryPoint[] }
export interface ObservationList { total: number; limit: number; offset: number; items: Observation[] }
export interface Observation { id: number; station_id: string; timestamp: string; temperature_c: number | null; pressure_hpa: number | null; relative_humidity_pct: number | null; data_source: string; is_injected: number; is_processed: number }
export interface SpatialNeighbor { station_id: string; name: string; distance_km: number; observation_timestamp: string | null; temperature_c: number | null; pressure_hpa: number | null; relative_humidity_pct: number | null; delta_temperature_c: number | null; delta_pressure_hpa: number | null; delta_relative_humidity_pct: number | null; relation: string; data_available: boolean }
export interface SpatialContext { target: { prediction_id: number; observation_id: number; station_id: string; name: string; timestamp: string; temperature_c: number | null; pressure_hpa: number | null; relative_humidity_pct: number | null; classification: string; confidence: number; is_anomaly: number; root_cause: string | null }; spatial_status: 'sufficient' | 'insufficient'; explanation: string; usable_neighbors_count: number; supporting_count: number; contradicting_count: number; neighbors: SpatialNeighbor[] }
export interface ReplayConfig { source: string; speeds: number[]; stations: string[]; min_timestamp?: string | null; max_timestamp?: string | null }
export interface ReplayStart { station_id?: string; start_time?: string; end_time?: string; speed: number; include_spatial?: boolean }
export interface ReplayStatus { state: string; running: boolean; paused: boolean; start_time?: string | null; end_time?: string | null; current_timestamp?: string | null; speed: number; processed_observations: number; total_observations: number; progress_pct: number; anomaly_count: number; current_station?: string | null; current_prediction?: Record<string, unknown> | null; current_health?: Record<string, unknown> | null; latest_spatial_evidence?: Record<string, unknown> | null; error?: string | null }
export interface SystemStatus { status: string; service: string; version: string; environment: string; database: { connected: boolean; dialect: string; total_observations: number; total_anomalies: number; monitored_stations: number }; ml_pipeline: { status: string; models_loaded: string[]; models_dir: string; frozen: boolean }; replay: { state: string; running: boolean; paused: boolean; speed: number; processed: number; total: number; progress_pct: number; current_station: string | null; current_timestamp: string | null }; genai: { configured: boolean; enabled: boolean; provider: string | null; model: string | null; fallback_active: boolean } }
export interface Explanation { prediction_id: number; provider: string; fallback: boolean; summary: string; explanation: string; contributing_signals: string[]; root_cause?: string | null; recommended_action?: string | null; sensor_health?: Record<string, unknown> | null }
export interface Assistant { provider: string; fallback: boolean; answer: string; context: Record<string, unknown> }
