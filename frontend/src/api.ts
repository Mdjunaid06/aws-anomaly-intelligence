import axios from 'axios'

const api = axios.create({ baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000' })
export const getStations = () => api.get<{ stations: Station[] }>('/stations/').then((r) => r.data)
export const getHealth = () => api.get<HealthList>('/health/').then((r) => r.data)
export const getAnomalies = () => api.get<AnomalyList>('/anomalies/').then((r) => r.data)
export const getAnomaly = (id: number) => api.get<Anomaly>(`/anomalies/${id}`).then((r) => r.data)
export const getReplayConfig = () => api.get<ReplayConfig>('/replay/config').then((r) => r.data)
export const getReplayStatus = () => api.get<ReplayStatus>('/replay/status').then((r) => r.data)
export const replayAction = (action: string, body?: unknown) => api.post<ReplayStatus>(`/replay/${action}`, body).then((r) => r.data)
export const explainAnomaly = (id: number) => api.post<Explanation>('/assistant/explain', { prediction_id: id }).then((r) => r.data)
export const askAssistant = (question: string, predictionId?: number) => api.post<Assistant>('/assistant/chat', { question, prediction_id: predictionId }).then((r) => r.data)

export interface Station { station_id: string; name: string; latitude: number; longitude: number; elevation_m: number; health_state: string; health_score: number | null; latest_timestamp: string | null; latest_anomaly: boolean }
export interface Health { station_id: string; overall_health: string; health_score: number | null; health_metrics?: Record<string, unknown> }
export interface HealthList { total: number; operational: number; degraded: number; failed: number; items: Health[] }
export interface Anomaly { id: number; observation_id: number; station_id: string; timestamp: string; is_anomaly: number; confidence: number; classification: string; evidence_detail: Record<string, number | null>; evidence?: Record<string, unknown>; root_cause?: string; recommended_action?: string; supporting_stations?: string[]; contradicting_stations?: string[]; affected_stations?: string[]; }
export interface AnomalyList { total: number; anomaly_count: number; items: Anomaly[] }
export interface ReplayConfig { source: string; speeds: number[]; stations: string[]; min_timestamp?: string; max_timestamp?: string }
export interface ReplayStatus { state: string; running: boolean; paused: boolean; start_time?: string; end_time?: string; current_timestamp?: string; speed: number; processed_observations: number; anomaly_count: number; current_station?: string; current_prediction?: Record<string, unknown>; current_health?: Record<string, unknown>; latest_spatial_evidence?: Record<string, unknown>; error?: string }
export interface Explanation { prediction_id: number; provider: string; fallback: boolean; summary: string; explanation: string; contributing_signals: string[]; root_cause?: string; recommended_action?: string; sensor_health?: Record<string, unknown> }
export interface Assistant { provider: string; fallback: boolean; answer: string; context: Record<string, unknown> }
