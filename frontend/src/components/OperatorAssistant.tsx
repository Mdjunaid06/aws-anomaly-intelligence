import { useState } from 'react'
import { Bot, Send } from 'lucide-react'
import * as api from '../api'

type Props = { selectedPredictionId?: number }
const prompts = ['Which stations need attention?', 'What evidence supports the latest alert?', 'Summarize current network status.']
export default function OperatorAssistant({ selectedPredictionId }: Props) {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState<api.Assistant | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const ask = async (value = question) => {
    if (!value.trim() || loading) return
    setQuestion(value); setLoading(true); setError('')
    try { setAnswer(await api.askAssistant(value.trim(), selectedPredictionId)) } catch { setError('The assistant could not reach the explanation service.') } finally { setLoading(false) }
  }
  return <div className="assistant-page"><div className="page-title-row"><div><p className="eyebrow">GROUNDED OPERATIONS SUPPORT</p><h1>Operator assistant</h1><p className="page-subtitle">Answers are based on stored pipeline evidence; the assistant cannot alter ML decisions.</p></div><Bot size={22} /></div><div className="assistant-layout"><section className="panel prompt-panel"><p className="eyebrow">SUGGESTED INVESTIGATIONS</p><h2>What do you need to know?</h2><div className="prompt-list">{prompts.map((prompt) => <button key={prompt} onClick={() => ask(prompt)}>{prompt}<span>→</span></button>)}</div>{selectedPredictionId && <p className="context-note">Context includes prediction {selectedPredictionId}.</p>}</section><section className="panel assistant-panel"><div className="assistant-output">{loading ? <div className="empty-state">Reviewing structured station and anomaly records…</div> : answer ? <><div className="assistant-provenance"><span className={`signal-dot ${answer.fallback ? 'offline' : 'online'}`} />{answer.fallback ? 'Evidence-based fallback' : answer.provider}</div><p>{answer.answer}</p><details><summary>Evidence context</summary><pre>{JSON.stringify(answer.context, null, 2)}</pre></details></> : <div className="empty-state"><Bot size={24} />Responses will appear here with their evidence provenance.</div>}{error && <div className="inline-error">{error}</div>}</div><form className="assistant-input" onSubmit={(event) => { event.preventDefault(); void ask() }}><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about stations, anomalies, or evidence…" maxLength={500} /><button className="command-button" disabled={!question.trim() || loading}><Send size={15} /> Send</button></form></section></div></div>
}
