import { Capacitor, CapacitorHttp } from '@capacitor/core'
import { Preferences } from '@capacitor/preferences'
export type Asset = { id: string; name: string; kind: string }
export type Visit = { id: string; patient_id: string; scenario: string; site: string; visit_date: string; symptoms: string; revision: number; status: string; task_id: string; payload: Record<string, any>; photo: Asset | null; reports: Asset[]; feedback: string; reviewer: string; quality: { messages?: string[] }; feedback_history: any[] }
export type Task = { id: string; patient_id: string; visit_id: string; source_date: string; site: string; kind: string; note: string; due: string; status: string; requested_items: string[]; submission_id: string }
export type ModelInputName = 'image' | 'history' | 'tct' | 'dna' | 'methylation'
export type InputContribution = { modality: ModelInputName; label: string; present: boolean; value: number }
export type ExplanationTarget = { id: string; label: string; value: number; reference_value: number; contributions: InputContribution[]; residual: number; interactions?: { first: ModelInputName; second: ModelInputName; value: number }[] }
export type DecisionCheck = ExplanationTarget & { threshold: number; passed: boolean }
export type ModelExplanation = { status: 'ready' | 'failed'; reason?: string; format?: string; default_target?: string; targets?: ExplanationTarget[]; decision_checks?: DecisionCheck[]; classifier_sha256?: string; input_revision?: number; method?: string; reference?: string }
export type ResearchResult = { status: string; reason?: string; probabilities?: number[] | null; median_class?: string; present?: Partial<Record<ModelInputName, boolean>>; explanation?: ModelExplanation; analysis_id?: string; input_revision?: number; [key: string]: unknown }
export type SavedRun = { id: string; visit_id: string; revision: number; created: string; result: ResearchResult }
export type RunSummary = { id: string; revision: number; created: string; visit_date: string; site: string; probabilities: number[]; median_class: string; relation: string }
export type RunComparison = { status: 'compared' | 'unavailable'; reason?: string; candidates: RunSummary[]; current?: RunSummary; previous?: RunSummary; probability_deltas?: number[]; changed_inputs?: { modality: ModelInputName; label: string; kind: 'added' | 'removed' | 'changed' | 'unchanged' }[]; explanation_changes?: { target: string; label: string; contributions: { modality: ModelInputName; label: string; before: number; after: number }[] }[] }
export const connection = { url: '', token: '' }
export const defaultConnection = {
  url: import.meta.env.VITE_DEMO_SERVICE_URL || '',
  token: import.meta.env.VITE_DEMO_PAIRING_TOKEN || '',
}
export async function restoreConnection() {
  const { value } = await Preferences.get({ key: 'connection' })
  Object.assign(connection, defaultConnection)
  if (value) { try { Object.assign(connection, JSON.parse(value)) } catch { /* Keep the packaged defaults. */ } }
}
export async function setConnection(url: string, token: string) {
  connection.url = url.trim().replace(/\/$/, ''); connection.token = token.trim()
  if (!/^https?:\/\//.test(connection.url)) throw new Error('请输入完整服务地址，如 http://192.168.1.5:8765')
  await Preferences.set({ key: 'connection', value: JSON.stringify(connection) })
}
export async function api<T = any>(path: string, method = 'GET', body?: any, requestId?: string): Promise<T> {
  if (!connection.url || !connection.token) throw new Error('请到“我的 → 连接设置”检查电脑连接')
  const headers: Record<string, string> = { Authorization: `Bearer ${connection.token}` }
  if (body !== undefined) { headers['Content-Type'] = 'application/json'; headers['X-Request-Id'] = requestId || crypto.randomUUID() }
  let status: number, data: any
  try {
    if (Capacitor.isNativePlatform()) {
      const result = await CapacitorHttp.request({ url: connection.url + path, method, headers, data: body, responseType: 'json', connectTimeout: 10000, readTimeout: 30000 })
      status = result.status; data = result.data
    } else {
      const response = await fetch(connection.url + path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(30000) })
      status = response.status; data = await response.text()
    }
  } catch { throw new Error('暂时连不上电脑服务，资料已保留。请检查服务是否运行后重试') }
  if (typeof data === 'string') {
    try { data = JSON.parse(data) } catch { data = null }
  }
  if (status >= 400) throw new Error(data?.error?.message || (typeof data?.detail === 'string' ? data.detail : '') || (status >= 500 ? '电脑服务处理失败，资料已保留，请稍后重试' : '操作失败，请重试'))
  if (data === null || data === undefined) throw new Error('电脑服务返回内容异常，资料已保留，请重试')
  return data as T
}
export async function photoUrl(id: string): Promise<string> {
  if (Capacitor.isNativePlatform()) {
    const result = await CapacitorHttp.get({ url: connection.url + '/api/assets/' + id, headers: { Authorization: `Bearer ${connection.token}` }, responseType: 'arraybuffer' })
    if (result.status !== 200) throw new Error('照片暂时无法读取')
    return 'data:image/png;base64,' + result.data
  }
  const response = await fetch(connection.url + '/api/assets/' + id, { headers: { Authorization: `Bearer ${connection.token}` } })
  if (!response.ok) throw new Error('照片暂时无法读取')
  return URL.createObjectURL(await response.blob())
}
export async function attachmentData(id: string): Promise<string> {
  if (Capacitor.isNativePlatform()) {
    const result = await CapacitorHttp.get({ url: connection.url + '/api/assets/' + id, headers: { Authorization: `Bearer ${connection.token}` }, responseType: 'arraybuffer' })
    if (result.status !== 200) throw new Error('附件暂时无法读取')
    return result.data
  }
  const response = await fetch(connection.url + '/api/assets/' + id, { headers: { Authorization: `Bearer ${connection.token}` } })
  if (!response.ok) throw new Error('附件暂时无法读取')
  const bytes = new Uint8Array(await response.arrayBuffer()); let text = ''; for (const b of bytes) text += String.fromCharCode(b); return btoa(text)
}
