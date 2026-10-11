import { Capacitor, CapacitorHttp } from '@capacitor/core'
import { Preferences } from '@capacitor/preferences'
export type Asset = { id: string; name: string; kind: string }
export type Visit = { id: string; patient_id: string; scenario: string; site: string; visit_date: string; symptoms: string; revision: number; status: string; task_id: string; payload: Record<string, any>; photo: Asset | null; reports: Asset[]; feedback: string; reviewer: string; quality: { messages?: string[] }; feedback_history: any[] }
export type Task = { id: string; patient_id: string; visit_id: string; source_date: string; site: string; kind: string; note: string; due: string; status: string; requested_items: string[]; submission_id: string }
export const connection = { url: '', token: '' }
export async function restoreConnection() {
  const { value } = await Preferences.get({ key: 'connection' })
  if (value) Object.assign(connection, JSON.parse(value))
}
export async function setConnection(url: string, token: string) {
  connection.url = url.trim().replace(/\/$/, ''); connection.token = token.trim()
  if (!/^https?:\/\//.test(connection.url)) throw new Error('请输入完整服务地址，如 http://192.168.1.5:8765')
  await Preferences.set({ key: 'connection', value: JSON.stringify(connection) })
}
export async function api<T = any>(path: string, method = 'GET', body?: any, requestId?: string): Promise<T> {
  if (!connection.url || !connection.token) throw new Error('请先在“我的”连接电脑服务')
  const headers: Record<string, string> = { Authorization: `Bearer ${connection.token}` }
  if (body !== undefined) { headers['Content-Type'] = 'application/json'; headers['X-Request-Id'] = requestId || crypto.randomUUID() }
  let status: number, data: any
  try {
    if (Capacitor.isNativePlatform()) {
      const result = await CapacitorHttp.request({ url: connection.url + path, method, headers, data: body, responseType: 'json', connectTimeout: 10000, readTimeout: 30000 })
      status = result.status; data = typeof result.data === 'string' ? JSON.parse(result.data) : result.data
    } else {
      const response = await fetch(connection.url + path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(30000) })
      status = response.status; data = await response.json()
    }
  } catch { throw new Error('连接中断，资料已保留。请检查电脑服务和Wi-Fi后重试') }
  if (status >= 400) throw new Error(data.error?.message || data.detail || '操作失败，请重试')
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
