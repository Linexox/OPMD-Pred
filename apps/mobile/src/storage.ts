import { Capacitor, registerPlugin } from '@capacitor/core'
import { Filesystem, Directory, Encoding } from '@capacitor/filesystem'
export type LocalFile = { path: string; name: string; kind: 'photo' | 'report'; uploaded?: string; upload_request?: string }
export const MobileFiles = registerPlugin<{ pickReport(): Promise<{ path: string; name: string }>; openFile(options: {path: string; mime: string}): Promise<void>; hideKeyboard(): Promise<void> }>('MobileFiles')
const webFiles = new Map<string, string>()
const pendingWrites = new Map<string, Promise<void>>()
export async function putFile(data: string, name: string, kind: 'photo' | 'report'): Promise<LocalFile> {
  const path = 'pending/' + crypto.randomUUID() + '.' + (name.split('.').pop() || 'jpg').toLowerCase()
  if (Capacitor.isNativePlatform()) await Filesystem.writeFile({ path, data, directory: Directory.Data, recursive: true })
  else {
    const db = await webDB(); const tx = db.transaction('files', 'readwrite'); tx.objectStore('files').put(data, path)
    await transaction(tx); db.close(); webFiles.set(path, data)
  }
  return { path, name, kind }
}
function webDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => { const r = indexedDB.open('opmd-mobile-demo', 1); r.onupgradeneeded = () => r.result.createObjectStore('files'); r.onsuccess = () => resolve(r.result); r.onerror = () => reject(r.error) })
}
function transaction(tx: IDBTransaction) { return new Promise<void>((resolve, reject) => { tx.oncomplete = () => resolve(); tx.onerror = () => reject(tx.error) }) }
export async function readFile(file: LocalFile): Promise<string> {
  if (Capacitor.isNativePlatform()) return (await Filesystem.readFile({ path: file.path, directory: Directory.Data })).data as string
  if (webFiles.has(file.path)) return webFiles.get(file.path)!
  const db = await webDB(); const tx = db.transaction('files'); const r = tx.objectStore('files').get(file.path)
  const value = await new Promise<string>((resolve, reject) => { r.onsuccess = () => r.result ? resolve(r.result) : reject(new Error('草稿附件已丢失，请重新选择')); r.onerror = () => reject(r.error) }); db.close(); return value
}
export function persist(key: string, value: any): Promise<void> {
  const text = JSON.stringify(value)
  const write = (pendingWrites.get(key) || Promise.resolve()).then(() => writeSnapshot(key, text))
  // The queue always continues after a failure; the caller still receives the error.
  const completion = write.then(() => {}, () => {})
  pendingWrites.set(key, completion)
  void completion.then(() => { if (pendingWrites.get(key) === completion) pendingWrites.delete(key) })
  return write
}
async function writeSnapshot(key: string, text: string): Promise<void> {
  if (Capacitor.isNativePlatform()) {
    const temporary = key + '.' + crypto.randomUUID() + '.tmp'
    try {
      await Filesystem.writeFile({ path: temporary, data: text, directory: Directory.Data, encoding: Encoding.UTF8 })
      await Filesystem.rename({ from: temporary, to: key + '.json', directory: Directory.Data })
    } catch (cause) {
      await Filesystem.deleteFile({ path: temporary, directory: Directory.Data }).catch(() => {})
      throw new Error('手机资料保存失败，请稍后重试', { cause })
    }
  } else {
    const db = await webDB(); const tx = db.transaction('files', 'readwrite'); tx.objectStore('files').put(text, key); await transaction(tx); db.close()
  }
}
export async function restore(key: string): Promise<any> {
  try {
    if (Capacitor.isNativePlatform()) return JSON.parse((await Filesystem.readFile({ path: key + '.json', directory: Directory.Data, encoding: Encoding.UTF8 })).data as string)
    const db = await webDB(); const tx = db.transaction('files'); const r = tx.objectStore('files').get(key)
    const value = await new Promise<any>((resolve) => { r.onsuccess = () => resolve(r.result ? JSON.parse(r.result) : null); r.onerror = () => resolve(null) }); db.close(); return value
  } catch { return null }
}
