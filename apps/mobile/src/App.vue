<script setup lang="ts">
import { computed, nextTick, onMounted, reactive, ref, watch } from 'vue'
import { Capacitor } from '@capacitor/core'
import { App as NativeApp } from '@capacitor/app'
import { Camera } from '@capacitor/camera'
import { Filesystem, Directory } from '@capacitor/filesystem'
import { Preferences } from '@capacitor/preferences'
import Icon from './Icon.vue'
import { api, connection, photoUrl, attachmentData, restoreConnection, setConnection, type Visit, type Task } from './api'
import { MobileFiles, persist, putFile, readFile, restore, type LocalFile } from './storage'

type Draft = { patient_id: string; local_key?: string; id?: string; revision?: number; scenario: string; site: string; visit_date: string; symptoms: string; payload: Record<string, any>; task_id: string; files: LocalFile[]; original_photo?: any; original_reports?: any[]; step: number; save_request?: string; saved?: Visit }
const page = ref('home'), busy = ref(false), error = ref(''), toast = ref(''), online = ref(false)
const patients = ref<{ id: string; alias: string }[]>([]), patientId = ref(''), visits = ref<Visit[]>([]), tasks = ref<Task[]>([])
const options = reactive<{ sites: string[]; history: any[] }>({ sites: ['右舌侧缘', '左舌侧缘', '舌背', '左颊黏膜', '右颊黏膜', '牙龈', '其他（备注说明）'], history: [] })
const current = ref<Visit | null>(null), draft = ref<Draft | null>(null), drafts = ref<Record<string, Draft>>({}), reviewDrafts = ref<Record<string, any>>({})
const review = ref<any>({ reviewer: '', feedback: '', requested_items: [], material_note: '', due: '', task_note: '' })
const run = ref<any>(null), runs = ref<any[]>([]), images = reactive<Record<string, string>>({})
const connectForm = reactive({ url: '', token: '' }), alias = ref(''), role = ref('patient'), showHelp = ref(false), showClinicalHelp = ref(false), savingDraft = ref(false)
const entryKey = computed(() => draft.value ? draft.value.id || `new:${draft.value.patient_id}` : '')
const activePatient = computed(() => patients.value.find(p => p.id === patientId.value))
const nextTask = computed(() => tasks.value.find(t => t.status === 'pending') || tasks.value.find(t => t.status === 'submitted'))
const recentFeedback = computed(() => visits.value.find(v => v.feedback))
const today = () => new Date().toLocaleDateString('sv-SE')
const statusText: Record<string, string> = { draft: '草稿', submitted: '等待医生', needs_more: '待补资料', reviewed: '已反馈', pending: '待补充', confirmed: '已确认', patient_reported: '自报已复诊', completed: '自报已复诊' }
const itemText: Record<string, string> = { photo: '照片', report: '报告', history: '病史', other: '其他' }
const cleanNote = (s: string) => s.replace(/【(?:模拟任务|虚构案例|模拟反馈，非真实医生意见)】/g, '').trim()
const isDemo = computed(() => activePatient.value?.alias.startsWith('模拟：'))
const doctorMode = computed(() => role.value === 'doctor' && ['doctor', 'research'].includes(page.value))
const draftKey = (v: Draft) => v.local_key || v.id || `new:${v.patient_id}`
let persistQueue = Promise.resolve(), toastTimer: any, pollTimer: any
function notify(s: string) { toast.value = s; clearTimeout(toastTimer); toastTimer = setTimeout(() => toast.value = '', 3500) }
function queuePersist() {
  savingDraft.value = true
  const data = JSON.parse(JSON.stringify({ drafts: drafts.value, reviewDrafts: reviewDrafts.value }))
  persistQueue = persistQueue.catch(() => {}).then(() => persist('drafts', data)).then(() => { savingDraft.value = false }).catch(() => { error.value = '草稿保存失败，请保持页面打开后重试'; savingDraft.value = false })
  return persistQueue
}
watch(draft, (v) => { if (v) { drafts.value[draftKey(v)] = JSON.parse(JSON.stringify(v)); queuePersist() } }, { deep: true })
watch(review, (v) => { if (current.value && page.value === 'doctor') { reviewDrafts.value[current.value.id] = JSON.parse(JSON.stringify(v)); queuePersist() } }, { deep: true })
watch(patientId, async (id, previous) => { if (id && id !== previous) { await Preferences.set({ key: 'patient', value: id }); await refresh() } })
async function act(fn: () => Promise<any>) { if (busy.value) return; (document.activeElement as HTMLElement)?.blur(); if(Capacitor.isNativePlatform()) MobileFiles.hideKeyboard().catch(()=>{}); busy.value = true; error.value = ''; try { return await fn() } catch (e: any) { error.value = e.message || '操作失败'; await nextTick(); document.querySelector<HTMLElement>('.error')?.focus() } finally { busy.value = false } }
async function refresh() {
  try {
    if (!patientId.value) return
    const [v, t] = await Promise.all([api<Visit[]>(`/api/patients/${patientId.value}/visits`), api<Task[]>(`/api/patients/${patientId.value}/tasks`)])
    visits.value = v; tasks.value = t; online.value = true
    await persist('cache', { patients: patients.value, patientId: patientId.value, visits: v, tasks: t })
    for (const row of v) if (row.photo && !images[row.photo.id]) photoUrl(row.photo.id).then(url => images[row.photo!.id] = url).catch(() => {})
  } catch (e: any) { online.value = false; error.value = e.message }
}
async function connect() { await act(async () => { await setConnection(connectForm.url, connectForm.token); await api('/api/health'); online.value = true; patients.value = await api('/api/patients'); Object.assign(options, await api('/api/options')); if (!patients.value.some(p => p.id === patientId.value)) patientId.value = patients.value[0]?.id || ''; await refresh(); notify('已连接电脑'); page.value = 'home' }) }
async function addPatient() { await act(async () => { const p = await api('/api/patients', 'POST', { alias: alias.value }); patients.value = await api('/api/patients'); patientId.value = p.id; alias.value = ''; notify('档案已建立'); page.value = 'home' }) }
async function loadDemo() { await act(async () => { const result = await api('/api/demo-fixtures', 'POST', {}); patients.value = await api('/api/patients'); patientId.value = result.patient_id; await refresh(); page.value = 'home'; notify('已打开模拟档案') }) }
function goto(p: string) { (document.activeElement as HTMLElement)?.blur(); if(Capacitor.isNativePlatform()) MobileFiles.hideKeyboard().catch(()=>{}); error.value = ''; if (p !== 'research') clearTimeout(pollTimer); page.value = p; window.scrollTo(0, 0) }
function newRecord() {
  if (!patientId.value) { goto('mine'); notify('先建立一个匿名档案'); return }
  const key = `new:${patientId.value}`
  draft.value = drafts.value[key] || { patient_id: patientId.value, scenario: 'self_check', site: '右舌侧缘', visit_date: today(), symptoms: '', payload: {}, task_id: '', files: [], step: 0 }
  for (const f of draft.value.files) localPreview(f).catch(() => {})
  goto('editor')
}
async function openVisit(v: Visit, doctor = false) {
  await act(async () => { current.value = await api<Visit>(`/api/visits/${v.id}`); if (doctor) { role.value = 'doctor'; review.value = reviewDrafts.value[v.id] || { reviewer: '', feedback: '', requested_items: [], material_note: '', due: '', task_note: '' }; goto('doctor') } else goto('detail') })
}
function editVisit(v: Visit) {
  draft.value = drafts.value[v.id] || { patient_id: v.patient_id, id: v.id, revision: v.revision, scenario: v.scenario, site: v.site, visit_date: v.visit_date, symptoms: v.symptoms, payload: { ...v.payload }, task_id: v.task_id, files: [], original_photo: v.photo, original_reports: v.reports, step: 0 }
  for (const f of draft.value.files) localPreview(f).catch(() => {})
  goto('editor')
}
async function startTask(t: Task) {
  if (t.kind === 'appointment') { await act(async () => { await api(`/api/tasks/${t.id}/completions`, 'POST', {}); await refresh(); notify('已记录：患者自报复诊完成') }); return }
  if (t.status !== 'pending') { notify(t.status === 'submitted' ? '资料已提交，等待医生确认' : '这项资料已确认'); return }
  await act(async () => { const v = await api<Visit>(`/api/tasks/${t.id}/submissions`, 'POST', {}); editVisit(v); if (draft.value) draft.value.step = 1 })
}
function invalidate() { if (draft.value) { draft.value.payload.report_confirmed = false; draft.value.saved = undefined; draft.value.save_request = undefined } }
async function localPreview(f: LocalFile) { if (f.kind === 'photo') images[f.path] = 'data:image/jpeg;base64,' + await readFile(f) }
async function acceptCamera(result: any) {
  if (!draft.value || !(result?.uri||result?.path)) throw new Error('拍摄结果未返回，草稿已保留，请重新拍照')
  const r = await Filesystem.readFile({ path: result.uri||result.path })
  const f = await putFile(r.data as string, '病损照片.' + (result.metadata?.format || result.format || 'jpg'), 'photo')
  draft.value.files = draft.value.files.filter(f => f.kind !== 'photo').concat(f); invalidate(); await localPreview(f); await queuePersist()
}
async function takePhoto(source: 'camera' | 'gallery') {
  await act(async () => {
    await queuePersist()
    if (!Capacitor.isNativePlatform()) { document.querySelector<HTMLInputElement>('#photo-input')?.click(); return }
    if(draft.value) await persist('active-draft', { key:draftKey(draft.value), patient_id:draft.value.patient_id, pending:true })
    try {
      if (source === 'camera') await acceptCamera(await Camera.takePhoto({ quality: 100, includeMetadata: true, saveToGallery: false }))
      else { const r = await Camera.chooseFromGallery({ includeMetadata: true }); if (r.results[0]) await acceptCamera(r.results[0]) }
    } catch (e: any) { if (/cancel|取消|0003/i.test(e.message || '')) { notify('已取消，草稿保留'); return }; throw new Error('相机或相册无法打开，请检查权限后重试') }
    finally { await persist('active-draft',{pending:false}) }
  })
}
async function chooseReport() {
  if (!Capacitor.isNativePlatform()) { document.querySelector<HTMLInputElement>('#report-input')?.click(); return }
  await act(async () => { await queuePersist(); if(draft.value)await persist('active-draft',{key:draftKey(draft.value),patient_id:draft.value.patient_id,pending:true}); try { const r = await MobileFiles.pickReport(); const data = await Filesystem.readFile({ path: r.path }); const f = await putFile(data.data as string, r.name, 'report'); draft.value?.files.push(f); invalidate(); await queuePersist() } catch (e: any) { if (/cancel|取消/.test(e.message || '')) notify('已取消，草稿保留'); else throw new Error('报告无法读取，请重新选择PDF或图片') } finally {await persist('active-draft',{pending:false})} })
}
async function openReport(r: any) {
  await act(async () => {
    const ext = r.name.split('.').pop()?.toLowerCase() || 'pdf'
    const mime = ext === 'pdf' ? 'application/pdf' : ext === 'png' ? 'image/png' : 'image/jpeg'
    const data = await attachmentData(r.id)
    if (Capacitor.isNativePlatform()) {
      const path = 'reports/' + r.id + '.' + ext
      await Filesystem.writeFile({ path, data, directory: Directory.Data, recursive: true })
      const uri = await Filesystem.getUri({ path, directory: Directory.Data })
      await MobileFiles.openFile({ path: uri.uri, mime })
    } else {
      const bytes = Uint8Array.from(atob(data), c => c.charCodeAt(0)); const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([bytes], {type:mime})); a.download = r.name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href),10000)
    }
  })
}
async function browserFiles(e: Event, kind: 'photo' | 'report') {
  const files = (e.target as HTMLInputElement).files
  if (!files || !draft.value) return
  await act(async () => { for (const file of files) { if (file.size > 20_000_000) throw new Error('文件超过20MB'); const data = await new Promise<string>((resolve, reject) => { const r = new FileReader(); r.onload = () => resolve(String(r.result).split(',')[1]!); r.onerror = () => reject(new Error('文件无法读取')); r.readAsDataURL(file) }); const f = await putFile(data, file.name, kind); if (kind === 'photo') draft.value!.files = draft.value!.files.filter(f => f.kind !== 'photo'); draft.value!.files.push(f); await localPreview(f) }; invalidate() })
  ;(e.target as HTMLInputElement).value = ''
}
const localPhoto = computed(() => draft.value?.files.find(f => f.kind === 'photo'))
const draftPhoto = computed(() => localPhoto.value ? images[localPhoto.value.path] : draft.value?.original_photo ? images[draft.value.original_photo.id] : '')
const taskContext = computed(() => tasks.value.find(t => t.id === draft.value?.task_id))
async function nextStep() {
  if (!draft.value) return
  if (!draft.value.visit_date || !draft.value.site) { error.value = '请填写记录日期和病损部位'; return }
  error.value = ''; draft.value.step = Math.min(2, draft.value.step + 1); window.scrollTo(0, 0)
}
async function saveRecord(submit = false) {
  await act(async () => {
    const d = draft.value!
    for (const f of d.files) if (!f.uploaded) { f.upload_request ||= crypto.randomUUID(); await queuePersist(); const r = await api('/api/assets', 'POST', { name: f.name, kind: f.kind, data: await readFile(f) }, f.upload_request); f.uploaded = r.id; await queuePersist() }
    if (!d.saved) {
      d.save_request ||= crypto.randomUUID(); await queuePersist()
      const body = { patient_id: d.patient_id, scenario: d.scenario, site: d.site, visit_date: d.visit_date, symptoms: d.symptoms, payload: d.payload, task_id: d.task_id, revision: d.revision,
        photo_id: d.files.find(f => f.kind === 'photo')?.uploaded, report_ids: d.files.filter(f => f.kind === 'report').map(f => f.uploaded) }
      d.saved = await api<Visit>(d.id ? `/api/visits/${d.id}` : '/api/visits', d.id ? 'PATCH' : 'POST', body, d.save_request)
      d.local_key ||= draftKey(d)
      d.id = d.saved.id; d.revision = d.saved.revision
      await queuePersist()
    }
    let v = d.saved!
    if (submit) v = await api<Visit>(`/api/visits/${v.id}/submissions`, 'POST', { revision: v.revision }, d.save_request + ':submit')
    delete drafts.value[draftKey(d)]; draft.value = null; await queuePersist(); await refresh(); current.value = v; goto('detail'); notify(submit ? '已提交，等待医生查看' : '草稿已保存')
  })
}
async function confirmReport() {
  if (!current.value) return
  await act(async () => { const v = current.value!; current.value = await api<Visit>(`/api/visits/${v.id}`, 'PATCH', { patient_id: v.patient_id, revision: v.revision, scenario: v.scenario, site: v.site, visit_date: v.visit_date, symptoms: v.symptoms, payload: { ...v.payload, report_confirmed: true } }); await refresh(); notify('检测字段已核对；请重新提交此版本') })
}
async function submitExisting() { await act(async () => { const v = current.value!; current.value = await api(`/api/visits/${v.id}/submissions`, 'POST', { revision: v.revision }); await refresh(); notify('已提交') }) }
async function sendReview() {
  await act(async () => { const v = current.value!; current.value = await api(`/api/visits/${v.id}/reviews`, 'POST', { revision: v.revision, ...review.value }); delete reviewDrafts.value[v.id]; await queuePersist(); await refresh(); notify('反馈已保存'); goto('detail') })
}
async function research() { if (!current.value) return; runs.value = await api(`/api/visits/${current.value.id}/runs`); run.value = null; goto('research') }
async function startRun() {
  await act(async () => { run.value = await api(`/api/visits/${current.value!.id}/runs`, 'POST', { revision: current.value!.revision }); await pollRun(run.value.id) })
}
async function pollRun(id: string) {
  try { run.value = await api(`/api/runs/${id}`); if (['queued', 'running'].includes(run.value.status)) pollTimer = setTimeout(() => pollRun(id), 1000); else runs.value = await api(`/api/visits/${current.value!.id}/runs`) } catch (e: any) { error.value = e.message }
}
function back() {
  if (page.value === 'editor' && draft.value && draft.value.step > 0) { draft.value.step--; return }
  if (page.value === 'research') { goto('doctor'); return }
  goto('home')
}
function doctorEntry() { role.value = 'doctor'; goto('doctor-list') }
onMounted(async () => {
  await restoreConnection(); Object.assign(connectForm, connection)
  const saved = await restore('drafts'); if (saved) { drafts.value = saved.drafts || {}; reviewDrafts.value = saved.reviewDrafts || {} }
  const cache = await restore('cache'); if (cache) { patients.value = cache.patients || []; visits.value = cache.visits || []; tasks.value = cache.tasks || [] }
  patientId.value = (await Preferences.get({ key: 'patient' })).value || cache?.patientId || ''
  if (connection.url) { try { await api('/api/health'); patients.value = await api('/api/patients'); Object.assign(options, await api('/api/options')); await refresh() } catch { online.value = false } }
  if (Capacitor.isNativePlatform()) {
    await NativeApp.addListener('backButton', () => { if (page.value === 'home') NativeApp.minimizeApp(); else back() })
    await NativeApp.addListener('appStateChange', async () => { await queuePersist() })
    await NativeApp.addListener('appRestoredResult', async (r) => {
      if(!r.success || !(r.pluginId==='Camera' || r.pluginId==='MobileFiles' && r.methodName==='pickReport'))return
      await act(async()=>{
        const active=await restore('active-draft'); const d=active?.key ? drafts.value[active.key] : null
        if(!d)throw new Error('已保留拍摄或选择结果，请重新打开对应草稿')
        patientId.value=d.patient_id;draft.value=d;goto('editor')
        if(r.pluginId==='Camera')await acceptCamera((r.data as any)?.results?.[0]||r.data)
        else { const value=r.data as any; const data=await Filesystem.readFile({path:value.path}); const f=await putFile(data.data as string,value.name,'report');draft.value!.files.push(f);invalidate();await queuePersist() }
        await persist('active-draft',{pending:false})
      })
    })
    setTimeout(async()=>{
      const active=await restore('active-draft')
      if(active?.pending && !busy.value && (await NativeApp.getState()).isActive && drafts.value[active.key]) {
        draft.value=drafts.value[active.key];patientId.value=draft.value!.patient_id;goto('editor')
        notify('拍摄或文件选择已中断，草稿已恢复，请重新选择')
        await persist('active-draft',{pending:false})
      }
    },1000)
  }
})
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <button v-if="!['home','records','mine'].includes(page)" class="icon-button" aria-label="返回" @click="back"><Icon name="back" /></button>
      <div v-else class="brand-icon"><Icon name="stethoscope" /></div>
      <span class="brand">{{ doctorMode || page === 'doctor-list' ? '医生工作台' : '口腔随访' }}</span>
      <span class="demo-badge">演示</span>
      <button class="icon-button right" aria-label="查看待办" @click="goto('home')"><Icon name="bell" /><i v-if="nextTask" /></button>
    </header>
    <main>
      <div v-if="error" class="error" role="alert" tabindex="-1">{{ error }}<button aria-label="关闭提示" @click="error=''">×</button></div>

      <template v-if="page === 'home'">
        <div class="welcome"><p>{{ activePatient ? activePatient.alias.replace('模拟：','') : '欢迎回来' }}<span v-if="isDemo" class="tiny-label">模拟档案</span></p><h1>把每一次记录<br>安心准备好。</h1></div>
        <section v-if="nextTask" class="hero-card">
          <div class="hero-head"><span class="pill light"><Icon :name="nextTask.kind === 'material' ? 'camera' : 'calendar'" />下一步</span><span>{{ statusText[nextTask.status] || nextTask.status }}</span></div>
          <h2>{{ nextTask.kind === 'material' ? '补充' + nextTask.requested_items.map(i=>itemText[i]).join('与') : '准备下一次复诊' }}</h2>
          <p>{{ cleanNote(nextTask.note) }}</p>
          <div class="hero-meta">{{ nextTask.site }} · {{ nextTask.due || nextTask.source_date }}</div>
          <button class="primary" :disabled="busy || nextTask.status !== 'pending'" @click="startTask(nextTask)">{{ nextTask.status === 'submitted' ? '等待医生确认' : nextTask.kind === 'material' ? '去补充资料' : '我已完成复诊' }}<Icon name="arrow" /></button>
        </section>
        <section v-else class="hero-card"><span class="pill light"><Icon name="check" />{{ patientId ? '当前没有待办' : '从一份记录开始' }}</span><h2>{{ patientId ? '准备好，再出发' : '记录口腔的变化' }}</h2><p>拍一张照片，整理已有资料，让下一次就诊更从容。</p><button class="primary" @click="newRecord">新增记录<Icon name="plus" /></button></section>
        <section v-if="recentFeedback" class="card feedback-card"><div class="section-title"><h2>最近反馈</h2><span v-if="isDemo" class="tiny-label">模拟</span></div><p>{{ cleanNote(recentFeedback.feedback) }}</p><button class="text-button" @click="openVisit(recentFeedback)">查看这次记录<Icon name="chevron" /></button></section>
        <div class="section-title spaced"><h2>最近记录</h2><button class="text-button" @click="goto('records')">全部<Icon name="chevron" /></button></div>
        <button v-for="v in visits.slice(0,2)" :key="v.id" class="record-row card" @click="openVisit(v)"><div class="record-thumb"><img v-if="v.photo && images[v.photo.id]" :src="images[v.photo.id]" alt="病损照片" /><Icon v-else name="file" /></div><div class="row-main"><strong>{{ v.site }}</strong><span>{{ v.visit_date }} · {{ v.scenario === 'self_check' ? '首次自查' : '随访' }}</span></div><span class="status" :class="v.status">{{ statusText[v.status] }}</span></button>
        <div v-if="!visits.length" class="empty"><Icon name="records" /><p>你的就诊记录会保存在这里</p><button v-if="!online" class="text-button" @click="goto('mine')">连接电脑服务<Icon name="arrow" /></button></div>
      </template>

      <template v-if="page === 'records' || page === 'doctor-list'">
        <div class="page-heading"><p>{{ page === 'doctor-list' ? '资料与反馈' : '每次就诊，各自保存' }}</p><h1>{{ page === 'doctor-list' ? '等待你查看' : '我的记录' }}</h1></div>
        <select class="patient-select" v-model="patientId" aria-label="选择档案"><option v-for="p in patients" :value="p.id">{{ p.alias }}</option></select>
        <button v-for="v in visits" :key="v.id" class="record-row card" @click="openVisit(v,page === 'doctor-list')"><div class="record-thumb"><img v-if="v.photo && images[v.photo.id]" :src="images[v.photo.id]" alt="病损照片" /><Icon v-else name="file" /></div><div class="row-main"><strong>{{ v.site }}</strong><span>{{ v.visit_date }}</span></div><span class="status" :class="v.status">{{ statusText[v.status] }}</span></button>
        <button class="secondary" @click="refresh">刷新记录</button>
      </template>

      <template v-if="page === 'editor' && draft">
        <div class="page-heading"><p>{{ taskContext ? '补到指定的这次记录' : '准备就诊资料' }}</p><h1>{{ ['描述情况','照片与资料','核对提交'][draft.step] }}</h1></div>
        <div class="stepper"><span v-for="(s,i) in ['情况','资料','核对']" :class="{active:i===draft.step,done:i<draft.step}"><b>{{ i+1 }}</b>{{ s }}</span></div>
        <div v-if="taskContext" class="task-context"><Icon name="link" /><div><strong>{{ taskContext.site }} · {{ taskContext.source_date }}</strong><p>{{ cleanNote(taskContext.note) }}</p></div></div>
        <section v-if="draft.step === 0" class="card form-card">
          <label>这次记录用于<select v-model="draft.scenario" :disabled="!!draft.task_id" @change="invalidate"><option value="self_check">首次自查 · 准备就诊</option><option value="followup">已就诊 · 随访</option></select></label>
          <label>病损部位<select v-model="draft.site" :disabled="!!draft.task_id" @change="invalidate"><option v-for="s in options.sites">{{ s }}</option></select></label>
          <label>记录日期<input v-model="draft.visit_date" type="date" @input="invalidate" /></label>
          <label>想告诉医生的情况<textarea v-model="draft.symptoms" rows="4" placeholder="例如：右侧舌缘不适，持续了一周" @input="invalidate" /></label>
          <details><summary>吸烟、饮酒与槟榔史<span>可稍后补充</span></summary><div v-for="prefix in ['smoking','alcohol','betel']" class="history-block"><label>{{ {smoking:'吸烟',alcohol:'饮酒',betel:'嚼槟榔'}[prefix as 'smoking'] }}<select v-model="draft.payload[prefix]" @change="invalidate"><option :value="null">暂不清楚</option><option :value="0">否</option><option :value="1">是</option></select></label><div v-if="draft.payload[prefix] === 1" class="two-fields"><label>年限（年）<input v-model="draft.payload[prefix+'_years']" type="number" min="0" @input="invalidate" /></label><label>{{ prefix==='smoking'?'支/天':prefix==='alcohol'?'ml/天':'颗/天' }}<input v-model="draft.payload[prefix+'_amount']" type="number" min="0" @input="invalidate" /></label></div></div></details>
        </section>
        <template v-if="draft.step === 1">
          <section class="card form-card"><div class="section-title"><h2>病损照片</h2><span class="tiny-label">保留原图</span></div><div v-if="draftPhoto" class="photo-frame"><img :src="draftPhoto" alt="本次病损照片" /></div><div v-else class="photo-empty"><Icon name="camera" /><p>对准指定部位，光线充足、拍清楚</p></div><div class="two-buttons"><button class="secondary" @click="takePhoto('camera')"><Icon name="camera" />拍一张</button><button class="secondary" @click="takePhoto('gallery')">从相册选</button></div><label class="small-label">拍摄日期<input v-model="draft.payload.photo_date" type="date" @input="invalidate" /></label></section>
          <section class="card form-card"><div class="section-title"><h2>已有报告</h2><span class="tiny-label">没有也可继续</span></div><div v-for="r in draft.original_reports" class="attachment"><Icon name="file" /><span>{{ r.name }}</span><small>已保存</small></div><div v-for="(f,i) in draft.files.filter(f=>f.kind==='report')" class="attachment"><Icon name="file" /><span>{{ f.name }}</span><button class="icon-button" aria-label="移除待上传报告" @click="draft.files=draft.files.filter(x=>x.path!==f.path);invalidate()">×</button></div><button class="secondary" @click="chooseReport"><Icon name="plus" />添加报告</button>
            <details><summary>填写检测数值<span>可由医生协助</span></summary><label>TCT意见<input v-model="draft.payload.tct" placeholder="按原报告填写" @input="invalidate" /></label><div class="two-fields"><label>DNA异常细胞数<input v-model="draft.payload.dna_count" type="number" min="0" @input="invalidate" /></label><label>总细胞数<input v-model="draft.payload.dna_total" type="number" min="0" @input="invalidate" /></label></div><label>报告DI阈值<select v-model="draft.payload.dna_threshold" @change="invalidate"><option :value="null">未填写</option><option :value="2.3">DI ≥ 2.3</option><option :value="2.5">DI ≥ 2.5</option></select></label><label>甲基化总数<input v-model="draft.payload.methylation" type="number" min="0" @input="invalidate" /></label><label>报告日期<input v-model="draft.payload.report_date" type="date" @input="invalidate" /></label><label>报告取材部位<select v-model="draft.payload.report_site" @change="invalidate"><option value="">未确认</option><option v-for="s in options.sites">{{ s }}</option></select></label><label>来源说明<textarea v-model="draft.payload.source_note" rows="2" placeholder="对应哪份报告、哪次取材" @input="invalidate" /></label></details>
          </section>
          <input id="photo-input" hidden type="file" accept="image/jpeg,image/png" @change="browserFiles($event,'photo')" /><input id="report-input" hidden type="file" accept="application/pdf,image/jpeg,image/png" multiple @change="browserFiles($event,'report')" />
        </template>
        <section v-if="draft.step === 2" class="card form-card"><div class="summary-row"><span>部位</span><strong>{{ draft.site }}</strong></div><div class="summary-row"><span>日期</span><strong>{{ draft.visit_date }}</strong></div><div class="summary-row"><span>照片</span><strong>{{ localPhoto || draft.original_photo ? '已添加' : '尚未添加' }}</strong></div><div class="summary-row"><span>报告</span><strong>{{ (draft.original_reports?.length || 0)+draft.files.filter(f=>f.kind==='report').length }}份</strong></div><p class="summary-note">{{ draft.symptoms || '暂无补充说明' }}</p><div v-if="taskContext" class="mini-note">本次提交关联：{{ taskContext.site }} · {{ taskContext.source_date }}</div><div class="mini-note">检测字段保存后，可在记录详情核对来源。</div></section>
        <div class="editor-actions"><button v-if="draft.step>0" class="secondary" @click="draft.step--">上一步</button><button v-if="draft.step<2" class="primary" :disabled="busy" @click="nextStep">下一步<Icon name="arrow" /></button><button v-else class="primary" :disabled="busy" @click="saveRecord(true)">{{ busy?'正在保存…':'提交给医生' }}<Icon name="arrow" /></button></div><button class="text-button centered" :disabled="busy" @click="saveRecord(false)">保存草稿，稍后继续</button><p class="autosave">{{ savingDraft?'正在保存到手机…':'输入已自动保留在手机' }}</p>
      </template>

      <template v-if="(page === 'detail' || page === 'doctor') && current">
        <div class="page-heading"><p>{{ current.visit_date }} · {{ current.scenario==='self_check'?'首次自查':'随访记录' }}</p><h1>{{ current.site }}</h1><span class="status" :class="current.status">{{ statusText[current.status] }}</span></div>
        <section class="card form-card"><div v-if="current.photo && images[current.photo.id]" class="photo-frame"><img :src="images[current.photo.id]" alt="本次病损照片" /></div><div v-else class="photo-empty"><Icon name="camera" /><p>本次尚未添加照片</p></div><p v-if="current.symptoms">{{ cleanNote(current.symptoms) }}</p><div v-for="m in current.quality.messages" class="mini-note">{{ m }}</div><button v-for="r in current.reports" class="attachment report-button" @click="openReport(r)"><Icon name="file" /><span>{{ r.name }}</span><Icon name="chevron" /></button><details><summary>查看病史与检测字段<span>{{ current.payload.report_confirmed ? '已核对来源' : '待核对' }}</span></summary><div v-for="h in options.history" class="summary-row"><span>{{ h.title }}</span><strong>{{ current.payload[h.key] ?? '未填写' }}</strong></div><div v-for="k in ['tct','dna_count','dna_total','dna_threshold','methylation','report_date','report_site','source_note']" class="summary-row"><span>{{ ({tct:'TCT',dna_count:'异常细胞数',dna_total:'总细胞数',dna_threshold:'DI阈值',methylation:'甲基化总数',report_date:'报告日期',report_site:'取材部位',source_note:'来源说明'} as Record<string,string>)[k] }}</span><strong>{{ current.payload[k] ?? '未填写' }}</strong></div><button v-if="!current.payload.report_confirmed" class="secondary" :disabled="busy" @click="confirmReport">我已核对以上字段与来源</button></details></section>
        <template v-if="page === 'detail'">
          <section v-if="current.feedback" class="card form-card"><div class="section-title"><h2>医生反馈</h2><span v-if="isDemo" class="tiny-label">模拟</span></div><p>{{ cleanNote(current.feedback) }}</p><p class="muted">{{ current.reviewer }}</p></section>
          <section v-if="current.feedback_history.length" class="card"><details><summary>以往反馈</summary><div v-for="f in current.feedback_history" class="history-feedback"><small>第{{ f.revision }}版 · {{ f.reviewed.slice(0,10) }}</small><p>{{ cleanNote(f.feedback) }}</p></div></details></section>
          <button class="secondary" @click="editVisit(current)">补充或修改资料</button><button v-if="current.status==='draft'" class="primary" :disabled="busy" @click="submitExisting">提交给医生<Icon name="arrow" /></button><button v-if="role==='doctor'" class="text-button centered" @click="openVisit(current,true)">返回医生复核</button>
        </template>
        <template v-else>
          <section class="card form-card"><h2>复核意见</h2><label>复核人代号<input v-model="review.reviewer" placeholder="例如：医生A" /></label><label>给患者的反馈<textarea v-model="review.feedback" rows="4" placeholder="说明下一步需要做什么" /></label><div class="chip-group"><label v-for="i in ['photo','report','history','other']" class="check-chip"><input type="checkbox" :value="i" v-model="review.requested_items" />补{{ itemText[i] }}</label></div><label v-if="review.requested_items.length">具体补充要求<textarea v-model="review.material_note" rows="2" placeholder="例如：补右侧舌缘照片，注明拍摄日期" /></label><details><summary>安排线下复诊</summary><label>复诊日期<input v-model="review.due" type="date" /></label><label>复诊事项<textarea v-model="review.task_note" rows="2" /></label></details><button class="primary" :disabled="busy" @click="sendReview">确认并发送反馈<Icon name="check" /></button><p class="autosave">未发送意见自动保留</p></section><button class="secondary" @click="research">查看研究辅助<Icon name="arrow" /></button>
        </template>
      </template>

      <template v-if="page === 'research' && current">
        <div class="page-heading"><p>{{ current.site }} · 第{{ current.revision }}版</p><h1>研究辅助<span class="tiny-label">研究</span></h1></div>
        <section class="card form-card"><div class="section-title"><h2>本次资料</h2><button class="icon-button" aria-label="模型适用说明" @click="showClinicalHelp=!showClinicalHelp"><Icon name="help" /></button></div><p v-if="showClinicalHelp" class="mini-note">用于SYSU正常/轻度、中度、重度三分类研究。尚未验证患者自拍初筛或长期癌变预测；三类分数不是癌变概率。</p><div class="chip-group"><span class="pill">{{ current.photo ? '已有照片' : '缺少照片' }}</span><span class="pill">{{ current.payload.report_confirmed ? '检测字段已核对' : '检测字段未确认' }}</span></div><button class="primary" :disabled="busy || ['queued','running'].includes(run?.status)" @click="startRun">{{ ['queued','running'].includes(run?.status)?'正在分析…':'运行真实模型' }}<Icon name="arrow" /></button></section>
        <section v-if="run" class="card form-card"><div class="section-title"><h2>运行状态</h2><span class="tiny-label">{{ ({queued:'排队中',running:'运行中',predicted:'已完成',abstain:'未输出',failed:'失败'} as Record<string,string>)[run.status] }}</span></div><p v-if="run.stale" class="mini-note">资料已更新，这是旧版本结果。</p><div v-if="run.result?.probabilities" class="score-list"><div v-for="(p,i) in run.result.probabilities"><div class="section-title"><span>{{ ['正常/轻度','中度','重度'][Number(i)] }}</span><strong>{{ (p*100).toFixed(1) }}%</strong></div><div class="score-track"><i :style="{width: p*100+'%'}" /></div></div></div><p v-if="run.result?.reason">{{ run.result.reason }}</p><div v-if="run.result?.present" class="modality-grid"><span v-for="(yes,k) in run.result.present" :class="{missing:!yes}">{{ ({image:'照片',history:'病史',tct:'TCT',dna:'DNA',methylation:'甲基化'} as Record<string,string>)[k] }} · {{ yes?'已有':'缺失' }}</span></div><details><summary>运行详情</summary><pre>{{ JSON.stringify(run,null,2) }}</pre></details></section>
        <section v-if="runs.length" class="card"><details><summary>历史运行记录（{{ runs.length }}）</summary><div v-for="r in runs" class="history-feedback"><p>第{{ r.revision }}版 · {{ r.result.status }}</p><small>{{ r.created }}</small></div></details></section>
      </template>

      <template v-if="page === 'mine'">
        <div class="page-heading"><p>资料与设置</p><h1>我的</h1></div>
        <section class="card form-card"><div class="profile-row"><div class="avatar"><Icon name="user" /></div><div><strong>{{ activePatient?.alias || '建立匿名档案' }}</strong><p class="muted">{{ online?'电脑服务已连接':'尚未连接电脑' }}</p></div></div><label v-if="patients.length">切换档案<select v-model="patientId"><option v-for="p in patients" :value="p.id">{{ p.alias }}</option></select></label><details><summary>新建档案</summary><label>匿名代号<input v-model="alias" placeholder="例如：小陈" /></label><button class="secondary" :disabled="busy" @click="addPatient">建立档案</button></details></section>
        <section class="card form-card"><div class="section-title"><h2>连接电脑</h2><span class="connection-dot" :class="{connected:online}">{{ online?'已连接':'未连接' }}</span></div><label>服务地址<input v-model="connectForm.url" type="url" placeholder="http://192.168.1.5:8765" autocapitalize="off" /></label><label>连接口令<input v-model="connectForm.token" type="password" placeholder="电脑启动服务时显示的口令" /></label><button class="primary" :disabled="busy" @click="connect">{{ busy?'正在连接…':'连接' }}<Icon name="link" /></button></section>
        <section class="card menu-card"><button @click="doctorEntry"><Icon name="stethoscope" /><span>演示医生视图</span><Icon name="chevron" /></button><button :disabled="busy" @click="loadDemo"><Icon name="records" /><span>载入模拟档案</span><Icon name="chevron" /></button><button @click="showHelp=!showHelp"><Icon name="help" /><span>使用帮助</span><Icon name="chevron" /></button><div v-if="showHelp" class="help-copy"><p>先连接同一Wi-Fi的电脑服务，再准备照片和资料。没有检测报告也可以提交。</p><p>演示角色切换用于流程试用。模拟档案和反馈均为虚构内容。</p><p>检测字段需核对报告来源。患者看到医生确认的反馈；模型输出只出现在医生研究页。</p><p>手机保留草稿，已保存记录存放在电脑。卸载前请备份资料。</p></div></section>
        <button v-if="Object.keys(drafts).length" class="secondary" @click="newRecord">继续未完成的记录</button>
      </template>
    </main>
    <nav v-if="['home','records','mine'].includes(page)" class="bottom-nav" aria-label="主导航"><button :class="{active:page==='home'}" @click="role='patient';goto('home')"><Icon name="home" /><span>首页</span></button><button :class="{active:page==='records'}" @click="goto('records')"><Icon name="records" /><span>记录</span></button><button class="add-tab" @click="newRecord"><b><Icon name="plus" /></b><span>新增</span></button><button :class="{active:page==='mine'}" @click="goto('mine')"><Icon name="user" /><span>我的</span></button></nav>
    <div v-if="toast" class="toast" role="status">{{ toast }}</div>
    <div v-if="busy" class="busy-dot" aria-label="正在处理" />
  </div>
</template>
