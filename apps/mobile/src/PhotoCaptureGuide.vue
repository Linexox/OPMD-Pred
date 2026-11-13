<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { api } from './api'
import { readFile, type LocalFile } from './storage'
import { siteGuide } from './site-guide'
const props = defineProps<{ site: string; photo?: LocalFile; hasPhoto: boolean }>()
const status = ref('idle'), messages = ref<string[]>([]), showGuide = ref(false)
let request = 0

const guide = computed(() => siteGuide(props.site))

// 取自远程口腔摄影文献里对患者可执行的建议
const TIPS = [
  '请他人帮忙拿手机最稳；自己拍就对着镜子，让手机和嘴同高。',
  '用后置摄像头，保持 1 倍焦距，不要用手指放大。',
  '光线要均匀，别背光；可以用台灯补光，避开反光和阴影。',
  '别只拍病损，把旁边一小块正常组织也拍进来。'
]

async function check() {
  const photo = props.photo, id = ++request
  messages.value = []; status.value = photo ? 'checking' : 'idle'
  if (!photo) return
  try {
    const result = await api<{ messages: string[] }>('/api/photo-quality', 'POST', { data: await readFile(photo) })
    if (!Array.isArray(result.messages) || !result.messages.every(m => typeof m === 'string')) throw new Error('检查返回异常')
    if (id !== request) return
    messages.value = result.messages; status.value = 'ready'
  } catch { if (id === request) status.value = 'failed' }
}
watch(() => props.photo?.path, check, { immediate: true })
</script>
<template>
  <div class="capture-guide">
    <button type="button" class="text-button cg-toggle" @click="showGuide = !showGuide">{{ showGuide ? '收起拍摄指引' : '不知道怎么拍？看拍摄指引' }}</button>

    <div v-if="showGuide" class="cg-panel">
      <p v-if="guide" class="cg-site">{{ guide.label }}：{{ guide.hint }}</p>
      <div v-if="guide" class="cg-do">
        <p><b>怎么摆姿势</b>：{{ guide.expose }}</p>
        <p><b>手机怎么放</b>：{{ guide.camera }}</p>
      </div>

      <ul class="cg-tips">
        <li v-for="tip in TIPS" :key="tip">{{ tip }}</li>
      </ul>
    </div>

    <div v-if="hasPhoto" class="mini-note" role="status" aria-live="polite">
      <p v-if="status === 'checking'">正在检查照片的尺寸、清晰度和亮度，可继续填写。</p>
      <p v-else-if="status === 'failed'">暂时无法检查照片，原图与已填资料已保留。可以继续填写，联网后再试。</p>
      <p v-else-if="status === 'ready' && messages.length">{{ messages[0] }}</p>
      <p v-else-if="status === 'ready'">未发现明显的尺寸、模糊或曝光问题。</p>
      <p v-else>照片已添加，可以继续填写。</p>
      <p class="cg-selfcheck">对照看一眼：病损看得清吗？周围组织带上了吗？有没有反光或阴影挡住？</p>
      <button v-if="status === 'failed'" type="button" class="text-button" @click="check">重新检查</button>
    </div>
  </div>
</template>
<style scoped>
.capture-guide p { margin: 8px 0; line-height: 1.6; }
.capture-guide .text-button { padding: 8px 0; }
.cg-toggle { text-align: left; }
.cg-panel { display: grid; gap: 4px; margin-top: 4px; }
.cg-site { font-size: 0.875rem; color: #6b6480; }
.cg-do { font-size: 0.875rem; color: #4b4180; background: #f5f2fd; border-radius: 12px; padding: 10px 12px; }
.cg-do p { margin: 4px 0; }
.cg-do b { color: #6b5bb5; }
.cg-tips { margin: 10px 0 0; padding-left: 20px; }
.cg-tips li { font-size: 0.8125rem; color: #6b6480; line-height: 1.7; margin: 4px 0; }
.cg-selfcheck { font-size: 0.8125rem; color: #6b5bb5; }
</style>
