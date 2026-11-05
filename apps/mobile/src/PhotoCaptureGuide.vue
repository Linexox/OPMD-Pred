<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { api } from './api'
import { readFile, type LocalFile } from './storage'
const props = defineProps<{ site: string; photo?: LocalFile; hasPhoto: boolean }>()
defineEmits<{ retake: [] }>()
const status = ref('idle'), messages = ref<string[]>([]), kept = ref(false), helper = ref(false)
let request = 0
const guidance = computed(() => props.site.includes('舌侧缘') ? '让选中的舌侧缘朝向镜头，拍到需要记录的位置及少量周围组织。' : props.site.includes('颊黏膜') ? '让选中一侧的面颊内侧朝向镜头，拍到需要记录的位置及少量周围组织。' : '对准选中的部位，拍到需要记录的位置及少量周围组织。')
const illustratedSite = computed(() => props.site.includes('舌侧缘') ? 'tongue' : props.site.includes('颊黏膜') ? 'cheek' : '')
async function check() {
  const photo = props.photo, id = ++request
  kept.value = false; messages.value = []; status.value = photo ? 'checking' : 'idle'
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
    <p>{{ guidance }} 尽量保持光线均匀，稳住手机后拍摄。</p>
    <details v-if="illustratedSite"><summary>查看拍摄示意</summary>
      <svg v-if="illustratedSite === 'tongue'" class="site-guide" viewBox="0 0 300 140" role="img" aria-label="舌缘拍摄示意：露出选中一侧的舌缘，拍到病损和少量周围组织">
        <rect x="1" y="1" width="298" height="138" rx="20" fill="#f7f4fc" />
        <path d="M43 68 Q150 -10 257 68 Q150 145 43 68Z" fill="#eee1e8" stroke="#af91a2" stroke-width="3" />
        <path d="M99 66 Q150 42 201 66 Q205 119 150 124 Q95 119 99 66Z" fill="#d89ba6" stroke="#b87786" stroke-width="3" />
        <path d="M105 75 Q99 95 113 110 M195 75 Q201 95 187 110" fill="none" stroke="#765fc4" stroke-width="6" stroke-linecap="round" />
      </svg>
      <svg v-else class="site-guide" viewBox="0 0 300 140" role="img" aria-label="颊黏膜拍摄示意：露出选中一侧面颊内侧，拍到病损和少量周围组织">
        <rect x="1" y="1" width="298" height="138" rx="20" fill="#f7f4fc" />
        <path d="M38 70 Q150 -12 262 70 Q150 152 38 70Z" fill="#eee1e8" stroke="#af91a2" stroke-width="3" />
        <path d="M65 69 Q89 44 110 49 L110 90 Q87 96 65 69 M235 69 Q211 44 190 49 L190 90 Q213 96 235 69" fill="#d8c8ed" stroke="#765fc4" stroke-width="4" />
        <ellipse cx="150" cy="72" rx="35" ry="27" fill="#d89ba6" />
      </svg>
      <p>露出选中的一侧，拍到病损和旁边少量组织。</p>
    </details>
    <button type="button" class="text-button" @click="helper = !helper">请他人帮忙</button>
    <p v-if="helper">可以请他人拿稳手机，对准你选中的部位。拍好后一起查看照片。</p>
    <div v-if="hasPhoto" class="mini-note" role="status" aria-live="polite">
      <p v-if="status === 'checking'">正在检查照片的尺寸、清晰度和亮度，可继续填写。</p>
      <p v-else-if="status === 'failed'">暂时无法检查照片，原图与已填资料已保留。可以继续填写，联网后再试。</p>
      <p v-else-if="status === 'ready' && messages.length">{{ messages[0] }}</p>
      <p v-else-if="status === 'ready'">未发现明显的尺寸、模糊或曝光问题。</p>
      <p v-else>照片已保留，可以继续填写。</p>
      <p v-if="kept">已保留这张照片，请继续填写。</p>
      <div v-if="!kept" class="two-buttons"><button type="button" class="secondary" @click="$emit('retake')">重拍</button><button type="button" class="secondary" @click="kept = true">保留并继续</button></div>
      <button v-if="status === 'failed'" type="button" class="text-button" @click="check">重新检查</button>
      <details v-if="status === 'ready'"><summary>查看检查范围</summary><p>这里只检查整张照片的技术表现，尚未经过口腔照片验证，可能误提示。不能判断病损、拍摄部位或病损是否完整，请自行核对照片中的内容。</p><p v-for="message in messages.slice(1)" :key="message">{{ message }}</p></details>
    </div>
  </div>
</template>
<style scoped>
.capture-guide p { margin: 8px 0; line-height: 1.6; }
.capture-guide .text-button { padding: 8px 0; }
.site-guide { display:block; width:100%; height:auto; max-height:170px; }
</style>
