<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { SITE_GUIDES, SITE_ORDER } from './site-guide'
import frontImage from './assets/mouth-front.png'
import underImage from './assets/mouth-under.png'

type View = 'front' | 'under'
const props = defineProps<{ site: string }>()
const emit = defineEmits<{ pick: [string] }>()

const view = ref<View>('front')
const current = computed(() => SITE_GUIDES[props.site])
const isOn = (name: string) => props.site === name

watch(() => props.site, (s) => { const g = SITE_GUIDES[s]; if (g) view.value = g.view }, { immediate: true })

function pick(name: string) {
  const g = SITE_GUIDES[name]
  if (g) view.value = g.view
  emit('pick', name)
}

// 图片是按 4:3 生成的，编号坐标用百分比换算，换图只要重新对一次
const HOTSPOTS: { name: string; view: View; x: number; y: number }[] = [
  { name: '上唇', view: 'front', x: 49.6, y: 6.0 },
  { name: '下唇', view: 'front', x: 49.6, y: 88.4 },
  { name: '左颊黏膜', view: 'front', x: 78.0, y: 46.0 },
  { name: '右颊黏膜', view: 'front', x: 21.5, y: 45.9 },
  { name: '牙龈', view: 'front', x: 49.6, y: 14.7 },
  { name: '上腭', view: 'front', x: 49.6, y: 30.4 },
  { name: '舌背', view: 'front', x: 49.6, y: 57.1 },
  { name: '左舌侧缘', view: 'front', x: 69.4, y: 56.6 },
  { name: '右舌侧缘', view: 'front', x: 30.6, y: 56.4 },
  { name: '舌腹', view: 'under', x: 49.6, y: 37.3 },
  { name: '口底', view: 'under', x: 35.9, y: 63.1 }
]
const spots = computed(() => HOTSPOTS.filter((h) => h.view === view.value))

const items = SITE_ORDER.map((name) => ({ name, ...SITE_GUIDES[name] }))
</script>

<template>
  <div class="mouth-map">
    <div class="mm-tabs">
      <button type="button" :class="{ on: view === 'front' }" @click="view = 'front'">正面张口</button>
      <button type="button" :class="{ on: view === 'under' }" @click="view = 'under'">舌顶上腭</button>
    </div>

    <div class="mm-frame">
      <img :src="view === 'front' ? frontImage : underImage"
        :alt="view === 'front' ? '正面张口的口腔部位编号示意图' : '舌尖顶住上腭时的口腔部位编号示意图'" />
      <button v-for="h in spots" :key="h.name" type="button" class="mm-hot" :class="{ on: isOn(h.name) }"
        :style="{ left: h.x + '%', top: h.y + '%' }" :aria-label="h.name" @click="pick(h.name)"></button>
    </div>

    <ul class="mm-legend">
      <li v-for="it in items" :key="it.name">
        <button type="button" :class="{ on: isOn(it.name) }" @click="pick(it.name)">
          <b>{{ it.number }}</b><span>{{ it.label }}</span>
        </button>
      </li>
    </ul>
    <p class="mm-note">图示按面对镜子时的方位，你的左边在图右边。图上的编号和下面的编号一一对应，点图上的圆圈或点编号都能选。</p>
    <p v-if="current" class="mm-pick">已选：{{ current.label }}。{{ current.hint }}</p>
  </div>
</template>

<style scoped>
.mouth-map { display: grid; gap: 10px; margin-top: 10px; }
.mm-tabs { display: flex; gap: 8px; }
.mm-tabs button { flex: 1; min-height: 44px; border: 1px solid #e2ddf2; border-radius: 14px; background: #f5f3fb; color: #6b6480; font-size: 0.875rem; }
.mm-tabs button.on { background: #efeafd; border-color: #b7a6ea; color: #4b4180; font-weight: 600; }
.mm-frame { position: relative; border-radius: 18px; overflow: hidden; background: #fff; line-height: 0; }
.mm-frame img { width: 100%; height: auto; display: block; }
.mm-hot { position: absolute; width: 10.5%; aspect-ratio: 1; transform: translate(-50%, -50%); border-radius: 50%; border: 2px solid transparent; background: transparent; padding: 0; cursor: pointer; }
.mm-hot.on { border-color: #6b5bb5; background: rgba(179, 160, 234, 0.28); box-shadow: 0 0 0 3px rgba(107, 91, 181, 0.22); }
.mm-hot:focus-visible { outline: 2px solid #7465ce; outline-offset: 2px; }
.mm-legend { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.mm-legend button { width: 100%; min-height: 44px; display: flex; align-items: center; gap: 8px; padding: 6px 10px; border: 1px solid #ece7f8; border-radius: 12px; background: #faf9fe; color: #57516b; font-size: 0.8125rem; text-align: left; }
.mm-legend button.on { border-color: #b7a6ea; background: #efeafd; color: #40386f; font-weight: 600; }
.mm-legend b { display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; border-radius: 50%; background: #8f7fd6; color: #fff; font-size: 11px; flex-shrink: 0; }
.mm-note { font-size: 0.75rem; color: #8d87a0; line-height: 1.6; margin: 0; }
.mm-pick { font-size: 0.8125rem; color: #57516b; background: #f5f2fd; border-radius: 12px; padding: 10px 12px; line-height: 1.6; margin: 0; }
</style>
