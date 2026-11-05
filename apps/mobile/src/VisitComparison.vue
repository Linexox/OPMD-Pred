<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { Visit } from './api'

const props = defineProps<{ current: Visit; visits: Visit[]; images: Record<string, string> }>()
const selectedId = ref('')
const candidates = computed(() => props.visits.filter(v => v.patient_id === props.current.patient_id && v.id !== props.current.id))
const selected = computed(() => candidates.value.find(v => v.id === selectedId.value))
watch(() => props.current.id, () => { selectedId.value = '' })
watch(candidates, rows => { if (!rows.some(v => v.id === selectedId.value)) selectedId.value = '' })

function value(v: Visit, key: string): string {
  const item = v.payload?.[key]
  return item === null || item === undefined || item === '' ? '未提供' : String(item)
}
function threshold(v: Visit): string {
  const result = value(v, 'dna_threshold')
  return result === '未提供' ? result : `DI ≥ ${result}`
}
</script>

<template>
  <section class="card visit-compare">
    <div class="section-title"><h2>比较两次记录</h2></div>
    <p v-if="!candidates.length" class="compare-hint">这个档案还没有其他记录。</p>
    <template v-else>
      <label class="compare-picker">选择要比较的记录
        <select v-model="selectedId">
          <option value="">请选择日期和部位</option>
          <option v-for="(item, index) in candidates" :key="item.id" :value="item.id">{{ item.visit_date }} · {{ item.site }} · 第{{ index + 1 }}条</option>
        </select>
      </label>
      <template v-if="selected">
        <p class="compare-hint">请核对两次记录是否为同一处病损。</p>
        <div class="visit-columns">
          <article v-for="item in [selected, current]" :key="item.id" class="visit-column">
            <div class="visit-date">{{ item.visit_date }}</div>
            <div class="visit-site">{{ item.site }}</div>
            <div class="visit-image">
              <img v-if="item.photo && images[item.photo.id]" :src="images[item.photo.id]" :alt="item.site + '照片'" />
              <span v-else>{{ item.photo ? '照片暂时无法读取' : '未提供照片' }}</span>
            </div>
          </article>
        </div>
        <div class="compare-row"><span>检测项目</span><strong>{{ selected.visit_date }}</strong><strong>{{ current.visit_date }}</strong></div>
        <div class="compare-row"><span>TCT</span><strong>{{ value(selected, 'tct') }}</strong><strong>{{ value(current, 'tct') }}</strong></div>
        <div class="compare-row"><span>DNA异常细胞数</span><strong>{{ value(selected, 'dna_count') }}</strong><strong>{{ value(current, 'dna_count') }}</strong></div>
        <div class="compare-row"><span>总细胞数</span><strong>{{ value(selected, 'dna_total') }}</strong><strong>{{ value(current, 'dna_total') }}</strong></div>
        <div class="compare-row"><span>DNA阈值</span><strong>{{ threshold(selected) }}</strong><strong>{{ threshold(current) }}</strong></div>
        <div class="compare-row"><span>甲基化总数</span><strong>{{ value(selected, 'methylation') }}</strong><strong>{{ value(current, 'methylation') }}</strong></div>
        <div class="compare-row"><span>报告日期</span><strong>{{ value(selected, 'report_date') }}</strong><strong>{{ value(current, 'report_date') }}</strong></div>
        <div class="compare-row"><span>取材部位</span><strong>{{ value(selected, 'report_site') }}</strong><strong>{{ value(current, 'report_site') }}</strong></div>
        <div class="compare-row"><span>医生反馈</span><strong>{{ selected.feedback || '暂无' }}</strong><strong>{{ current.feedback || '暂无' }}</strong></div>
      </template>
    </template>
  </section>
</template>

<style scoped>
.compare-picker{display:block;color:#686473;font-size:.875rem;margin-top:16px}.compare-hint{color:#8b8298;font-size:.8rem;line-height:1.6;margin:14px 0}.visit-columns{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:16px}.visit-column{min-width:0}.visit-date{font-size:.8rem;color:#8b8298}.visit-site{font-size:.9rem;font-weight:600;margin:6px 0;overflow-wrap:anywhere}.visit-image{height:145px;border-radius:14px;background:#f7f5fa;display:flex;align-items:center;justify-content:center;color:#9990a6;font-size:.75rem;text-align:center;overflow:hidden}.visit-image img{width:100%;height:100%;object-fit:contain}.compare-row{display:grid;grid-template-columns:minmax(65px,1fr) minmax(0,1fr) minmax(0,1fr);gap:8px;padding:13px 0;border-bottom:1px solid #efedf5;align-items:start;font-size:.75rem;line-height:1.5}.compare-row span{color:#8b8298}.compare-row strong{text-align:right;overflow-wrap:anywhere;font-weight:500}.compare-row:last-child{border-bottom:0}
</style>
