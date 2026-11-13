<script setup lang="ts">
import { computed } from 'vue'

type Result = {
  probabilities?: number[] | null
  median_class?: string
  class_labels?: string[]
}

const props = defineProps<{ result: Result }>()
const score = computed(() => {
  const value = props.result.probabilities?.[1]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
})
const percent = computed(() => score.value === null ? '' : `${(score.value * 100).toFixed(1)}%`)
const direction = computed(() => score.value !== null && score.value >= 0.5 ? '中度及以上' : '正常/轻度')
const nearBoundary = computed(() => score.value !== null && Math.abs(score.value - 0.5) < 0.1)
const ariaLabel = computed(() => score.value === null ? '二分类结果暂不可用' : `模型更偏向${direction.value}，模型分数${percent.value}`)
</script>

<template>
  <section v-if="score !== null" class="binary-result" aria-label="二分类分析结果">
    <div class="binary-result-heading">
      <div>
        <p class="binary-result-kicker">模型提示</p>
        <h3>{{ direction }}</h3>
      </div>
      <div class="binary-result-score"><span>中度及以上分数</span><strong>{{ percent }}</strong></div>
    </div>
    <div class="binary-direction" role="img" :aria-label="ariaLabel">
      <div class="binary-direction-track">
        <i class="binary-direction-fill" :style="{ width: `${score * 100}%` }" />
        <b class="binary-direction-marker" :style="{ left: `${score * 100}%` }" />
      </div>
      <div class="binary-direction-labels"><span>正常/轻度</span><span>中度及以上</span></div>
    </div>
    <p v-if="nearBoundary" class="binary-result-note">结果靠近中间位置</p>
  </section>
  <p v-else class="mini-note">暂时无法显示模型分数。</p>
</template>

<style scoped>
.binary-result{margin:18px 0 4px;padding:18px 16px;background:#f8f6fc;border-radius:16px}
.binary-result-heading{display:flex;justify-content:space-between;align-items:flex-start;gap:16px}
.binary-result-kicker{font-size:.75rem;color:#887d97;margin:0 0 5px}
.binary-result h3{font-size:1.35rem;line-height:1.35;margin:0;color:#302c39}
.binary-result-score{text-align:right;white-space:nowrap}
.binary-result-score span{display:block;font-size:.7rem;color:#8c8198}
.binary-result-score strong{display:block;font-size:1.45rem;color:#6652b3;margin-top:4px;font-variant-numeric:tabular-nums}
.binary-direction{margin:22px 2px 0}
.binary-direction-track{position:relative;height:12px;background:linear-gradient(90deg,#dfeeea 0%,#eeeaf6 50%,#eadde8 100%);border-radius:8px}
.binary-direction-fill{display:block;height:100%;background:#8772cf;border-radius:8px}
.binary-direction-marker{position:absolute;top:-7px;width:3px;height:26px;background:#312c38;border-radius:3px;transform:translateX(-50%)}
.binary-direction-labels{display:flex;justify-content:space-between;gap:12px;margin-top:12px;color:#777086;font-size:.7rem}
.binary-result-note{font-size:.75rem;color:#83788e;line-height:1.65;margin:15px 0 0}
</style>
