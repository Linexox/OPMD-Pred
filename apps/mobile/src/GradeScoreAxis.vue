<script setup lang="ts">
import { computed } from 'vue'

type Reference = { score: number; label: string; count: number }
const props = defineProps<{ score: number; references: Reference[] }>()
const valid = computed(() => Number.isFinite(props.score) && props.score >= 0 && props.score <= 1 &&
  props.references.length === 3 && props.references.every(r => Number.isFinite(r.score) && r.score >= 0 && r.score <= 1))
const percent = (value: number) => `${(value * 100).toFixed(1)}`
</script>

<template>
  <section v-if="valid" class="grade-axis" aria-label="分级参考">
    <div class="axis-heading"><h3>分级参考</h3><span>本次分数 <strong>{{ percent(score) }}</strong></span></div>
    <div class="axis-plot" role="img" :aria-label="`本次分数${percent(score)}，参考位置：${references.map(r => `${r.label}${percent(r.score)}`).join('，')}`">
      <div class="axis-track"></div>
      <div class="axis-current" :style="{ left: `${score * 100}%` }"><span>本次</span><i></i></div>
      <div v-for="(r, index) in references" :key="r.label" class="axis-reference" :class="`grade-${index}`" :style="{ left: `${r.score * 100}%` }"><i></i><span>{{ r.label }}</span></div>
      <div class="axis-endpoints"><span>0</span><span>100</span></div>
    </div>
    <p class="axis-caption">圆点是已知病例各等级的分数中位数，箭头是你本次的位置。</p>
    <details class="axis-details"><summary>参考依据</summary><p>横轴为模型给“中度及以上”的分数。参考来自112份开发资料的交叉验证留出预测；各等级分数有重叠，位置不等于病理诊断。</p><div v-for="r in references" :key="r.label">{{ r.label }} · {{ r.count }}份 · 中位数 {{ percent(r.score) }}</div></details>
  </section>
</template>

<style scoped>
.grade-axis{margin:24px 0}.axis-heading{display:flex;align-items:center;justify-content:space-between;gap:8px}.axis-heading h3{font-size:1rem;margin:0}.axis-heading>span{font-size:.75rem;color:#82778f}.axis-heading strong{font-size:1.35rem;color:#6552af;margin-left:5px}.axis-plot{position:relative;margin:12px 24px 0;height:125px}.axis-track{position:absolute;top:43px;left:0;right:0;height:9px;border-radius:8px;background:linear-gradient(90deg,#9dbfbd,#bab3dc 52%,#d6a6b7)}.axis-current{position:absolute;top:0;transform:translateX(-50%);z-index:2;text-align:center}.axis-current span{display:block;white-space:nowrap;background:#44384e;color:#fff;border-radius:8px;padding:5px 8px;font-size:.7rem}.axis-current i{display:block;width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-top:8px solid #44384e;margin:3px auto 0}.axis-reference{position:absolute;top:44px;transform:translateX(-50%);text-align:center}.axis-reference>i{display:block;width:9px;height:9px;border:2px solid white;border-radius:50%;margin:0 auto 12px;box-shadow:0 0 0 2px #8573ad;background:#8573ad}.axis-reference>span{white-space:nowrap;font-size:.75rem;color:#665870}.grade-0>i{background:#699c97;box-shadow:0 0 0 2px #699c97}.grade-2>i{background:#b86e8b;box-shadow:0 0 0 2px #b86e8b}.axis-endpoints{position:absolute;top:96px;left:0;right:0;display:flex;justify-content:space-between;color:#a297ad;font-size:.7rem}.axis-caption{font-size:.75rem;line-height:1.7;color:#85798e;margin:0}.axis-details{font-size:.75rem;color:#85798e;margin-top:8px}.axis-details p{line-height:1.7;margin:8px 0}.axis-details>div{margin:5px 0}@media(prefers-reduced-motion:no-preference){.axis-current{transition:left .25s ease}}
</style>
