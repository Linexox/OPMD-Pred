<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { ResearchResult, RunComparison } from './api'

const props = defineProps<{ result: ResearchResult; comparison: RunComparison | null; loading: boolean; error: string }>()
const emit = defineEmits<{ compare: [previousId: string]; retry: [] }>()
const targetId = ref('class_2')
const names: Record<string, string> = { image: '照片', history: '病史', tct: 'TCT', dna: 'DNA', methylation: '甲基化' }
const labels = ['正常/轻度', '中度', '重度']
const explanation = computed(() => props.result.explanation)
const ready = computed(() => explanation.value?.status === 'ready' && explanation.value.format === 'modality-shap-v1')
const target = computed(() => explanation.value?.targets?.find(t => t.id === targetId.value))
const contributions = computed(() => [...(target.value?.contributions || [])].sort((a, b) => Number(b.present) - Number(a.present) || Math.abs(b.value) - Math.abs(a.value)))
const extent = computed(() => Math.max(.01, ...contributions.value.map(c => Math.abs(c.value))))
const changed = computed(() => props.comparison?.changed_inputs?.filter(c => c.kind !== 'unchanged') || [])
const contributionChanges = computed(() => props.comparison?.explanation_changes?.find(t => t.target === targetId.value))
const percent = (value: number) => (value * 100).toFixed(1) + '%'
const points = (value: number) => Math.abs(value) < .0005 ? '≈0' : (value > 0 ? '+' : '−') + (Math.abs(value) * 100).toFixed(1)
const changeText: Record<string, string> = { added: '新增', removed: '移除', changed: '更新' }
watch(() => props.result.explanation, value => { targetId.value = value?.default_target || 'class_2' }, { immediate: true })
</script>

<template>
  <section class="card explanation-card">
    <div class="section-title"><h2>本次判断依据</h2><span class="tiny-label">资料贡献</span></div>
    <template v-if="ready && target">
      <div class="target-tabs" role="group" aria-label="选择解释的类别">
        <button v-for="t in explanation!.targets" :key="t.id" :aria-pressed="targetId === t.id" :class="{ selected: targetId === t.id }" @click="targetId = t.id">{{ t.label }}</button>
      </div>
      <div class="explanation-score"><span>{{ target.label }}分数</span><strong>{{ percent(target.value) }}</strong></div>
      <p class="chart-caption">哪些资料推高或拉低这个分数 · 单位：百分点</p>
      <div class="contribution-list">
        <div v-for="item in contributions" :key="item.modality" class="contribution-row" :class="{ absent: !item.present }">
          <span>{{ item.label }}</span>
          <div class="contribution-track" aria-hidden="true"><i class="zero-line" /><i v-if="item.present" class="contribution-bar" :class="item.value >= 0 ? 'positive' : 'negative'" :style="{ width: Math.abs(item.value) / extent * 50 + '%' }" /></div>
          <strong :class="item.value >= 0 ? 'positive-text' : 'negative-text'">{{ item.present ? points(item.value) : '未参与' }}</strong>
        </div>
      </div>
      <details class="explanation-details">
        <summary>资料一起使用的影响</summary>
        <div v-for="pair in target.interactions" :key="pair.first + pair.second" class="compact-contribution"><span>{{ names[pair.first] }} × {{ names[pair.second] }}</span><strong>{{ points(pair.value) }}</strong></div>
        <p v-if="!target.interactions?.length" class="chart-caption">至少两类资料参与时，才能查看共同影响。</p>
        <p class="chart-caption">表示两项共同影响分数的方向和幅度，不代表医学因果，也不等于补上就会判得更准。</p>
      </details>
      <details class="explanation-details">
        <summary>这张图怎么看</summary>
        <p class="chart-caption">以本模型全部输入缺失时的数学参考分数 {{ percent(target.reference_value) }} 为起点，各项贡献相加得到本次分数 {{ percent(target.value) }}。参考分数不作为诊断。</p>
        <p class="chart-caption">未填写的检测不参与本次解释。贡献的正负描述模型分数，不表示病情改善或恶化。</p>
      </details>
    </template>
    <p v-else class="empty-explanation">{{ explanation?.status === 'failed' ? explanation.reason : '这条分析尚无解释。重新分析当前资料后，即可查看。' }}</p>
  </section>

  <section class="card explanation-card">
    <div class="section-title"><h2>补资料前后</h2><span v-if="comparison?.status === 'compared'" class="tiny-label">同一模型</span></div>
    <p v-if="loading" class="empty-explanation" role="status">正在读取两次分析…</p>
    <div v-else-if="error" class="empty-explanation"><p>{{ error }}</p><button class="text-button" @click="emit('retry')">重新读取</button></div>
    <template v-else-if="comparison?.status === 'compared' && comparison.current && comparison.previous">
      <label v-if="comparison.candidates.length > 1" class="comparison-select">与哪次分析对照<select :value="comparison.previous.id" @change="emit('compare', ($event.target as HTMLSelectElement).value)"><option v-for="candidate in comparison.candidates" :key="candidate.id" :value="candidate.id">{{ candidate.relation === 'material_task_source' ? '来源就诊 · ' : '' }}第{{ candidate.revision }}版 · {{ candidate.created.slice(0,16).replace('T',' ') }}</option></select></label>
      <div class="comparison-grades"><div><small>{{ comparison.previous.relation === 'material_task_source' ? '来源就诊' : '较早资料' }} · 第{{ comparison.previous.revision }}版</small><strong>{{ comparison.previous.median_class }}</strong></div><span aria-hidden="true">→</span><div><small>本次资料 · 第{{ comparison.current.revision }}版</small><strong>{{ comparison.current.median_class }}</strong></div></div>
      <div class="changed-inputs"><span v-for="item in changed" :key="item.modality">{{ changeText[item.kind] }}{{ item.label }}</span><span v-if="!changed.length">模型实际接收的资料未改变</span></div>
      <div class="comparison-table" role="table" aria-label="补充前后三类分数">
        <div class="comparison-table-row table-head" role="row"><span role="columnheader">类别</span><span role="columnheader">之前</span><span role="columnheader">本次</span><span role="columnheader">变化</span></div>
        <div v-for="(label, index) in labels" :key="label" class="comparison-table-row" role="row"><span role="cell">{{ label }}</span><span role="cell">{{ percent(comparison.previous.probabilities[index]!) }}</span><strong role="cell">{{ percent(comparison.current.probabilities[index]!) }}</strong><span role="cell">{{ points(comparison.probability_deltas![index]!) }}</span></div>
      </div>
      <p class="chart-caption">变化单位为百分点。这是资料更新后的模型反应，不用于自动判断病情进展。</p>
      <p v-if="changed.length > 1" class="chart-caption">多项资料一起变化，不能把全部分数变化归到其中某一项。</p>
      <details v-if="contributionChanges" class="explanation-details"><summary>{{ contributionChanges.label }}分数的贡献变化</summary><div class="comparison-table-row table-head contribution-comparison"><span>资料</span><span>之前</span><span>本次</span></div><div v-for="item in contributionChanges.contributions" :key="item.modality" class="comparison-table-row contribution-comparison"><span>{{ item.label }}</span><span>{{ points(item.before) }}</span><strong>{{ points(item.after) }}</strong></div><p class="chart-caption">单位：百分点。贡献按各次实际资料组合计算。</p></details>
    </template>
    <p v-else class="empty-explanation">{{ comparison?.reason || '先完成本次分析；更新资料后再分析，即可对照前后变化。' }}</p>
  </section>
</template>
