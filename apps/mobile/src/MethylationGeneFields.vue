<script setup lang="ts">
import { computed, ref } from 'vue'

const genes = ['CDKN2A', 'TIMP3', 'ZNF582', 'TMEFF2', 'TFPI2', 'SHOX2', 'PAX1', 'MGMT', 'SOX1', 'DAPK1']
const choices = [{ value: 1, label: '阳性' }, { value: 0, label: '阴性' }] as const
const model = defineModel<Record<string, number | null>>()
const results = computed(() => model.value ?? {})
const filledCount = computed(() => genes.filter(gene => hasResult(gene)).length)
const batchMode = ref(false)
const selected = ref<string[]>([])

function hasResult(gene: string) {
  return results.value[gene] === 0 || results.value[gene] === 1
}

function setResults(targets: string[], result: 0 | 1 | null) {
  const next = { ...results.value }
  for (const gene of targets) {
    if (result === null) delete next[gene]
    else next[gene] = result
  }
  model.value = next
}

function toggleBatch() {
  batchMode.value = !batchMode.value
  selected.value = []
}

function applyBatch(result: 0 | 1) {
  setResults(selected.value, result)
  selected.value = []
}
</script>

<template>
  <details class="nested-details gene-fields">
    <summary>按基因填写甲基化<span class="gene-count">{{ filledCount ? `已录入 ${filledCount} 项` : '选填' }}</span></summary>
    <p class="gene-hint">只填报告里写明结果的项目，其余留空。</p>
    <div class="gene-toolbar" :class="{ 'gene-toolbar-batch': batchMode }">
      <div class="gene-toolbar-heading">
        <button type="button" class="text-button" @click="toggleBatch">{{ batchMode ? '退出批量填写' : '批量填写' }}</button>
        <span v-if="batchMode" aria-live="polite">已勾选 {{ selected.length }} 项</span>
      </div>
      <div v-if="batchMode" class="gene-choices">
        <button v-for="choice in choices" :key="choice.value" type="button" :disabled="!selected.length" @click="applyBatch(choice.value)">所选标{{ choice.label }}</button>
      </div>
    </div>

    <div v-for="gene in genes" :key="gene" class="gene-row">
      <label v-if="batchMode" class="gene-check">
        <input v-model="selected" type="checkbox" :value="gene" />
        <span class="gene-name">{{ gene }}</span>
        <span class="gene-status">{{ results[gene] === 1 ? '阳性' : results[gene] === 0 ? '阴性' : '未填写' }}</span>
      </label>
      <template v-else>
        <span class="gene-name">{{ gene }}</span>
        <div class="gene-choices" role="group" :aria-label="`${gene} 检测结果`">
          <button v-for="choice in choices" :key="choice.value" type="button" :aria-pressed="results[gene] === choice.value" @click="setResults([gene], choice.value)">{{ choice.label }}</button>
        </div>
        <button v-if="hasResult(gene)" type="button" class="text-button gene-clear" :aria-label="`清除 ${gene} 的结果`" @click="setResults([gene], null)">清除</button>
        <span v-else class="gene-status">未填写</span>
      </template>
    </div>
  </details>
</template>

<style scoped>
.gene-count,
.gene-hint,
.gene-status,
.gene-toolbar-heading span {
  color: #625970;
  font-size: 0.8125rem;
}
.gene-count {
  flex: none;
}
.gene-hint {
  margin: 0;
  line-height: 1.6;
}
.gene-toolbar-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.gene-toolbar-batch {
  position: sticky;
  top: var(--safe-top);
  z-index: 1;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--line);
  background: #fff;
}
.gene-toolbar .text-button,
.gene-clear {
  color: #5c4b91;
}
.gene-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto 2.5rem;
  align-items: center;
  gap: 8px;
  min-height: 52px;
  padding: 4px 0;
  border-top: 1px solid var(--line);
}
.gene-name {
  overflow-wrap: anywhere;
  font-size: 0.875rem;
  font-weight: 600;
}
.gene-choices {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}
.gene-choices button {
  min-width: 44px;
  min-height: 44px;
  padding: 4px 7px;
  border: 1px solid #e7e0f1;
  border-radius: 9px;
  background: #f8f6fc;
  color: #50485e;
  font-size: 0.8125rem;
  white-space: nowrap;
}
.gene-choices button[aria-pressed='true'] {
  border-color: #7465ce;
  background: #ece6fa;
  color: #453578;
  font-weight: 700;
}
.gene-row .gene-check {
  display: flex;
  grid-column: 1 / -1;
  align-items: center;
  gap: 10px;
  min-height: 44px;
  margin: 0;
  cursor: pointer;
}
.gene-check .gene-status {
  margin-left: auto;
}
.gene-clear {
  justify-content: flex-end;
  min-height: 44px;
  font-size: 0.8125rem;
}
.gene-row > .gene-status {
  text-align: right;
  white-space: nowrap;
}
</style>
