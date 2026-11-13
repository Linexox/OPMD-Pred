<script setup lang="ts">
import { ref } from 'vue'
import Icon from './Icon.vue'
import oralChangesIllustration from './assets/oral-changes-illustration.jpeg'
import oralExamIllustration from './assets/oral-exam-illustration.jpeg'

defineProps<{ context: 'home' | 'photo' }>()

const oralCancerGuide = 'https://www.nidcr.nih.gov/health-info/oral-cancer'
const exampleZoomed = ref(false)
</script>

<template>
  <details class="patient-education" :class="context === 'home' ? 'card education-home' : 'education-photo'">
    <summary>
      <Icon name="help" class="education-icon" />
      <span class="education-heading">
        <strong>{{ context === 'home' ? '口腔健康科普' : '拍照时留意哪些变化？' }}</strong>
        <small>{{ context === 'home' ? '认识口腔黏膜异常，了解为什么要检查' : '白斑、红斑、溃疡和肿块' }}</small>
      </span>
      <Icon name="chevron" class="education-chevron" />
    </summary>

    <div v-if="context === 'home'" class="education-content">
      <dl class="education-topics">
        <div>
          <dt>什么是口腔潜在恶性疾患（OPMD）？</dt>
          <dd>这是一组与较高的口腔癌风险有关的黏膜疾病，比如口腔白斑病、口腔红斑病。每个人的风险不同，医生会根据检查结果安排后续观察或复诊。</dd>
        </div>
        <div>
          <dt>什么是口腔鳞癌（OSCC）？</dt>
          <dd>它是最常见的一类口腔癌，起源于口腔黏膜表层，可发生在舌头、牙龈、脸颊内侧或舌头下方。早期可能不痛。</dd>
        </div>
        <div>
          <dt>为什么要尽早检查？</dt>
          <dd>外观看起来相似的变化，原因可能不同。口腔科医生会看一看、摸一摸，必要时取一小块组织做病理检查，明确病变性质，决定是否需要治疗或定期复查。</dd>
        </div>
      </dl>
      <figure class="education-exam">
        <img :src="oralExamIllustration" alt="" width="668" height="479" loading="lazy" decoding="async" />
        <figcaption>医生会用灯光查看口腔里不容易自己看清的地方。</figcaption>
      </figure>
      <p class="education-note">戒烟、戒槟榔、减少饮酒有助于降低口腔癌风险。已经确诊黏膜疾病的患者，请按医生约定的时间复诊。</p>
      <p class="education-sources">
        <a href="https://pubmed.ncbi.nlm.nih.gov/33128420/" target="_blank" rel="noopener noreferrer">口腔潜在恶性疾患专家共识</a>
        <a href="https://www.cancer.gov/types/head-and-neck/patient/adult/lip-mouth-treatment-pdq" target="_blank" rel="noopener noreferrer">口腔癌患者指南</a>
        <a href="https://publications.iarc.who.int/617" target="_blank" rel="noopener noreferrer">口腔癌预防资料</a>
      </p>
    </div>

    <div v-else class="education-content">
      <dl class="education-topics">
        <div>
          <dt>白色、红色或红白相间的斑片</dt>
          <dd>留意颜色、范围有没有变化。轻轻张口观察即可，别用力刮擦。</dd>
        </div>
        <div>
          <dt>溃疡（口腔里破了一小块）</dt>
          <dd>记下哪天发现、是不是一直在同一处、有没有缩小或愈合。</dd>
        </div>
        <div>
          <dt>凸起、肿块或局部变厚</dt>
          <dd>留意有没有变大、变硬或出血，即使不痛也可以记下来告诉医生。</dd>
        </div>
      </dl>
      <details class="education-example">
        <summary>查看口腔变化示意图</summary>
        <figure>
          <div class="education-example-image" :class="{ zoomed: exampleZoomed }" :tabindex="exampleZoomed ? 0 : undefined" role="region" aria-label="口腔变化示意图">
            <img :src="oralChangesIllustration" alt="八格示意图展示口腔不同部位的斑片、破损或凸起外观" width="1000" height="428" loading="lazy" decoding="async" />
          </div>
          <div class="education-example-actions">
            <button type="button" class="text-button" @click="exampleZoomed = !exampleZoomed">{{ exampleZoomed ? '恢复全图' : '放大示意图' }}</button>
            <span v-if="exampleZoomed">左右滑动查看</span>
          </div>
          <figcaption>看图了解斑片、破损和凸起的样子。记录时写下自己实际看到的变化即可。</figcaption>
        </figure>
      </details>
      <p>拍下变化的部位和旁边的正常组织。就诊时可以把照片、发现时间和后续变化一起告诉医生。</p>
      <p class="education-note"><strong>什么时候去看医生？</strong> 变化超过两周仍未好转，或出现明显增大、反复出血等情况，请尽快到口腔科检查。照片有助于记录变化，诊断需要医生检查。</p>
      <p class="education-sources">
        <a :href="oralCancerGuide" target="_blank" rel="noopener noreferrer">口腔异常与就诊提示来源</a>
      </p>
    </div>
  </details>
</template>

<style scoped>
.education-home {
  padding: 4px 18px;
}
.education-photo {
  margin-top: 14px;
  padding: 2px 12px;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: #f8f6fc;
}
.patient-education > summary {
  justify-content: flex-start;
  gap: 10px;
  padding: 10px 0;
  color: #4b3f62;
}
.patient-education > summary::after,
.patient-education > summary::-webkit-details-marker {
  display: none;
}
.education-heading {
  flex: 1;
  min-width: 0;
  margin: 0;
  color: inherit;
  font-size: 0.9375rem;
}
.education-heading small {
  display: block;
  margin-top: 4px;
  color: #625970;
  font-size: 0.8125rem;
  line-height: 1.45;
}
.education-icon,
.education-chevron {
  flex: none;
  width: 18px;
  height: 18px;
  color: #6856a2;
}
.education-chevron {
  width: 14px;
  height: 14px;
  transform: rotate(90deg);
}
.patient-education[open] .education-chevron {
  transform: rotate(-90deg);
}
.education-content {
  padding: 12px 0;
  border-top: 1px solid var(--line);
  color: #50485e;
  font-size: 0.875rem;
  line-height: 1.65;
}
.education-topics {
  display: grid;
  gap: 12px;
  margin: 0;
}
.education-topics dt {
  color: #3e3450;
  font-weight: 650;
}
.education-topics dd {
  margin: 3px 0 0;
}
.education-exam {
  display: grid;
  grid-template-columns: minmax(0, 144px) minmax(0, 1fr);
  align-items: center;
  gap: 12px;
  margin: 12px 0 0;
}
.education-exam img {
  display: block;
  width: 100%;
  height: auto;
  border-radius: 10px;
}
.education-exam figcaption {
  color: #625970;
  font-size: 0.8125rem;
}
.education-example {
  margin-top: 12px;
  padding: 0 10px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #fff;
}
.education-example summary {
  color: #5c4b91;
  font-weight: 600;
}
.education-example figure {
  margin: 0 0 10px;
}
.education-example-image {
  overflow-x: auto;
  overscroll-behavior-x: contain;
  border-radius: 8px;
}
.education-example-image img {
  display: block;
  width: 100%;
  height: auto;
}
.education-example-image.zoomed img {
  width: max(100%, 640px);
  max-width: none;
}
.education-example-actions {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0 10px;
}
.education-example-actions .text-button {
  color: #5c4b91;
}
.education-example figcaption,
.education-example-actions span {
  color: #625970;
  font-size: 0.8125rem;
}
.education-content p {
  margin: 12px 0 0;
}
.education-note {
  padding: 10px 12px;
  border-radius: 10px;
  background: #f0ebfa;
  color: #473865;
}
.education-sources {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 12px;
  font-size: 0.8125rem;
}
.education-sources a {
  color: #5c4b91;
  text-underline-offset: 2px;
}
</style>
