// 病损部位的说明与示意图分区。
// 拍摄动作取自远程口腔摄影文献里按部位给出的操作建议，改写成患者能读懂的句子。
export type SiteGuide = {
  number: number
  label: string
  view: 'front' | 'under'
  hint: string
  expose: string
  camera: string
}

// 顺序与服务端 SITES 一致，编号也按这个顺序。
export const SITE_ORDER = [
  '上唇', '下唇', '左颊黏膜', '右颊黏膜', '牙龈', '上腭',
  '舌背', '左舌侧缘', '右舌侧缘', '舌腹', '口底', '其他（备注说明）'
]

export const SITE_GUIDES: Record<string, SiteGuide> = {
  '上唇': {
    number: 1, label: '上唇', view: 'front',
    hint: '上嘴唇本身，翻开嘴唇能看到的内侧也算。',
    expose: '用手指轻轻把上唇往上翻，露出唇红和唇内侧的黏膜。',
    camera: '手机放在正前方，与嘴唇同高，镜头正对。'
  },
  '下唇': {
    number: 2, label: '下唇', view: 'front',
    hint: '下嘴唇本身，翻开嘴唇能看到的内侧也算。',
    expose: '用手指把下唇往下翻，露出唇红和唇内侧的黏膜。',
    camera: '手机放在正前方，与嘴唇同高，镜头正对。'
  },
  '左颊黏膜': {
    number: 3, label: '左颊黏膜', view: 'front',
    hint: '你的左边脸颊内侧那层肉（图示按照镜子的方位，你的左边在图的右边）。',
    expose: '用干净的手指把左侧嘴角拉开，让左边脸颊内侧露出来；可以稍微侧一点脸迎着光。',
    camera: '手机稍微偏向左侧，与脸平行，别从正上方拍。'
  },
  '右颊黏膜': {
    number: 4, label: '右颊黏膜', view: 'front',
    hint: '你的右边脸颊内侧那层肉（图示按照镜子的方位，你的右边在图的左边）。',
    expose: '用干净的手指把右侧嘴角拉开，让右边脸颊内侧露出来；可以稍微侧一点脸迎着光。',
    camera: '手机稍微偏向右侧，与脸平行，别从正上方拍。'
  },
  '牙龈': {
    number: 5, label: '牙龈', view: 'front',
    hint: '牙齿根部那一圈肉，上牙下牙都有。',
    expose: '翻起上下嘴唇，让牙齿根部和牙龈交界的地方完整露出来。',
    camera: '手机正对，稍微退后一点，让上下牙龈都进画面。'
  },
  '上腭': {
    number: 6, label: '上腭', view: 'front',
    hint: '口腔顶部，舌头舔上去的那一面，仰头才看得到。',
    expose: '头往后仰、张大嘴，让口腔顶部尽量露出来，舌头放松放低。',
    camera: '手机放到下巴前下方，镜头朝上，注意别用自己的手挡住光。'
  },
  '舌背': {
    number: 7, label: '舌背', view: 'front',
    hint: '舌头朝上的那一面，伸出来看到的就是它。',
    expose: '把舌头伸出来，舌面尽量平展，不要卷起来或用力绷紧。',
    camera: '手机放在嘴的正前方，稍微高一点，能看全整个舌面。'
  },
  '左舌侧缘': {
    number: 8, label: '左舌侧缘', view: 'front',
    hint: '舌头左边那条边。',
    expose: '舌头伸出来并偏向右侧，左边那条边就转出来了。',
    camera: '手机正对舌头的左侧，不要从正上方拍。'
  },
  '右舌侧缘': {
    number: 9, label: '右舌侧缘', view: 'front',
    hint: '舌头右边那条边。',
    expose: '舌头伸出来并偏向左侧，右边那条边就转出来了。',
    camera: '手机正对舌头的右侧，不要从正上方拍。'
  },
  '舌腹': {
    number: 10, label: '舌腹', view: 'under',
    hint: '舌头朝下的那一面，要卷起舌尖才看得到。',
    expose: '舌尖顶住上腭，把舌头下面翻出来，保持几秒钟不要晃动。',
    camera: '手机从下往上拍，位置低于下巴，镜头对着舌头下方。'
  },
  '口底': {
    number: 11, label: '口底', view: 'under',
    hint: '舌头底下那块地方，舌头抬起时才露出来。',
    expose: '舌尖顶住上腭，下巴同时放低，露出舌头底下那一块。',
    camera: '手机从下往上拍，比拍舌腹时再退远一点，把口底拍全。'
  },
  '其他（备注说明）': {
    number: 12, label: '其他（备注说明）', view: 'front',
    hint: '不在上面十一处，或者自己也说不清，选这个并在“想告诉医生的情况”里描述。',
    expose: '拍一张能看清大致位置的照片，病损和旁边的正常组织都要在画面里。',
    camera: '手机正对，退远一点拍全景，能看出是口腔的哪个区域。'
  }
}

export function siteGuide(site: string): SiteGuide | undefined {
  return SITE_GUIDES[site]
}
