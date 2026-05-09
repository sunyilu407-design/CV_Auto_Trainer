/**
 * 类别词 CLIP 友好度启发式打分。
 * 对应 v9.0 优化文档「P0 修复：词汇验证层」。
 *
 * 规则要点：
 *  - 抽象概念 / 状态 / 区域 / 事件名 → 严重扣分（CLIP 无法理解）
 *  - 单个具体可见物体名词 → 加分
 *  - 命中 CLIP 训练分布常见词 → 加分
 *  - 仅中文、无英文 prompt → 扣分（YOLO-World 走 CLIP 文本编码器，需要英文）
 *  - 多词短语（>3 个英文单词）→ 扣分
 *
 * 输出 0-10 分，颜色阈值：>=7 绿 / 4-6 黄 / <4 红。
 */

import type { VLMClass } from '../store/taskStore'

export interface CategoryValidation {
  clipScore: number          // 0-10
  specificityScore: number   // 0-10
  level: 'green' | 'yellow' | 'red'
  warnings: string[]
  suggestions: string[]
}

// CLIP / YOLO-World 训练集中表现稳健的常见名词集合
const CLIP_STRONG_NOUNS = new Set([
  // 人物
  'person', 'worker', 'pedestrian', 'cyclist', 'driver', 'patient', 'child', 'adult',
  // 车辆
  'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'forklift', 'excavator', 'sedan',
  // 安全装备
  'hard hat', 'helmet', 'safety helmet', 'safety vest', 'face mask', 'glove', 'safety shoe',
  // 工业 / 日常
  'bottle', 'box', 'pallet', 'pipe', 'valve', 'screw', 'bearing', 'chair', 'table',
  'cup', 'fork', 'knife', 'laptop', 'phone', 'book',
  // 动物
  'dog', 'cat', 'bird', 'cow', 'horse',
  // 医疗
  'syringe', 'pill', 'tablet', 'bandage', 'stethoscope',
])

const ABSTRACT_TERMS = [
  // 抽象概念
  'danger', 'safety', 'area', 'zone', 'region', 'space', 'place',
  'event', 'incident', 'situation', 'condition', 'status', 'state',
  'violation', 'compliance', 'risk', 'hazard',
  // 中文抽象词
  '危险', '安全', '区域', '范围', '空间', '事件', '状态', '违规', '风险',
]

const STATE_OR_ACTION_TERMS = [
  'running', 'walking', 'working', 'broken', 'damaged', 'idle', 'moving',
  'parked', 'standing', 'sitting',
  '工作中', '运行中', '损坏', '空闲', '运动中', '静止',
]

const ATTRIBUTE_ONLY_TERMS = [
  'red', 'blue', 'green', 'yellow', 'orange', 'large', 'small', 'big', 'tiny',
  '红色', '蓝色', '绿色', '黄色', '橙色', '大', '小',
]

function hasChinese(s: string): boolean {
  return /[\u4e00-\u9fff]/.test(s)
}

function isOnlyAttribute(text: string): boolean {
  const cleaned = text.toLowerCase().trim()
  return ATTRIBUTE_ONLY_TERMS.includes(cleaned)
}

function containsAnyTerm(haystack: string, terms: string[]): string | null {
  const lower = haystack.toLowerCase()
  for (const term of terms) {
    if (lower.includes(term.toLowerCase())) return term
  }
  return null
}

function clamp(value: number, lo = 0, hi = 10): number {
  return Math.max(lo, Math.min(hi, value))
}

export function validateCategory(classItem: VLMClass, promptAliases: string[] = []): CategoryValidation {
  const warnings: string[] = []
  const suggestions: string[] = []
  let clipScore = 7        // 默认中性偏正
  let specificityScore = 7

  const className = (classItem.class_name || '').trim()
  const englishPrompt = (classItem.prompt && !hasChinese(classItem.prompt) ? classItem.prompt : '').trim()
  const haystackEn = [className, englishPrompt, ...promptAliases].filter(Boolean).join(' ')
  const haystackZh = [classItem.display_name_zh, classItem.display_prompt_zh].filter(Boolean).join(' ')
  const allText = `${haystackEn} ${haystackZh}`.trim()

  // 没有任何英文词：CLIP/YOLO-World 几乎无法工作
  if (!haystackEn) {
    clipScore -= 5
    warnings.push('类别完全没有英文 prompt，YOLO-World 的 CLIP 文本编码器无法理解')
    suggestions.push('在 class_name 或 prompt 字段填入对应英文常见名词，如 person / car / helmet')
  }

  // 抽象概念
  const abstractHit = containsAnyTerm(allText, ABSTRACT_TERMS)
  if (abstractHit) {
    clipScore -= 4
    specificityScore -= 4
    warnings.push(`包含抽象词 "${abstractHit}"，CLIP 无法定位「区域 / 事件 / 危险」这类非视觉实体`)
    suggestions.push('换成具体可见的物体，比如 "vehicle in lane" 而不是 "danger zone"')
  }

  // 状态/动作
  const stateHit = containsAnyTerm(allText, STATE_OR_ACTION_TERMS)
  if (stateHit) {
    clipScore -= 3
    specificityScore -= 3
    warnings.push(`包含状态/动作词 "${stateHit}"，CLIP 不擅长检测动作，仅适合检测「物体本身」`)
    suggestions.push('用静态名词描述，如 "person" 而不是 "person running"')
  }

  // 仅属性词（红色 / 大 / 蓝色）
  if (isOnlyAttribute(className) || isOnlyAttribute(englishPrompt)) {
    clipScore -= 3
    specificityScore -= 4
    warnings.push('类别仅是颜色/尺寸属性词，缺少具体物体名词')
    suggestions.push('补全主体名词，如 "red wheel chock"、"yellow safety vest"')
  }

  // 命中 CLIP 强词集
  const lowerEn = haystackEn.toLowerCase()
  let strongHit = false
  for (const noun of CLIP_STRONG_NOUNS) {
    if (lowerEn.includes(noun)) {
      strongHit = true
      break
    }
  }
  if (strongHit) {
    clipScore += 2
    specificityScore += 1
  } else if (haystackEn) {
    clipScore -= 1
    warnings.push('类别词不在 CLIP 常见名词集合中，可能召回率偏低')
    suggestions.push('考虑改用 COCO / Objects365 训练集中的常见词，或提供 5-10 张人工种子框微调专用模型')
  }

  // 多词短语 (>3 英文单词)
  if (englishPrompt) {
    const wordCount = englishPrompt.split(/\s+/).filter(Boolean).length
    if (wordCount > 3) {
      specificityScore -= 2
      warnings.push(`prompt "${englishPrompt}" 超过 3 个单词，长描述会稀释 CLIP 注意力`)
      suggestions.push('精简为单一名词短语，如 "wheel chock" 而不是 "small triangular block placed behind a wheel"')
    }
  }

  // 长度过短
  if (haystackEn && haystackEn.replace(/\s/g, '').length < 3) {
    clipScore -= 2
    warnings.push('英文 prompt 过短，CLIP 文本嵌入信息量不足')
  }

  // 同义词过多其实是一种补偿，给个小加分
  if (promptAliases.length >= 3) {
    clipScore += 1
  }

  clipScore = clamp(clipScore)
  specificityScore = clamp(specificityScore)
  const overall = (clipScore + specificityScore) / 2

  let level: CategoryValidation['level'] = 'green'
  if (overall < 4) level = 'red'
  else if (overall < 7) level = 'yellow'

  return { clipScore, specificityScore, level, warnings, suggestions }
}
