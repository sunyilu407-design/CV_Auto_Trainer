import type { VLMClass } from '../store/taskStore'

const SYNONYM_PROMPTS: Array<{ pattern: RegExp; prompts: string[] }> = [
  {
    // Keep <=6 diverse, short prompts — too many similar CLIP embeddings cause
    // softmax dilution in YOLO-World and destroy confidence scores
    pattern: /wheel[_\s-]*chock|chock|三角木|止轮|挡轮|掩木|轮挡|楔块|塞车/i,
    prompts: [
      'wheel chock',
      'tire chock',
      'chock block',
      'rubber wedge',
      'wheel stopper',
    ],
  },
  {
    pattern: /helmet|hardhat|安全帽/i,
    prompts: ['hard hat', 'safety helmet', 'construction helmet'],
  },
  {
    pattern: /forklift|叉车/i,
    prompts: ['forklift', 'forklift truck', 'industrial forklift'],
  },
  {
    pattern: /person|worker|human|人员|工人|行人|操作人员/i,
    prompts: ['person', 'worker', 'human', 'standing person'],
  },
  {
    pattern: /truck|lorry|卡车|货车/i,
    prompts: ['truck', 'lorry', 'cargo truck', 'heavy truck'],
  },
]

// YOLO-World softmax 稀释上限：每类最多展开 8 个 CLIP 词条
const MAX_ALIASES_PER_CLASS = 8

function uniqueNonEmpty(items: Array<string | null | undefined>) {
  const seen = new Set<string>()
  const result: string[] = []
  for (const item of items) {
    const value = item?.trim()
    if (!value) continue
    const key = value.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    result.push(value)
  }
  return result
}

function normalizeClassName(value: string) {
  return value.replace(/_/g, ' ').replace(/\s+/g, ' ').trim()
}

export function buildYoloWorldPrompt(classItem: VLMClass) {
  // Only use class identity (class_name + display_name_zh) for synonym matching;
  // prompt field may be a long VLM description containing incidental words
  const identityHaystack = [
    classItem.class_name,
    classItem.display_name_zh,
  ].filter(Boolean).join(' ')

  const synonymPrompts = SYNONYM_PROMPTS.flatMap((entry) => (entry.pattern.test(identityHaystack) ? entry.prompts : []))
  const baseClassName = normalizeClassName(classItem.class_name)
  // Only include prompt if it's a short noun phrase (<=5 words) and English-only
  const rawPrompt = classItem.prompt || ''
  const isShortEnglish = !/[\u4e00-\u9fff]/.test(rawPrompt) && rawPrompt.split(/\s+/).length <= 5
  const englishPrompt = isShortEnglish ? rawPrompt : ''
  const rawColorHint = classItem.color_hint || ''
  const isShortColor = !/[\u4e00-\u9fff]/.test(rawColorHint) && rawColorHint.split(/\s+/).length <= 3 && !rawColorHint.includes(';')
  const englishColorHint = isShortColor ? rawColorHint : ''

  const promptAliases = uniqueNonEmpty([
    ...synonymPrompts,
    baseClassName,
    englishPrompt,
    englishColorHint ? `${englishColorHint} ${baseClassName}` : '',
  ]).slice(0, MAX_ALIASES_PER_CLASS)

  return promptAliases[0] || baseClassName
}

export function buildDetectionClass(classItem: VLMClass) {
  // Only match category-identity fields, NOT description fields which may contain
  // incidental characters (e.g. "person" in "carried by a person" triggering person class)
  const identityHaystack = [
    classItem.class_name,
    classItem.display_name_zh,
  ].filter(Boolean).join(' ')
  const synonymPrompts = SYNONYM_PROMPTS.flatMap((entry) => (entry.pattern.test(identityHaystack) ? entry.prompts : []))
  const baseClassName = normalizeClassName(classItem.class_name)
  // Only include prompt if it's a short noun phrase (<=5 words) and English-only
  const rawPrompt = classItem.prompt || ''
  const isShortEnglish = !/[\u4e00-\u9fff]/.test(rawPrompt) && rawPrompt.split(/\s+/).length <= 5
  const englishPrompt = isShortEnglish ? rawPrompt : ''
  // Only use color_hint if it's a short color phrase (<=3 words) — long descriptive
  // hints like "Brown and yellow are most common; may also appear in other colors"
  // create garbage CLIP prompts that destroy detection
  const rawColorHint = classItem.color_hint || ''
  const isShortColor = !/[\u4e00-\u9fff]/.test(rawColorHint) && rawColorHint.split(/\s+/).length <= 3 && !rawColorHint.includes(';')
  const englishColorHint = isShortColor ? rawColorHint : ''
  const promptAliases = uniqueNonEmpty([
    ...synonymPrompts,
    baseClassName,
    englishPrompt,
    englishColorHint ? `${englishColorHint} ${baseClassName}` : '',
  ]).slice(0, MAX_ALIASES_PER_CLASS)

  return {
    class_name: classItem.class_name,
    prompt: promptAliases[0] || buildYoloWorldPrompt(classItem),
    prompt_aliases: promptAliases,
    negative_prompt: classItem.negative_prompt || classItem.display_negative_prompt_zh,
    color_hint: classItem.color_hint || classItem.display_color_hint_zh,
    display_name: classItem.display_name_zh || classItem.class_name,
    display_prompt: classItem.display_prompt_zh || classItem.prompt,
  }
}
