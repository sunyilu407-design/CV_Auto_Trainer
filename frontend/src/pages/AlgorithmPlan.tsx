import { useEffect, useMemo, useRef, useState } from 'react'
import { algorithmApi, type RuntimeCapability, type TrainingRecommendation } from '../api/backend'
import { previewYoloWorldDetection, type DetectionPreviewResult } from '../api/worker'
import { useTaskStore, DEVICE_PROFILES } from '../store/taskStore'
import { useSettingsStore } from '../store/settingsStore'
import { buildDetectionClass } from '../utils/detectionPrompts'
import { validateCategory, type CategoryValidation } from '../utils/categoryValidator'

const SCENARIO_LABELS: Record<string, string> = {
  occupancy_monitoring: '占位监测',
  parking_violation: '违规停车',
  intrusion_monitoring: '区域闯入',
  dwell_time_monitoring: '超时滞留',
  object_tracking: '目标跟踪',
  object_counting: '目标计数',
  safety_compliance: '安全合规',
  quality_inspection: '质量检测',
  feature_matching: '特征匹配',
  classification: '分类识别',
  custom_event_monitoring: '自定义事件',
}

const ROLE_LABELS: Record<string, string> = {
  primary_detector: '主检测器',
  secondary_detector: '辅助检测器',
  classifier: '分类器',
  feature_matcher: '特征匹配器',
  tracker: '目标跟踪器',
  rule_engine: '规则引擎',
  ocr: '文字识别',
}

const DIFFICULTY_LABELS: Record<string, string> = {
  simple: '简单',
  moderate: '中等',
  complex: '复杂',
  very_complex: '非常复杂',
}

const CAPABILITY_KIND_LABELS: Record<string, string> = {
  detection: '检测能力',
  classification: '分类能力',
  tracking: '跟踪能力',
  rule: '规则能力',
}

function renderTagList(items: string[], fallback: string) {
  const values = items.length > 0 ? items : [fallback]
  return (
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
      {values.map((item) => (
        <span key={item} className="badge badge-blue">
          {item}
        </span>
      ))}
    </div>
  )
}

export default function AlgorithmPlan() {
  const {
    taskId,
    userDescription,
    vlmResult,
    vlmStatus,
    algorithmPlan,
    setAlgorithmPlan,
    setStage,
    applyRecommendedTrainConfig,
    deviceProfileId,
    datasetImages,
    algorithmRegionBoxes,
    labelingImageDir,
    negotiatedConfig,
  } = useTaskStore()
  const { settings } = useSettingsStore()

  const [loading, setLoading] = useState(false)
  const loadingRef = useRef(false)
  const [confirming, setConfirming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [revising, setRevising] = useState(false)
  const [showRevisionModal, setShowRevisionModal] = useState(false)
  const [revisionInput, setRevisionInput] = useState('')
  const [rollingBack, setRollingBack] = useState(false)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewError, setPreviewError] = useState<string | null>(null)
  const [previewResult, setPreviewResult] = useState<DetectionPreviewResult | null>(null)

  const deviceProfile = DEVICE_PROFILES[deviceProfileId]
  const effectiveGpuType = deviceProfile.gpuType || settings.defaultGpuType
  const effectivePlatform = deviceProfile.platform || (typeof navigator !== 'undefined' ? navigator.platform : '')

  const runtimeCapability = useMemo<RuntimeCapability>(() => {
    const platform = typeof navigator === 'undefined' ? '' : navigator.platform.toLowerCase()
    const userAgent = typeof navigator === 'undefined' ? '' : navigator.userAgent.toLowerCase()
    let preferredDevice: RuntimeCapability['preferred_device'] = 'cpu'

    if (platform.includes('mac') || userAgent.includes('mac os')) {
      preferredDevice = 'mps'
    } else if (/rtx|nvidia|cuda|a100|h100|l40/i.test(settings.defaultGpuType)) {
      preferredDevice = 'cuda'
    }

    const availableExportFormats =
      preferredDevice === 'mps'
        ? (['onnx', 'coreml'] as RuntimeCapability['available_export_formats'])
        : preferredDevice === 'cuda'
          ? (['onnx', 'engine', 'openvino'] as RuntimeCapability['available_export_formats'])
          : (['onnx', 'openvino'] as RuntimeCapability['available_export_formats'])

    return {
      local_training_available: true,
      preferred_device: preferredDevice,
      available_export_formats: availableExportFormats,
      supports_cloud_training: true,
    }
  }, [settings.defaultGpuType])

  const applyRecommendation = (recommendation: TrainingRecommendation) => {
    applyRecommendedTrainConfig({
      model: recommendation.recommended_config.model,
      epochs: recommendation.recommended_config.epochs,
      imgsz: recommendation.recommended_config.imgsz,
      lr0: recommendation.recommended_config.lr0,
      patience: recommendation.recommended_config.patience,
      conf: recommendation.recommended_config.conf,
      iou: recommendation.recommended_config.iou,
      exportFormats: recommendation.recommended_config.export_formats,
      trainMode: recommendation.recommended_config.train_mode,
      gpuType: recommendation.recommended_config.gpu_type,
    })
  }

  // 优先使用协商确认后的 classes（negotiatedConfig），其次用原始 VLM parse 结果
  const effectiveClasses = useMemo(() => {
    if (negotiatedConfig?.classes && negotiatedConfig.classes.length > 0) {
      return negotiatedConfig.classes
    }
    return vlmResult?.classes ?? []
  }, [negotiatedConfig, vlmResult])

  const detectionClasses = useMemo(() => {
    return effectiveClasses.map(buildDetectionClass)
  }, [effectiveClasses])

  // P0-C 词汇验证：每类一份 CLIP 友好度评估，用于 UI 红黄绿色徽章
  const classValidations = useMemo<Record<string, CategoryValidation>>(() => {
    const result: Record<string, CategoryValidation> = {}
    effectiveClasses.forEach((src, idx) => {
      const detection = detectionClasses[idx]
      result[src.class_name] = validateCategory(src, detection?.prompt_aliases ?? [])
    })
    return result
  }, [effectiveClasses, detectionClasses])

  const validationCounts = useMemo(() => {
    let red = 0, yellow = 0, green = 0
    for (const v of Object.values(classValidations)) {
      if (v.level === 'red') red++
      else if (v.level === 'yellow') yellow++
      else green++
    }
    return { red, yellow, green, total: red + yellow + green }
  }, [classValidations])
  const previewImageDir = taskId ? labelingImageDir || `../backend/uploads/${taskId}/images` : ''

  useEffect(() => {
    if (!taskId || loadingRef.current || algorithmPlan) return

    let cancelled = false
    loadingRef.current = true
    const load = async () => {
      setLoading(true)
      setError(null)
      try {
        const existing = await algorithmApi.getPlan(taskId).catch(() => null)
        if (cancelled) return
        if (existing) {
          setAlgorithmPlan(existing)
          return
        }

        const generated = await algorithmApi.generatePlan({
          task_id: taskId,
          user_description: userDescription,
          vlm_result: effectiveClasses.length > 0 ? { classes: effectiveClasses } : null,
          runtime_capability: runtimeCapability,
          gpu_type: effectiveGpuType,
          platform: effectivePlatform,
          device_description: deviceProfile.label,
          image_count: 0,
          use_vlm_planner: true,
          algorithm_hints: (negotiatedConfig?.algorithm_hints as Record<string, unknown>) || null,
        })
        if (!cancelled) {
          setAlgorithmPlan(generated)
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : '能力草图生成失败')
        }
      } finally {
        loadingRef.current = false
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    load()
    return () => {
      cancelled = true
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [algorithmPlan, runtimeCapability, setAlgorithmPlan, taskId, userDescription, effectiveClasses])

  if (!taskId) {
    return (
      <div className="card-section text-center" style={{ padding: '64px 24px' }}>
        <p style={{ color: 'var(--gray-500)' }}>无可用任务上下文，请先输入业务需求</p>
        <button className="btn btn-secondary" onClick={() => setStage('upload')} style={{ marginTop: 16 }}>
          返回上传
        </button>
      </div>
    )
  }

  const draft = algorithmPlan?.algorithm_plan
  const negotiation = draft?.negotiation_summary
  const scenarioLabel = draft ? SCENARIO_LABELS[draft.scenario_type] ?? draft.scenario_type : ''
  const confidence = draft ? Math.round((draft.confidence ?? 0) * 100) : null
  const pipeline = algorithmPlan?.pipeline_config
  const trainingRecommendation = pipeline?.training_recommendation
  const isTextOnlyDraft = vlmStatus === 'failed' || !vlmResult
  const modelPipeline = draft?.model_pipeline
  const trainingStrategy = draft?.training_strategy
  const difficultyLabel = draft?.difficulty_level ? DIFFICULTY_LABELS[draft.difficulty_level] ?? draft.difficulty_level : null
  const summaryZh = draft?.summary_zh
  const handleConfirm = async () => {
    if (!taskId) return
    setConfirming(true)
    setError(null)
    try {
      const confirmed = await algorithmApi.confirmPlan(taskId, {
        runtime_capability: runtimeCapability,
      })
      setAlgorithmPlan(confirmed)
      if (confirmed.pipeline_config?.training_recommendation) {
        applyRecommendation(confirmed.pipeline_config.training_recommendation)
      }
      setStage('environment')
    } catch (e) {
      setError(e instanceof Error ? e.message : '确认能力草图失败')
    } finally {
      setConfirming(false)
    }
  }

  const handleRevise = async () => {
    if (!taskId || !revisionInput.trim()) return
    setRevising(true)
    setError(null)
    try {
      const revised = await algorithmApi.revisePlan(taskId, {
        user_feedback: revisionInput.trim(),
        runtime_capability: runtimeCapability,
        gpu_type: effectiveGpuType,
        platform: effectivePlatform,
        device_description: deviceProfile.label,
      })
      setAlgorithmPlan(revised)
      setRevisionInput('')
      setShowRevisionModal(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : '方案修订失败')
    } finally {
      setRevising(false)
    }
  }

  const handleRollback = async (version: number) => {
    if (!taskId) return
    setRollingBack(true)
    setError(null)
    try {
      const restored = await algorithmApi.rollbackPlan(taskId, version)
      setAlgorithmPlan(restored)
    } catch (e) {
      setError(e instanceof Error ? e.message : '回滚失败')
    } finally {
      setRollingBack(false)
    }
  }

  const handlePreviewDetection = async () => {
    if (!taskId) return
    if (detectionClasses.length === 0) {
      setPreviewError('当前方案没有可用于 YOLO-World 的检测对象，请先返回需求协商页补充具体可见物体。')
      return
    }
    setPreviewLoading(true)
    setPreviewError(null)
    try {
      const result = await previewYoloWorldDetection({
        image_dir: previewImageDir,
        classes: detectionClasses.map((item) => ({
          class_name: item.class_name,
          prompt: item.prompt,
          prompt_aliases: item.prompt_aliases,
          negative_prompt: item.negative_prompt,
          color_hint: item.color_hint,
        })),
        max_images: Math.min(Math.max(datasetImages.length, 1), 2),
        conf_threshold: 0.12,
        diagnostic_conf_threshold: 0.03,
        iou_threshold: 0.45,
        imgsz: 1280,
      })
      setPreviewResult(result)
    } catch (e) {
      setPreviewError(e instanceof Error ? e.message : 'YOLO-World 预览失败')
    } finally {
      setPreviewLoading(false)
    }
  }

  const revisionSnapshots: Array<{ version: number; summary_zh: string; timestamp: number }> =
    draft?.revision_snapshots ?? []

  return (
    <div>
      <div className="page-header" style={{ marginBottom: 32 }}>
        <div className="flex items-center gap-3 mb-4">
          <div className="badge badge-blue" style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Stage 2
          </div>
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: 8,
              background: 'var(--develop-blue)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 6h18" />
              <path d="M7 12h10" />
              <path d="M10 18h4" />
            </svg>
          </div>
        </div>
        <h1 className="page-title">确认算法方案</h1>
        <p className="page-subtitle">先确认系统会检测什么、在哪里判断、什么情况触发事件；不确定时先用 1-2 张样图试打标。</p>
      </div>

      {loading && (
        <div className="card-section" style={{ marginBottom: 24 }}>
          <div className="flex items-center gap-3">
            <div className="spinner" />
            <div>
              <p style={{ fontSize: 14, fontWeight: 600, color: 'var(--gray-900)' }}>正在生成能力草图</p>
              <p style={{ fontSize: 12, color: 'var(--gray-500)', marginTop: 4 }}>系统正在整理检测对象、监测区域和事件触发条件。</p>
            </div>
          </div>
        </div>
      )}

      {isTextOnlyDraft && (
        <div className="card-section" style={{ marginBottom: 16, background: '#fff9db', border: '1px solid #facc15', boxShadow: 'none' }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#854d0e', marginBottom: 6 }}>当前草图主要基于文字需求生成</div>
          <div style={{ fontSize: 12, color: '#713f12', lineHeight: 1.6 }}>
            你后续仍可补充样板图或参考框，用于细化监测对象的视觉定义。
          </div>
        </div>
      )}

      {error && (
        <div style={{ marginBottom: 16, padding: '12px 14px', background: '#fff5f5', borderRadius: 8, boxShadow: 'var(--shadow-border)', color: 'var(--ship-red)', fontSize: 13 }}>
          {error}
        </div>
      )}

      {draft && negotiation && (
        <>
          <div className="card-section" style={{ marginBottom: 24 }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 16, marginBottom: 16 }}>
              <div>
                <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
                  <div className="badge badge-dark">{scenarioLabel}</div>
                  {difficultyLabel && <div className="badge badge-blue">{difficultyLabel}</div>}
                </div>
                <h2 className="text-heading-sm" style={{ marginBottom: 6 }}>{summaryZh || draft.summary}</h2>
                <p className="text-body-sm" style={{ color: 'var(--gray-500)' }}>{userDescription}</p>
              </div>
              <div style={{ minWidth: 120, textAlign: 'right' }}>
                <div style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 4 }}>规划置信度</div>
                <div style={{ fontSize: 28, fontWeight: 600, letterSpacing: '-1.2px', color: 'var(--gray-900)' }}>
                  {confidence}%
                </div>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 12 }}>
              <div style={{ padding: '14px 16px', borderRadius: 8, background: 'var(--gray-50)', boxShadow: 'var(--shadow-ring)' }}>
                <div style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 8 }}>监测对象</div>
                {renderTagList(negotiation.objects, '关键监测对象')}
              </div>
              <div style={{ padding: '14px 16px', borderRadius: 8, background: 'var(--gray-50)', boxShadow: 'var(--shadow-ring)' }}>
                <div style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 8 }}>监测区域</div>
                {renderTagList(negotiation.regions, '主监测区域')}
              </div>
              <div style={{ padding: '14px 16px', borderRadius: 8, background: 'var(--gray-50)', boxShadow: 'var(--shadow-ring)' }}>
                <div style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 8 }}>时长约束</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--gray-900)' }}>
                  {negotiation.duration_seconds ? `${negotiation.duration_seconds} 秒` : '未明确'}
                </div>
              </div>
              <div style={{ padding: '14px 16px', borderRadius: 8, background: 'var(--gray-50)', boxShadow: 'var(--shadow-ring)' }}>
                <div style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 8 }}>触发结果</div>
                {renderTagList(negotiation.events, '生成业务事件')}
              </div>
            </div>
          </div>

          <div className="card-section" style={{ marginBottom: 24, border: '1px solid rgba(10,114,239,0.18)', background: 'rgba(10,114,239,0.03)' }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 16, marginBottom: 16 }}>
              <div>
                <h3 className="text-heading-sm" style={{ marginBottom: 6 }}>请重点确认：这套方案实际会这样做</h3>
                <p style={{ fontSize: 12, color: 'var(--gray-500)', lineHeight: 1.7, margin: 0 }}>
                  下面不是内部能力编号，而是后续自动打标和事件判断真正会使用的内容。
                </p>
              </div>
              <button
                className="btn btn-primary"
                onClick={handlePreviewDetection}
                disabled={previewLoading || detectionClasses.length === 0}
                style={{ padding: '9px 16px', fontSize: 13, flexShrink: 0 }}
              >
                {previewLoading ? '正在试打标...' : '用 1-2 张图试打标'}
              </button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.1fr) minmax(0, 0.9fr)', gap: 12, marginBottom: 12 }}>
              <div style={{ padding: 16, borderRadius: 10, background: '#fff', boxShadow: 'var(--shadow-ring)' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginBottom: 10 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--develop-blue)' }}>YOLO-World 会检测这些对象</div>
                  {validationCounts.total > 0 && (
                    <div style={{ display: 'flex', gap: 6, fontSize: 11 }}>
                      {validationCounts.green > 0 && (
                        <span style={{ padding: '2px 8px', borderRadius: 999, background: 'rgba(16,185,129,0.12)', color: 'var(--success-green)' }}>✓ {validationCounts.green}</span>
                      )}
                      {validationCounts.yellow > 0 && (
                        <span style={{ padding: '2px 8px', borderRadius: 999, background: 'rgba(245,158,11,0.14)', color: '#92400e' }}>⚠ {validationCounts.yellow}</span>
                      )}
                      {validationCounts.red > 0 && (
                        <span style={{ padding: '2px 8px', borderRadius: 999, background: 'rgba(239,68,68,0.14)', color: 'var(--ship-red)' }}>✕ {validationCounts.red}</span>
                      )}
                    </div>
                  )}
                </div>
                {validationCounts.red > 0 && (
                  <div style={{ marginBottom: 10, padding: '8px 10px', borderRadius: 8, background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.2)', fontSize: 12, color: 'var(--ship-red)' }}>
                    有 {validationCounts.red} 个类别 CLIP 友好度过低，YOLO-World 大概率检不到。建议返回需求确认页修改这些词，或在打标失败后切换到种子框/专用模型微调。
                  </div>
                )}
                {detectionClasses.length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                    {detectionClasses.map((item) => {
                      const validation = classValidations[item.class_name]
                      const levelColor = !validation ? 'var(--gray-400)'
                        : validation.level === 'green' ? 'var(--success-green)'
                        : validation.level === 'yellow' ? '#92400e'
                        : 'var(--ship-red)'
                      const levelBg = !validation ? 'var(--gray-50)'
                        : validation.level === 'green' ? 'rgba(16,185,129,0.06)'
                        : validation.level === 'yellow' ? 'rgba(245,158,11,0.08)'
                        : 'rgba(239,68,68,0.06)'
                      const levelBorder = !validation ? 'transparent'
                        : validation.level === 'green' ? 'rgba(16,185,129,0.18)'
                        : validation.level === 'yellow' ? 'rgba(245,158,11,0.25)'
                        : 'rgba(239,68,68,0.25)'
                      return (
                      <div key={item.class_name} style={{ padding: '10px 12px', borderRadius: 8, background: levelBg, border: `1px solid ${levelBorder}` }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginBottom: 6 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
                            <strong style={{ fontSize: 13, color: 'var(--gray-900)' }}>{item.display_name}</strong>
                            {validation && (
                              <span
                                title={`CLIP 友好度 ${validation.clipScore}/10  ·  具体度 ${validation.specificityScore}/10`}
                                style={{ fontSize: 10, fontWeight: 700, padding: '2px 7px', borderRadius: 999, color: '#fff', background: levelColor }}
                              >
                                {validation.level === 'green' ? `CLIP ${validation.clipScore}` : validation.level === 'yellow' ? `CLIP ${validation.clipScore}` : `CLIP ${validation.clipScore}`}
                              </span>
                            )}
                          </div>
                          <span className="text-mono" style={{ fontSize: 11, color: 'var(--gray-400)' }}>{item.class_name}</span>
                        </div>
                        <div style={{ fontSize: 12, color: 'var(--gray-600)', lineHeight: 1.7 }}>
                          <strong>中文理解：</strong>{item.display_prompt}
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--gray-400)', lineHeight: 1.6, marginTop: 4 }}>
                          给 YOLO-World 的英文词：{(item.prompt_aliases ?? [item.prompt]).join(' / ')}
                        </div>
                        {item.negative_prompt && (
                          <div style={{ fontSize: 11, color: 'var(--gray-400)', lineHeight: 1.6, marginTop: 4 }}>
                            排除：{item.negative_prompt}
                          </div>
                        )}
                        {validation && validation.warnings.length > 0 && (
                          <ul style={{ margin: '8px 0 0', paddingLeft: 18, fontSize: 11, color: levelColor, lineHeight: 1.7 }}>
                            {validation.warnings.map((w) => (
                              <li key={w}>{w}</li>
                            ))}
                          </ul>
                        )}
                        {validation && validation.suggestions.length > 0 && (
                          <ul style={{ margin: '4px 0 0', paddingLeft: 18, fontSize: 11, color: 'var(--gray-500)', lineHeight: 1.7 }}>
                            {validation.suggestions.map((s) => (
                              <li key={s}>建议：{s}</li>
                            ))}
                          </ul>
                        )}
                      </div>
                      )
                    })}
                  </div>
                ) : (
                  <div style={{ fontSize: 12, color: 'var(--ship-red)', lineHeight: 1.7 }}>
                    当前没有具体检测对象。请返回协商页，把“异常/违规/占位”改成图片里能看见的物体，例如车轮、三角木、人员、安全帽。
                  </div>
                )}
              </div>

              <div style={{ padding: 16, borderRadius: 10, background: '#fff', boxShadow: 'var(--shadow-ring)' }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--develop-blue)', marginBottom: 10 }}>监测区域怎么理解</div>
                <div style={{ fontSize: 12, color: 'var(--gray-600)', lineHeight: 1.8 }}>
                  {algorithmRegionBoxes.length > 0
                    ? `你已经在样图上框选了 ${algorithmRegionBoxes.length} 个参考区域，后续会优先按这些区域理解“主监测区域”。`
                    : '当前没有人工框选区域。系统会默认在整张图或 VLM 推断的主区域里判断。'}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 12 }}>
                  {draft.regions.map((region) => (
                    <div key={region.region_id} style={{ padding: '9px 10px', borderRadius: 8, background: 'var(--gray-50)', fontSize: 12 }}>
                      <strong>{region.name}</strong>
                      <span style={{ color: 'var(--gray-400)' }}> · {region.source === 'user_reference_box' ? '来自你框选的参考区域' : region.source}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div style={{ padding: 16, borderRadius: 10, background: '#fff', boxShadow: 'var(--shadow-ring)', marginBottom: 12 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--develop-blue)', marginBottom: 10 }}>触发规则请这样验收</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {draft.events.map((event) => {
                  const region = draft.regions.find((item) => item.region_id === event.trigger.region_id)
                  const temporal = draft.temporal_constraints.find((item) => item.constraint_id === event.trigger.temporal_constraint_id)
                  const target = detectionClasses.find((item) => item.class_name === event.trigger.target_class)
                  return (
                    <div key={event.event_code} style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--gray-50)' }}>
                      <strong style={{ fontSize: 13, color: 'var(--gray-900)' }}>{event.name}</strong>
                      <div style={{ fontSize: 12, color: 'var(--gray-600)', lineHeight: 1.7, marginTop: 4 }}>
                        当画面中出现 <strong>{target?.display_name || event.trigger.target_class}</strong>
                        ，位于 <strong>{region?.name || event.trigger.region_id}</strong>
                        {temporal ? `，并持续 ${temporal.duration_seconds} 秒` : ''}
                        时，触发事件：<strong>{event.name}</strong>。
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {previewError && (
              <div style={{ padding: '10px 12px', borderRadius: 8, background: 'rgba(255,91,79,0.08)', color: 'var(--ship-red)', fontSize: 12, marginBottom: 12 }}>
                {previewError}
              </div>
            )}

            {previewResult && (
              <div style={{ padding: 16, borderRadius: 10, background: '#fff', boxShadow: 'var(--shadow-ring)' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginBottom: 12 }}>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--gray-900)' }}>小样本试打标结果</div>
                    <div style={{ fontSize: 12, color: (previewResult.accepted_box_count ?? previewResult.raw_box_count) > 0 ? 'var(--gray-500)' : 'var(--ship-red)', marginTop: 4 }}>
                      {previewResult.message} · 达到阈值 {(previewResult.accepted_box_count ?? previewResult.raw_box_count)} 个 / 低阈值候选 {(previewResult.candidate_box_count ?? previewResult.raw_box_count)} 个
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--gray-400)', marginTop: 4 }}>
                      蓝色框：达到当前阈值的候选；橙色框：低分候选。框的位置或语义不对时，不建议直接进入训练。
                      {previewResult.imgsz ? ` 本次推理尺寸：${previewResult.imgsz}` : ''}
                    </div>
                  </div>
                  <button className="btn btn-secondary" onClick={() => setPreviewResult(null)} style={{ padding: '6px 12px', fontSize: 12 }}>
                    收起
                  </button>
                </div>
                {previewResult.prompts_used && previewResult.prompts_used.length > 0 && (
                  <div style={{ fontSize: 11, color: 'var(--gray-500)', lineHeight: 1.6, marginBottom: 12 }}>
                    本次英文检测词：{previewResult.prompts_used.join(' / ')}
                  </div>
                )}
                {previewResult.suggestions && previewResult.suggestions.length > 0 && (
                  <div style={{ padding: '10px 12px', borderRadius: 8, background: '#fff9db', color: '#854d0e', fontSize: 12, lineHeight: 1.7, marginBottom: 12 }}>
                    {previewResult.suggestions.map((item) => (
                      <div key={item}>{item}</div>
                    ))}
                  </div>
                )}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12 }}>
                  {previewResult.results.map((result) => (
                    <div key={result.image_name} style={{ border: '1px solid var(--gray-100)', borderRadius: 10, overflow: 'hidden', background: '#fff' }}>
                      <img
                        src={`data:image/jpeg;base64,${result.image_base64}`}
                        alt={result.image_name}
                        style={{ width: '100%', height: 210, objectFit: 'cover', display: 'block' }}
                      />
                      <div style={{ padding: '10px 12px' }}>
                        <div style={{ fontSize: 11, color: 'var(--gray-500)', marginBottom: 8 }}>{result.image_name}</div>
                        {result.detections.length > 0 ? (
                          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                            {result.detections.map((det, idx) => (
                              <span key={idx} className={det.accepted ? 'badge badge-green' : 'badge badge-blue'}>
                                {det.accepted ? '可用' : '低分'} {det.class_name} {(det.confidence * 100).toFixed(0)}%
                              </span>
                            ))}
                          </div>
                        ) : (
                          <div style={{ fontSize: 12, color: 'var(--ship-red)' }}>这张图没有检出目标，请调整检测对象或提示词后再试。</div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="card-section" style={{ marginBottom: 24 }}>
            <div style={{ marginBottom: 14 }}>
              <h3 className="text-heading-sm" style={{ marginBottom: 6 }}>能力模块明细</h3>
              <p style={{ fontSize: 12, color: 'var(--gray-500)', lineHeight: 1.7, margin: 0 }}>
                这里保留给开发和交付人员看：哪些能力需要训练，哪些只是规则组合。
              </p>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {draft.capabilities.map((capability) => (
                <div key={capability.capability_id} style={{ padding: '16px 18px', borderRadius: 8, background: 'var(--gray-50)', boxShadow: 'var(--shadow-ring)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginBottom: 10 }}>
                    <strong style={{ fontSize: 14, color: 'var(--gray-900)' }}>{capability.label}</strong>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      <span className="badge badge-blue">{CAPABILITY_KIND_LABELS[capability.kind] ?? capability.kind}</span>
                      <span className={capability.trainable ? 'badge badge-pink' : 'badge badge-dark'}>
                        {capability.trainable ? '可训练' : '规则组合'}
                      </span>
                    </div>
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--gray-500)' }}>
                    能力标识：<span className="text-mono">{capability.capability_id}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="card-section" style={{ marginBottom: 24 }}>
            <h3 className="text-heading-sm" style={{ marginBottom: 14 }}>事件判断明细</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {draft.events.map((event) => (
                <div key={event.event_code} style={{ padding: '16px 18px', borderRadius: 8, background: 'var(--gray-50)', boxShadow: 'var(--shadow-ring)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginBottom: 10 }}>
                    <strong style={{ fontSize: 14, color: 'var(--gray-900)' }}>{event.name}</strong>
                    <span className="text-mono" style={{ color: 'var(--gray-500)' }}>{event.event_code}</span>
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--gray-500)', lineHeight: 1.7 }}>
                    目标：{event.trigger.target_class} · 区域：{event.trigger.region_id} · 时序：{event.trigger.temporal_constraint_id ?? '无'}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {modelPipeline && modelPipeline.length > 0 && (
            <div className="card-section" style={{ marginBottom: 24 }}>
              <div style={{ marginBottom: 14 }}>
                <h3 className="text-heading-sm" style={{ marginBottom: 6 }}>模型方案</h3>
                <p style={{ fontSize: 12, color: 'var(--gray-500)', lineHeight: 1.7, margin: 0 }}>
                  系统根据您的需求和设备情况自动选择了最合适的模型组合。
                  {trainingStrategy && ` 共需训练 ${trainingStrategy.total_models_to_train} 个模型，预计总时长约 ${trainingStrategy.estimated_total_hours} 小时。`}
                </p>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {modelPipeline.map((step, idx) => {
                  const isOcr = step.role === 'ocr'
                  return (
                  <div key={step.step_id} style={{
                    padding: '16px 18px',
                    borderRadius: 8,
                    background: isOcr ? 'rgba(139,92,246,0.04)' : step.reuse_cache_id ? 'rgba(22,163,74,0.04)' : 'var(--gray-50)',
                    boxShadow: 'var(--shadow-ring)',
                    border: isOcr ? '1px solid rgba(139,92,246,0.2)' : step.reuse_cache_id ? '1px solid rgba(22,163,74,0.2)' : 'none',
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginBottom: 10 }}>
                      <div className="flex items-center gap-2">
                        <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--gray-400)', width: 20 }}>#{idx + 1}</span>
                        <strong style={{ fontSize: 14, color: 'var(--gray-900)' }}>{step.recommended_model_id}</strong>
                      </div>
                      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                        <span className="badge badge-blue">{ROLE_LABELS[step.role] ?? step.role}</span>
                        {isOcr ? (
                          <span style={{ fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 20, background: 'rgba(139,92,246,0.1)', color: '#7c3aed' }}>
                            内置引擎
                          </span>
                        ) : step.requires_training ? (
                          <span className="badge badge-pink">需要训练</span>
                        ) : step.reuse_cache_id ? (
                          <span className="badge badge-green">复用已有模型</span>
                        ) : (
                          <span className="badge badge-dark">无需训练</span>
                        )}
                      </div>
                    </div>
                    <div style={{ fontSize: 12, color: 'var(--gray-500)', lineHeight: 1.7 }}>
                      {step.reason_zh}
                    </div>
                    {step.reuse_info_zh && (
                      <div style={{ fontSize: 11, color: '#15803d', marginTop: 6 }}>
                        {step.reuse_info_zh}
                      </div>
                    )}
                    {isOcr && (
                      <div style={{ fontSize: 11, color: '#7c3aed', marginTop: 6 }}>
                        文字识别引擎，首次运行自动下载模型文件，无需训练
                      </div>
                    )}
                    {step.alternative_model_ids && step.alternative_model_ids.length > 0 && (
                      <div style={{ fontSize: 11, color: 'var(--gray-400)', marginTop: 6 }}>
                        备选模型：{step.alternative_model_ids.join(', ')}
                      </div>
                    )}
                  </div>
                )})}
              </div>

              {trainingStrategy?.train_mode_reason_zh && (
                <div style={{ padding: '12px 16px', borderRadius: 8, background: 'rgba(10,114,239,0.04)', marginTop: 12 }}>
                  <div style={{ fontSize: 12, color: 'var(--gray-500)', lineHeight: 1.7 }}>
                    <strong>训练模式建议：</strong>{trainingStrategy.train_mode_reason_zh}
                  </div>
                </div>
              )}
            </div>
          )}

          {pipeline && (
            <div className="card-section" style={{ marginBottom: 32, background: 'var(--gray-50)', boxShadow: 'none', border: '1px solid var(--gray-100)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16, marginBottom: 14 }}>
                <div>
                  <h3 className="text-heading-sm" style={{ marginBottom: 4 }}>高级配置预览</h3>
                  <p style={{ fontSize: 12, color: 'var(--gray-500)', margin: 0 }}>给开发交付人员查看的流水线配置，普通用户主要确认上面的检测效果和触发规则即可。</p>
                </div>
                <span className="text-mono" style={{ color: 'var(--gray-500)' }}>{pipeline.packaging.config_path}</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 12, marginBottom: 16 }}>
                <div style={{ padding: '14px 16px', borderRadius: 8, background: 'var(--gray-50)', boxShadow: 'var(--shadow-ring)' }}>
                  <div style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 4 }}>Detectors</div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--gray-900)' }}>{pipeline.detectors.length}</div>
                </div>
                <div style={{ padding: '14px 16px', borderRadius: 8, background: 'var(--gray-50)', boxShadow: 'var(--shadow-ring)' }}>
                  <div style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 4 }}>Trackers</div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--gray-900)' }}>{pipeline.trackers.length}</div>
                </div>
                <div style={{ padding: '14px 16px', borderRadius: 8, background: 'var(--gray-50)', boxShadow: 'var(--shadow-ring)' }}>
                  <div style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 4 }}>Rules</div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--gray-900)' }}>{pipeline.rules.length}</div>
                </div>
              </div>

              {trainingRecommendation && (
                <div style={{ padding: '14px 16px', borderRadius: 8, background: 'rgba(10,114,239,0.04)', boxShadow: 'var(--shadow-ring)' }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-900)', marginBottom: 6 }}>训练建议</div>
                  <div style={{ fontSize: 12, color: 'var(--gray-500)', lineHeight: 1.7 }}>
                    推荐模型：{trainingRecommendation.recommended_model} · 推荐模式：
                    {trainingRecommendation.train_mode === 'local' ? '本地训练' : '云端训练'} · 导出：
                    {trainingRecommendation.export_formats.join(', ')}
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--gray-500)', lineHeight: 1.7, marginTop: 6 }}>
                    {trainingRecommendation.reason_summary}
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}

      {draft?.revision_history && draft.revision_history.length > 0 && (
        <div className="card-section" style={{ marginBottom: 16, background: 'var(--gray-50)', boxShadow: 'none', border: '1px solid var(--gray-100)' }}>
          <h3 className="text-heading-sm" style={{ marginBottom: 10, fontSize: 13 }}>协商历史</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {draft.revision_history.map((msg, idx) => (
              <div key={idx} style={{
                padding: '8px 12px',
                borderRadius: 6,
                background: msg.role === 'user' ? 'rgba(10,114,239,0.06)' : msg.role === 'system' ? 'rgba(255,200,50,0.08)' : 'var(--white)',
                border: '1px solid var(--gray-100)',
              }}>
                <div style={{ fontSize: 10, fontWeight: 600, color: msg.role === 'user' ? 'var(--develop-blue)' : msg.role === 'system' ? '#92400e' : 'var(--gray-500)', marginBottom: 4 }}>
                  {msg.role === 'user' ? '你' : msg.role === 'system' ? '系统操作' : '系统'}
                </div>
                <div style={{ fontSize: 12, color: 'var(--gray-700)', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                  {msg.content}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {revisionSnapshots.length > 0 && (
        <div className="card-section" style={{ marginBottom: 16, background: '#fefce8', boxShadow: 'none', border: '1px solid #fde68a' }}>
          <h3 className="text-heading-sm" style={{ marginBottom: 10, fontSize: 13 }}>方案版本历史</h3>
          <p style={{ fontSize: 11, color: '#92400e', marginBottom: 12, lineHeight: 1.5 }}>
            每次修改前的方案会自动保存。你可以回滚到任意历史版本。
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {revisionSnapshots.map((snap) => (
              <div key={snap.version} style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                padding: '10px 14px', borderRadius: 8,
                background: '#fff', border: '1px solid #fde68a',
              }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--gray-900)', marginBottom: 2 }}>
                    版本 {snap.version}
                    <span style={{ fontWeight: 400, color: 'var(--gray-500)', marginLeft: 8, fontSize: 11 }}>
                      {snap.timestamp ? new Date(snap.timestamp * 1000).toLocaleString('zh-CN') : ''}
                    </span>
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--gray-600)', lineHeight: 1.5, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {snap.summary_zh || '—'}
                  </div>
                </div>
                <button
                  className="btn btn-sm btn-secondary"
                  onClick={() => handleRollback(snap.version)}
                  disabled={rollingBack || revising || confirming}
                  style={{ flexShrink: 0, marginLeft: 12, padding: '6px 14px', fontSize: 11 }}
                >
                  {rollingBack ? '回滚中...' : '回滚到此版本'}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="flex gap-3">
        <button
          className="btn btn-primary"
          onClick={handleConfirm}
          disabled={!draft || loading || confirming || revising}
          style={{ padding: '10px 24px', fontSize: 14, fontWeight: 600 }}
        >
          {confirming ? '确认中...' : '确认草图并进入数据准备'}
        </button>
        <button
          className="btn btn-secondary"
          onClick={() => setStage('manual_annotation')}
          disabled={!draft || loading || confirming || revising}
          style={{ padding: '10px 20px', borderColor: '#f59e0b', color: '#f59e0b' }}
          title="跳过自动打标，手动标注少量图片后滚雪球训练"
        >
          手动标注 (增量打标)
        </button>
        <button
          className="btn btn-secondary"
          onClick={() => setShowRevisionModal(true)}
          disabled={!draft || loading || confirming || revising}
          style={{ padding: '10px 20px' }}
        >
          反馈修改
        </button>
        <button className="btn btn-secondary" onClick={() => setStage('intent_confirm')} style={{ padding: '10px 20px' }}>
          返回协商页
        </button>
      </div>

      {showRevisionModal && (
        <div
          onClick={(e) => { if (e.target === e.currentTarget) setShowRevisionModal(false) }}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
          }}
        >
          <div style={{
            width: 560,
            maxWidth: '90vw',
            background: 'var(--white)',
            borderRadius: 12,
            padding: 24,
            boxShadow: '0 20px 48px rgba(0,0,0,0.2)',
          }}>
            <h3 className="text-heading-sm" style={{ marginBottom: 8 }}>反馈修改意见</h3>
            <p style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 16, lineHeight: 1.6 }}>
              用你自己的话说出想调整的地方，系统会重新生成方案。示例："太复杂了，简化一点" / "把模型换成更快的" / "增加一个颜色分类"。
            </p>
            <textarea
              className="input"
              value={revisionInput}
              onChange={(e) => setRevisionInput(e.target.value)}
              placeholder="例如：方案太复杂了，能不能只用一个模型？"
              style={{ height: 120, fontSize: 13, lineHeight: 1.6, marginBottom: 16 }}
              autoFocus
            />
            <div className="flex gap-3" style={{ justifyContent: 'flex-end' }}>
              <button
                className="btn btn-secondary"
                onClick={() => { setShowRevisionModal(false); setRevisionInput('') }}
                disabled={revising}
                style={{ padding: '8px 16px' }}
              >
                取消
              </button>
              <button
                className="btn btn-primary"
                onClick={handleRevise}
                disabled={revising || !revisionInput.trim()}
                style={{ padding: '8px 20px' }}
              >
                {revising ? '生成中...' : '提交并重新生成'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
