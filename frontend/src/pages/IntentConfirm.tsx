import { useTaskStore } from '../store/taskStore'
import PreAnnotatedToggle from '../components/PreAnnotatedToggle'

function detectScenarioLabel(userDescription: string): string {
  if (/[仓位占位占用空位]/.test(userDescription)) return '占位监测'
  if (userDescription.includes('停车') || userDescription.includes('违停')) return '违规停车'
  if (userDescription.includes('闯入') || userDescription.includes('进入区域') || userDescription.includes('越线')) {
    return '区域闯入'
  }
  if (userDescription.includes('滞留') || userDescription.includes('停留')) return '超时滞留'
  if (/tracking|track/i.test(userDescription)) return '目标跟踪'
  return '自定义事件'
}

function extractDurationLabel(userDescription: string): string {
  const secondMatch = userDescription.match(/(\d+)\s*秒/)
  if (secondMatch) return `${secondMatch[1]} 秒`

  const minuteMatch = userDescription.match(/(\d+)\s*分/)
  if (minuteMatch) return `${minuteMatch[1]} 分钟`

  return '未明确指定'
}

function extractRegions(userDescription: string): string[] {
  const matches = userDescription.match(/([A-Za-z0-9一二三四五六七八九十甲乙丙丁]+区)/g) ?? []
  return Array.from(new Set(matches))
}

function extractEvents(userDescription: string): string[] {
  const events: string[] = []
  if (userDescription.includes('进入') || userDescription.includes('闯入') || userDescription.includes('越线')) {
    events.push('进入目标区域')
  }
  if (userDescription.includes('离开') || userDescription.includes('退出')) {
    events.push('离开目标区域')
  }
  if (userDescription.includes('持续') || userDescription.includes('滞留') || userDescription.includes('停留')) {
    events.push('持续停留')
  }
  if (userDescription.includes('占位') || userDescription.includes('占用')) {
    events.push('占位成立')
  }
  if (userDescription.includes('告警') || userDescription.includes('报警')) {
    events.push('触发告警')
  }
  return events.length > 0 ? events : ['生成业务事件']
}

function extractObjects(userDescription: string, visualObjects: string[]): string[] {
  if (visualObjects.length > 0) {
    return Array.from(new Set(visualObjects.filter(Boolean)))
  }

  const inferredObjects: string[] = []
  if (userDescription.includes('人') || userDescription.includes('人员') || userDescription.includes('工人')) {
    inferredObjects.push('人员')
  }
  if (userDescription.includes('车') || userDescription.includes('车辆') || userDescription.includes('叉车')) {
    inferredObjects.push('车辆')
  }
  if (userDescription.includes('帽') || userDescription.includes('工帽') || userDescription.includes('安全帽')) {
    inferredObjects.push('工帽')
  }
  if (userDescription.includes('箱') || userDescription.includes('货柜') || userDescription.includes('货物')) {
    inferredObjects.push('货物')
  }

  return inferredObjects.length > 0 ? inferredObjects : ['关键监测对象']
}

export default function IntentConfirm() {
  const { vlmResult, vlmStatus, vlmErrorMessage, userDescription, updateVLMClass, setStage } = useTaskStore()
  const hasVisualCandidates = !!vlmResult && vlmStatus !== 'failed'

  if (!hasVisualCandidates && vlmStatus !== 'failed') {
    return (
      <div className="card-section text-center" style={{ padding: '64px 24px' }}>
        <p style={{ color: 'var(--gray-500)' }}>无需求理解结果，请先输入业务需求</p>
        <button className="btn btn-secondary" onClick={() => setStage('upload')} style={{ marginTop: 16 }}>
          返回上传
        </button>
      </div>
    )
  }

  const confidence =
    hasVisualCandidates && typeof vlmResult?.confidence === 'number' && Number.isFinite(vlmResult.confidence)
      ? Math.round((vlmResult.confidence as number) * 100)
      : null

  const getDisplayName = (classItem: NonNullable<typeof vlmResult>['classes'][number]) =>
    classItem.display_name_zh || classItem.class_name
  const getDisplayPrompt = (classItem: NonNullable<typeof vlmResult>['classes'][number]) =>
    classItem.display_prompt_zh || classItem.prompt
  const getDisplayNegative = (classItem: NonNullable<typeof vlmResult>['classes'][number]) =>
    classItem.display_negative_prompt_zh || classItem.negative_prompt
  const getDisplayColorHint = (classItem: NonNullable<typeof vlmResult>['classes'][number]) =>
    classItem.display_color_hint_zh ?? classItem.color_hint ?? ''

  const visualObjects = vlmResult?.classes.map((classItem) => getDisplayName(classItem)) ?? []
  const scenarioLabel = detectScenarioLabel(userDescription)
  const durationLabel = extractDurationLabel(userDescription)
  const regions = extractRegions(userDescription)
  const events = extractEvents(userDescription)
  const objects = extractObjects(userDescription, visualObjects)

  return (
    <div>
      <div className="page-header" style={{ marginBottom: 32 }}>
        <div className="flex items-center gap-3 mb-4">
          <div className="badge badge-blue" style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Stage 1
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
              <path d="M4 7h16" />
              <path d="M4 12h10" />
              <path d="M4 17h7" />
            </svg>
          </div>
        </div>
        <h1 className="page-title">确认需求协商</h1>
        <p className="page-subtitle">
          请确认系统对监测对象、区域、时长和触发结果的理解。确认后会生成能力草图与策略草案。
        </p>
      </div>

      {vlmStatus === 'failed' && (
        <div
          className="card-section"
          style={{ marginBottom: 16, padding: '14px 16px', background: '#fff9db', border: '1px solid #facc15', boxShadow: 'none' }}
        >
          <div style={{ fontSize: 13, fontWeight: 700, color: '#854d0e', marginBottom: 6 }}>当前协商结果主要基于文字需求</div>
          <div style={{ fontSize: 12, color: '#713f12', lineHeight: 1.6 }}>
            {vlmErrorMessage || '系统暂时无法完成视觉理解，本轮先按文字需求整理业务要点。'}
          </div>
        </div>
      )}

      <div className="card-section" style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 16, marginBottom: 16 }}>
          <div>
            <div className="badge badge-dark" style={{ marginBottom: 10 }}>{scenarioLabel}</div>
            <h2 className="text-heading-sm" style={{ marginBottom: 6 }}>系统整理出的业务需求摘要</h2>
            <p className="text-body-sm" style={{ color: 'var(--gray-500)' }}>{userDescription}</p>
          </div>
          {hasVisualCandidates && (
            <div style={{ minWidth: 120, textAlign: 'right' }}>
              <div style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 4 }}>理解置信度</div>
              <div style={{ fontSize: 28, fontWeight: 600, letterSpacing: '-1.2px', color: 'var(--gray-900)' }}>
                {confidence === null ? '--' : `${confidence}%`}
              </div>
            </div>
          )}
        </div>

        {/* VLM 视觉推理洞察 — 展示 AI 的分析思路 */}
        {hasVisualCandidates && vlmResult?.scenario_hint && (
          <div style={{ marginTop: 16, padding: '14px 16px', borderRadius: 8, background: 'linear-gradient(135deg, #f8faff 0%, #f0f7ff 100%)', border: '1px solid rgba(59,130,246,0.15)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"/>
                <path d="M12 16v-4"/>
                <path d="M12 8h.01"/>
              </svg>
              <span style={{ fontSize: 12, fontWeight: 600, color: '#3b82f6' }}>VLM 视觉推理分析</span>
              <span style={{ fontSize: 11, color: 'var(--gray-400)', marginLeft: 'auto' }}>仅供确认参考</span>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 10 }}>
              <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 4, background: 'rgba(139,92,246,0.1)', color: '#7c3aed', border: '1px solid rgba(139,92,246,0.2)' }}>
                场景: {vlmResult.scenario_hint}
              </span>
              <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 4, background: 'rgba(245,158,11,0.1)', color: '#b45309', border: '1px solid rgba(245,158,11,0.2)' }}>
                难度: {vlmResult.difficulty_hint ?? 'moderate'}
              </span>
            </div>
            {vlmResult.visual_insights && vlmResult.visual_insights.length > 0 && (
              <div style={{ fontSize: 12, color: 'var(--gray-600)', lineHeight: 1.7 }}>
                <div style={{ fontWeight: 500, color: 'var(--gray-700)', marginBottom: 4 }}>观察到的视觉特征：</div>
                <ul style={{ margin: 0, paddingLeft: 16 }}>
                  {vlmResult.visual_insights.map((insight, i) => (
                    <li key={i}>{insight}</li>
                  ))}
                </ul>
              </div>
            )}
            {vlmResult.special_considerations && vlmResult.special_considerations.length > 0 && (
              <div style={{ fontSize: 12, color: '#b45309', lineHeight: 1.7, marginTop: 6 }}>
                <div style={{ fontWeight: 500, marginBottom: 4 }}>特别注意事项：</div>
                <ul style={{ margin: 0, paddingLeft: 16 }}>
                  {vlmResult.special_considerations.map((item, i) => (
                    <li key={i}>{item}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 12 }}>
          <div style={{ padding: '14px 16px', borderRadius: 8, background: 'var(--gray-50)', boxShadow: 'var(--shadow-ring)' }}>
            <div style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 8 }}>监测对象</div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {objects.map((objectName) => (
                <span key={objectName} className="badge badge-blue">{objectName}</span>
              ))}
            </div>
          </div>
          <div style={{ padding: '14px 16px', borderRadius: 8, background: 'var(--gray-50)', boxShadow: 'var(--shadow-ring)' }}>
            <div style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 8 }}>关注区域</div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {(regions.length > 0 ? regions : ['主监测区域']).map((region) => (
                <span key={region} className="badge badge-pink">{region}</span>
              ))}
            </div>
          </div>
          <div style={{ padding: '14px 16px', borderRadius: 8, background: 'var(--gray-50)', boxShadow: 'var(--shadow-ring)' }}>
            <div style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 8 }}>时长要求</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--gray-900)' }}>{durationLabel}</div>
          </div>
          <div style={{ padding: '14px 16px', borderRadius: 8, background: 'var(--gray-50)', boxShadow: 'var(--shadow-ring)' }}>
            <div style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 8 }}>触发结果</div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {events.map((event) => (
                <span key={event} className="badge badge-red">{event}</span>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div
        className="card-section"
        style={{ marginBottom: 16, padding: '12px 16px', background: 'var(--gray-50)', boxShadow: 'none', border: '1px solid var(--gray-100)' }}
      >
        <p style={{ fontSize: 12, color: 'var(--gray-500)', margin: 0, lineHeight: 1.7 }}>
          当前页面编辑的是业务侧定义，用来帮助后续生成能力与策略草案。这里的中文说明不会覆盖底层保存的视觉提示词。
        </p>
      </div>

      {/* 跳过打标选项 — 适用于用户已有预标注数据 */}
      <PreAnnotatedToggle />

      {hasVisualCandidates ? (
        <div className="card-section" style={{ marginBottom: 32 }}>
          <div style={{ marginBottom: 16 }}>
            <h3 className="text-heading-sm" style={{ marginBottom: 6 }}>监测对象业务定义</h3>
            <p style={{ fontSize: 12, color: 'var(--gray-500)', lineHeight: 1.7, margin: 0 }}>
              请把系统识别出的候选对象改成业务团队能直接理解的表述。下一步能力草图会沿用这些名称与解释。
            </p>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }} className="stagger-children">
            {vlmResult?.classes.map((classItem, index) => (
              <div key={index} className="card-section animate-fade-in" style={{ animationDelay: `${index * 80}ms`, margin: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
                  <div
                    style={{
                      width: 32,
                      height: 32,
                      borderRadius: 8,
                      background: index === 0 ? 'var(--develop-blue)' : 'var(--gray-900)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0,
                    }}
                  >
                    <span style={{ fontSize: 12, fontWeight: 700, color: '#fff' }}>{index + 1}</span>
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 11, color: 'var(--gray-500)', marginBottom: 6 }}>业务名称</div>
                    <input
                      className="input"
                      value={getDisplayName(classItem)}
                      onChange={(e) => updateVLMClass(index, { display_name_zh: e.target.value })}
                      placeholder="例如：未佩戴工帽人员、叉车、货箱"
                      style={{ fontSize: 13, fontWeight: 600 }}
                    />
                  </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  <div className="form-group">
                    <label className="form-label">业务定义</label>
                    <textarea
                      className="input"
                      value={getDisplayPrompt(classItem)}
                      onChange={(e) => updateVLMClass(index, { display_prompt_zh: e.target.value })}
                      rows={3}
                      style={{ fontSize: 12, lineHeight: 1.6, minHeight: 88 }}
                    />
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                    <div className="form-group">
                      <label className="form-label">不计入该对象的情况</label>
                      <input
                        className="input"
                        value={getDisplayNegative(classItem)}
                        onChange={(e) => updateVLMClass(index, { display_negative_prompt_zh: e.target.value })}
                        style={{ fontSize: 12 }}
                      />
                    </div>
                    <div className="form-group">
                      <label className="form-label">辅助识别线索</label>
                      <input
                        className="input"
                        value={getDisplayColorHint(classItem)}
                        onChange={(e) => updateVLMClass(index, { display_color_hint_zh: e.target.value || null })}
                        placeholder="例如：黄色安全帽、红色车身、蓝色工服"
                        style={{ fontSize: 12 }}
                      />
                    </div>
                  </div>

                  {/* VLM 推理出的视觉特征 — 展示 AI 从图片中推断出的信息 */}
                  {classItem.estimated_size_hint && (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, padding: '10px 12px', background: 'rgba(0,0,0,0.02)', borderRadius: 6 }}>
                      <div style={{ fontSize: 11, color: 'var(--gray-400)', width: '100%', marginBottom: 4 }}>AI 从样板图推断的视觉特征（仅供参考）</div>
                      {classItem.estimated_size_hint && (
                        <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 4, background: 'rgba(34,197,94,0.08)', color: '#15803d', border: '1px solid rgba(34,197,94,0.15)' }}>
                          目标尺寸: {classItem.estimated_size_hint}
                        </span>
                      )}
                      {classItem.typical_perspective && (
                        <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 4, background: 'rgba(59,130,246,0.08)', color: '#1d4ed8', border: '1px solid rgba(59,130,246,0.15)' }}>
                          视角: {classItem.typical_perspective}
                        </span>
                      )}
                      {classItem.occlusion_tolerance && (
                        <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 4, background: 'rgba(245,158,11,0.08)', color: '#b45309', border: '1px solid rgba(245,158,11,0.15)' }}>
                          遮挡容忍: {classItem.occlusion_tolerance}
                        </span>
                      )}
                      {classItem.rotation_invariant !== undefined && (
                        <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 4, background: 'rgba(139,92,246,0.08)', color: '#7c3aed', border: '1px solid rgba(139,92,246,0.15)' }}>
                          旋转不变: {String(classItem.rotation_invariant)}
                        </span>
                      )}
                      {classItem.data_augmentation_priority && classItem.data_augmentation_priority.length > 0 && (
                        <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 4, background: 'rgba(236,72,153,0.08)', color: '#be185d', border: '1px solid rgba(236,72,153,0.15)' }}>
                          增强策略: {classItem.data_augmentation_priority.join(', ')}
                        </span>
                      )}
                    </div>
                  )}

                  <div style={{ fontSize: 11, color: 'var(--gray-400)' }}>原始候选标识：{classItem.class_name}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="card-section" style={{ marginBottom: 32 }}>
          <h3 className="text-heading-sm" style={{ marginBottom: 6 }}>本轮协商方式</h3>
          <p style={{ fontSize: 12, color: 'var(--gray-500)', lineHeight: 1.7, margin: 0 }}>
            当前没有可编辑的视觉候选对象，系统会依据文字需求直接生成能力草图与策略草案。后续补充样板图后，仍可继续细化对象定义。
          </p>
        </div>
      )}

      <div className="flex gap-3">
        <button
          className="btn btn-primary"
          onClick={() => setStage('algorithm_plan')}
          style={{ padding: '10px 24px', fontSize: 14, fontWeight: 600 }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polygon points="5 3 19 12 5 21 5 3" />
          </svg>
          进入能力草案
        </button>
        <button className="btn btn-secondary" onClick={() => setStage('upload')} style={{ padding: '10px 20px' }}>
          返回修改需求
        </button>
      </div>
    </div>
  )
}
