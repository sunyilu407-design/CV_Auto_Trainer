import { useState } from 'react'
import { useTaskStore } from '../store/taskStore'
import { negotiateApi, algorithmApi } from '../api/backend'

export default function ConfirmPanel() {
  const {
    taskId,
    conversationId,
    negotiatedConfig,
    negotiationConverged,
    setStage,
    setVLMResult,
    setAlgorithmPlan,
    userDescription,
    vlmResult,
    deviceProfileId,
  } = useTaskStore()

  const [confirming, setConfirming] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const canConfirm = negotiationConverged && negotiatedConfig && conversationId

  async function handleConfirm() {
    if (!taskId || !conversationId) return
    setConfirming(true)
    setError(null)

    try {
      const resp = await negotiateApi.confirm(taskId, conversationId)

      // 用确认后的 classes 更新 vlmResult
      const classes = (resp.finalized_config as any)?.classes
      if (classes && classes.length > 0) {
        setVLMResult({
          classes,
          raw_vlm_response: '',
          confidence: null,
        })
      }

      // 生成算法方案（复用 IntentConfirm 的逻辑）
      const vlmClasses = (classes ?? []).map((c: any) => ({
        class_name: c.class_name,
        prompt: c.prompt,
        negative_prompt: c.negative_prompt,
        color_hint: c.color_hint,
        display_name_zh: c.display_name_zh,
        display_prompt_zh: c.display_prompt_zh,
        display_negative_prompt_zh: c.display_negative_prompt_zh,
        display_color_hint_zh: c.display_color_hint_zh,
      }))
      const plan = await algorithmApi.generatePlan({
        task_id: taskId,
        user_description: userDescription ?? '',
        vlm_result: vlmClasses.length > 0 ? { classes: vlmClasses } : null,
        gpu_type: deviceProfileId,
        use_vlm_planner: true,
        algorithm_hints: negotiatedConfig?.algorithm_hints as any ?? null,
      })
      setAlgorithmPlan(plan)

      // 跳转到下一阶段
      setStage('algorithm_plan')
    } catch (err: any) {
      setError(err?.message || '确认失败，请重试')
    } finally {
      setConfirming(false)
    }
  }

  // 配置摘要
  const classNames = negotiatedConfig?.classes?.map(
    (c) => c.display_name_zh || c.class_name
  ) || []
  const events = negotiatedConfig?.algorithm_hints?.events || []

  return (
    <div
      style={{
        padding: '12px 16px',
        background: canConfirm ? 'rgba(16,185,129,0.04)' : 'var(--gray-50)',
        borderRadius: 10,
        border: `1px solid ${canConfirm ? 'rgba(16,185,129,0.3)' : 'var(--gray-100)'}`,
      }}
    >
      {/* Summary */}
      {negotiatedConfig && (
        <div style={{ marginBottom: 10, fontSize: 11, color: 'var(--gray-600)' }}>
          {classNames.length > 0 && (
            <div>
              <b>检测目标:</b> {classNames.join(', ')}
            </div>
          )}
          {events.length > 0 && (
            <div>
              <b>事件:</b> {events.map((e) => e.name_zh).join(', ')}
            </div>
          )}
        </div>
      )}

      {/* Status */}
      {!canConfirm && (
        <div style={{ fontSize: 11, color: 'var(--gray-400)', marginBottom: 8 }}>
          还需继续和 AI 沟通确认需求...
        </div>
      )}

      {error && (
        <div style={{ fontSize: 11, color: '#dc2626', marginBottom: 8 }}>{error}</div>
      )}

      {/* Buttons */}
      <div style={{ display: 'flex', gap: 8 }}>
        <button
          onClick={handleConfirm}
          disabled={!canConfirm || confirming}
          style={{
            flex: 1,
            padding: '10px 16px',
            borderRadius: 8,
            border: 'none',
            background: canConfirm && !confirming ? '#059669' : 'var(--gray-200)',
            color: canConfirm && !confirming ? '#fff' : 'var(--gray-400)',
            fontSize: 13,
            fontWeight: 700,
            cursor: canConfirm && !confirming ? 'pointer' : 'not-allowed',
          }}
        >
          {confirming ? '确认中...' : '✓ 确认，进入方案规划'}
        </button>
        <button
          onClick={() => setStage('upload')}
          style={{
            padding: '10px 16px',
            borderRadius: 8,
            border: '1px solid var(--gray-200)',
            background: '#fff',
            color: 'var(--gray-500)',
            fontSize: 13,
            cursor: 'pointer',
          }}
        >
          返回修改
        </button>
      </div>
    </div>
  )
}
