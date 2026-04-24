import { useEffect, useState } from 'react'
import { useTaskStore } from '../store/taskStore'
import { useSettingsStore } from '../store/settingsStore'
import { workerClient } from '../api/worker'
import { trainingApi, type AutoDLRecoveryInfo, type AutoDLRecoveryStep } from '../api/backend'
import MetricsChart from '../components/MetricsChart'

interface MultiModelProgress {
  currentIndex: number
  totalModels: number
  currentStepId: string
  currentModelId: string
  source: 'trained' | 'reuse' | null
}

interface MultiModelArtifact {
  source: 'trained' | 'reuse' | 'failed'
  model_id: string
  role: string
  cache_id?: string
  weight_path?: string
  artifacts?: Record<string, string>
}

export default function TrainingMonitor() {
  const { taskId, trainingProgress, trainConfig, setStage, setTrainingProgress, setArtifacts, artifacts, previewResults, setPreviewResults } = useTaskStore()
  const { settings } = useSettingsStore()
  const [autoDLRecovery, setAutoDLRecovery] = useState<AutoDLRecoveryInfo | null>(null)
  const [recoverySteps, setRecoverySteps] = useState<AutoDLRecoveryStep[]>([])
  const [loadingRecovery, setLoadingRecovery] = useState(false)
  const [multiModelProgress, setMultiModelProgress] = useState<MultiModelProgress | null>(null)
  const [multiModelArtifacts, setMultiModelArtifacts] = useState<Record<string, MultiModelArtifact> | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewError, setPreviewError] = useState<string | null>(null)

  useEffect(() => {
    if (trainConfig.trainMode === 'cloud' && taskId) {
      const poll = setInterval(async () => {
        try {
          const status = await trainingApi.getStatus(taskId)
          setTrainingProgress({
            state: status.state,
            currentEpoch: status.current_epoch,
            totalEpochs: status.total_epochs,
            currentMap: status.current_map,
            startedAt: new Date().toISOString(),
          })
          if (status.artifact_paths) {
            setArtifacts(status.artifact_paths)
          }
          if (status.autodl_recovery) {
            setAutoDLRecovery(status.autodl_recovery)
          }
          if (status.total_models && status.total_models > 0) {
            setMultiModelProgress({
              currentIndex: status.current_model_index ?? 0,
              totalModels: status.total_models,
              currentStepId: status.current_step_id ?? '',
              currentModelId: status.current_model_id ?? '',
              source: status.current_model_source ?? null,
            })
          }
          if (status.multi_model_artifacts) {
            setMultiModelArtifacts(status.multi_model_artifacts)
          }
          if (status.state === 'done') {
            clearInterval(poll)
            setStage('video_inference')
          }
        } catch {
          // ignore
        }
      }, 5000)
      return () => clearInterval(poll)
    }
  }, [taskId, trainConfig.trainMode, setArtifacts, setStage, setTrainingProgress])

  useEffect(() => {
    if (!autoDLRecovery || !taskId || recoverySteps.length > 0 || loadingRecovery) return
    setLoadingRecovery(true)
    trainingApi.getRecovery(taskId)
      .then((data) => setRecoverySteps(data.instructions_zh))
      .catch(() => setRecoverySteps([]))
      .finally(() => setLoadingRecovery(false))
  }, [autoDLRecovery, taskId, recoverySteps.length, loadingRecovery])

  // 训练完成后自动触发预览推理
  useEffect(() => {
    if (!trainingProgress || trainingProgress.state !== 'done') return
    if (previewLoading || previewResults.length > 0) return
    if (!artifacts || !taskId) return

    const bestWeight = artifacts["best.pt"] || artifacts["bestMap"] || artifacts["best_weight"]
    if (!bestWeight) return

    setPreviewLoading(true)
    setPreviewError(null)
    trainingApi.previewInference({
      task_id: taskId,
      weights_path: bestWeight,
      sample_images_dir: `../backend/uploads/${taskId}/labeled_images`,
      conf: trainConfig.conf,
      iou: trainConfig.iou,
      max_images: 8,
    }).then((result) => {
      const raw = result as unknown as { results: Array<Record<string, unknown>> }
      const mapped = raw.results.map((r) => ({
        imageName: r.image_name as string,
        imageBase64: r.image_base64 as string | undefined,
        detections: (r.detections as Array<Record<string, unknown>>).map((d) => ({
          className: d.class_name as string,
          confidence: d.confidence as number,
          bbox: d.bbox_xywhn as [number, number, number, number],
        })),
      }))
      setPreviewResults(mapped)
    }).catch((e) => {
      setPreviewError(`预览推理失败: ${e instanceof Error ? e.message : String(e)}`)
    }).finally(() => {
      setPreviewLoading(false)
    })
  }, [trainingProgress?.state, artifacts, taskId, trainConfig.conf, trainConfig.iou])

  const handleCancel = async () => {
    if (trainConfig.trainMode === 'local') {
      workerClient.cancel()
    } else if (taskId) {
      await trainingApi.cancel(taskId)
    }
    setStage('train_config')
  }

  const progressPercent = trainingProgress
    ? Math.round((trainingProgress.currentEpoch / trainingProgress.totalEpochs) * 100)
    : 0

  const mapValue = ((trainingProgress?.currentMap ?? 0) * 100).toFixed(1)
  const isCloudTraining = trainConfig.trainMode === 'cloud'
  const isManualFallbackUrgent = trainingProgress?.state === 'error'
  const datasetDir = taskId ? `backend/uploads/${taskId}/dataset` : 'backend/uploads/<task_id>/dataset'
  const manualBundleDir = '/tmp/cv_manual_training/manual_cloud_training'
  const exportFormats = trainConfig.exportFormats.length > 0 ? trainConfig.exportFormats : ['onnx']
  const manualBundleCommand = [
    'python scripts/prepare_manual_cloud_training.py \\',
    `  --dataset-dir ${datasetDir} \\`,
    '  --output-dir /tmp/cv_manual_training \\',
    `  --model ${trainConfig.model} \\`,
    `  --epochs ${trainConfig.epochs} \\`,
    `  --imgsz ${trainConfig.imgsz} \\`,
    ...exportFormats.map((format, index) => {
      const suffix = index === exportFormats.length - 1 ? '' : ' \\'
      return `  --export-format ${format}${suffix}`
    }),
  ].join('\n')
  const cloudTrainCommand = [
    'ssh root@<host>',
    'cd /root/workspace',
    'unzip -q dataset.zip -d dataset',
    (
      `screen -dmS train bash -c "cd /root/workspace && python cloud_scripts/train.py ` +
      `--data /root/workspace/dataset/data.yaml --model ${trainConfig.model} ` +
      `--epochs ${trainConfig.epochs} --imgsz ${trainConfig.imgsz} --lr0 ${trainConfig.lr0} ` +
      `--patience ${trainConfig.patience} --project /root/workspace/training_output"`
    ),
  ].join('\n')

  return (
    <div>
      {/* Page Header */}
      <div className="page-header" style={{ marginBottom: 32 }}>
        <div className="flex items-center gap-3 mb-4">
          <div className="badge badge-red" style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Stage 4</div>
          <div style={{ width: 32, height: 32, borderRadius: 8, background: 'var(--ship-red)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
          </div>
        </div>
        <h1 className="page-title">
          {trainConfig.trainMode === 'local' ? '本地训练' : '云端训练'}进行中
        </h1>
        <p className="page-subtitle">实时监控训练进度，训练完成后自动进入交付阶段</p>
      </div>

      {/* Progress Card */}
      <div className="card-section" style={{ marginBottom: 16 }}>
        {/* Progress header */}
        <div className="flex justify-between items-center mb-6">
          <div>
            <h2 style={{ fontSize: 15, fontWeight: 600, marginBottom: 2 }}>训练进度</h2>
            <p style={{ fontSize: 12, color: 'var(--gray-400)' }}>
              {trainConfig.trainMode === 'local' ? '本机 GPU' : settings.cloudProvider === 'autodl' ? 'AutoDL 云端' : 'SSH 云端'}
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
            <span style={{ fontSize: 36, fontWeight: 700, color: 'var(--gray-900)', letterSpacing: '-2px' }}>{progressPercent}</span>
            <span style={{ fontSize: 20, color: 'var(--gray-400)' }}>%</span>
          </div>
        </div>

        {/* Animated progress bar */}
        <div className="progress-bar" style={{ height: 8, borderRadius: 4, marginBottom: 24 }}>
          <div
            className="progress-bar-fill animated"
            style={{ width: `${progressPercent}%`, background: 'linear-gradient(90deg, var(--develop-blue), var(--preview-pink))' }}
          />
        </div>

        {/* Metrics */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 24 }}>
          <div className="metric-card">
            <span className="metric-label">当前 Epoch</span>
            <span className="metric-value" style={{ fontSize: 28 }}>
              {trainingProgress?.currentEpoch ?? 0}
              <span style={{ fontSize: 14, color: 'var(--gray-400)', fontWeight: 400, marginLeft: 4 }}>/ {trainingProgress?.totalEpochs ?? trainConfig.epochs}</span>
            </span>
          </div>
          <div className="metric-card">
            <span className="metric-label">mAP50</span>
            <span className="metric-value" style={{ fontSize: 28 }}>
              {mapValue}
              <span style={{ fontSize: 14, color: 'var(--gray-400)', fontWeight: 400, marginLeft: 2 }}>%</span>
            </span>
          </div>
          <div className="metric-card">
            <span className="metric-label">训练模式</span>
            <span className="metric-value" style={{ fontSize: 16, marginTop: 6 }}>
              {trainConfig.trainMode === 'local' ? (
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--develop-blue)" strokeWidth="2"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
                  本地 GPU
                </span>
              ) : (
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--preview-pink)" strokeWidth="2"><path d="M18 10h-1.26A8 8 0 109 20h9a5 5 0 000-10z"/></svg>
                  云端
                </span>
              )}
            </span>
          </div>
        </div>
      </div>

      {multiModelProgress && multiModelProgress.totalModels > 1 && (
        <div className="card-section" style={{ marginBottom: 16, background: 'rgba(10,114,239,0.04)', boxShadow: 'none', border: '1px solid rgba(10,114,239,0.2)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12, gap: 12, flexWrap: 'wrap' }}>
            <div>
              <h3 className="text-heading-sm" style={{ margin: 0, marginBottom: 4 }}>多模型流水线训练</h3>
              <p style={{ fontSize: 12, color: 'var(--gray-500)', margin: 0 }}>
                按优先级顺序训练，一个完成后自动训练下一个
              </p>
            </div>
            <div className="badge badge-blue" style={{ fontSize: 12 }}>
              第 {multiModelProgress.currentIndex + 1} / {multiModelProgress.totalModels} 个
            </div>
          </div>

          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {Array.from({ length: multiModelProgress.totalModels }).map((_, idx) => {
              const isCurrent = idx === multiModelProgress.currentIndex
              const isDone = idx < multiModelProgress.currentIndex
              return (
                <div
                  key={idx}
                  style={{
                    flex: '1 1 140px',
                    padding: '10px 12px',
                    borderRadius: 8,
                    background: isCurrent ? 'var(--develop-blue)' : isDone ? 'rgba(22,163,74,0.12)' : 'var(--gray-100)',
                    color: isCurrent ? '#fff' : isDone ? '#15803d' : 'var(--gray-500)',
                    fontSize: 11,
                    fontWeight: 600,
                    transition: 'all 0.2s ease',
                  }}
                >
                  <div style={{ fontSize: 10, opacity: 0.8, marginBottom: 2 }}>
                    模型 #{idx + 1}
                  </div>
                  {isCurrent && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div className="spinner" style={{ width: 10, height: 10, borderWidth: 2 }} />
                      <span>{multiModelProgress.currentModelId}</span>
                    </div>
                  )}
                  {isDone && <div>✓ 完成</div>}
                  {!isCurrent && !isDone && <div>等待中</div>}
                </div>
              )
            })}
          </div>

          {multiModelProgress.source === 'reuse' && (
            <div style={{ marginTop: 10, padding: '8px 12px', background: 'rgba(22,163,74,0.08)', borderRadius: 6, fontSize: 12, color: '#15803d' }}>
              当前步骤复用了之前训练好的模型，跳过训练
            </div>
          )}

          {multiModelArtifacts && Object.keys(multiModelArtifacts).length > 0 && (
            <div style={{ marginTop: 12, fontSize: 11, color: 'var(--gray-600)' }}>
              已完成步骤：{Object.entries(multiModelArtifacts).filter(([, a]) => a.source !== 'failed').length} 个
            </div>
          )}
        </div>
      )}

      {/* Metrics Chart */}
      <MetricsChart currentEpoch={trainingProgress?.currentEpoch ?? 0} currentMap={trainingProgress?.currentMap ?? 0} />

      {/* Preview Inference Results */}
      {(previewLoading || previewResults.length > 0 || previewError) && (
        <div className="card-section" style={{ marginTop: 24 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
            <div>
              <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 2 }}>效果预览</h3>
              <p style={{ fontSize: 12, color: 'var(--gray-400)' }}>用训练好的模型在样板上推理，直观判断识别效果</p>
            </div>
            {previewLoading && <div className="spinner" />}
          </div>

          {previewError && (
            <div style={{ padding: '12px 16px', background: 'rgba(255,91,79,0.06)', border: '1px solid rgba(255,91,79,0.2)', borderRadius: 8, fontSize: 13, color: 'var(--ship-red)', marginBottom: 16 }}>
              {previewError}
            </div>
          )}

          {previewResults.length > 0 && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 16 }}>
              {previewResults.map((result) => (
                <div key={result.imageName} style={{ border: '1px solid var(--gray-100)', borderRadius: 10, overflow: 'hidden' }}>
                  <img
                    src={`data:image/jpeg;base64,${result.imageBase64}`}
                    alt={result.imageName}
                    style={{ width: '100%', height: 200, objectFit: 'cover', display: 'block' }}
                  />
                  <div style={{ padding: '10px 12px' }}>
                    <div style={{ fontSize: 11, color: 'var(--gray-500)', marginBottom: 6, fontWeight: 500 }}>
                      {result.imageName}
                    </div>
                    {result.detections.length > 0 ? (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                        {result.detections.map((det, i) => (
                          <div key={i} style={{
                            display: 'inline-flex', alignItems: 'center', gap: 4,
                            padding: '3px 8px', borderRadius: 4,
                            background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.2)',
                            fontSize: 11, color: '#15803d',
                          }}>
                            <span style={{ fontWeight: 600 }}>{det.className}</span>
                            <span style={{ opacity: 0.7 }}>{(det.confidence * 100).toFixed(0)}%</span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div style={{ fontSize: 12, color: 'var(--ship-red)', fontStyle: 'italic' }}>未检出目标</div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}

          {!previewLoading && previewResults.length === 0 && !previewError && (
            <div style={{ fontSize: 13, color: 'var(--gray-400)', textAlign: 'center', padding: '20px 0' }}>
              训练完成，正在生成预览...
            </div>
          )}
        </div>
      )}

      {/* Delivery Button */}
      {trainingProgress?.state === 'done' && (
        <div style={{ marginTop: 24, display: 'flex', gap: 12 }}>
          <button
            className="btn btn-primary"
            onClick={() => setStage('video_inference')}
            style={{ padding: '12px 28px', fontSize: 15, fontWeight: 600 }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="23 7 16 12 23 5 21 2 16 12 21 22 23 17 16 22 8 13 16 8 3 17 8 12 3 17 8 12"/></svg>
            视频推理演示
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => setStage('delivery')}
            style={{ padding: '12px 24px', fontSize: 14 }}>
            交付物导出
          </button>
        </div>
      )}

      {isCloudTraining && !autoDLRecovery && (
        <div
          className="card-section"
          style={{
            marginTop: 24,
            background: isManualFallbackUrgent ? 'rgba(255, 91, 79, 0.06)' : 'rgba(222, 29, 141, 0.04)',
            border: `1px solid ${isManualFallbackUrgent ? 'rgba(255, 91, 79, 0.24)' : 'rgba(222, 29, 141, 0.16)'}`,
          }}
        >
          <div className="flex justify-between items-center mb-4" style={{ gap: 12, flexWrap: 'wrap' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
                <span className="badge" style={{ background: isManualFallbackUrgent ? 'rgba(255, 91, 79, 0.12)' : 'rgba(222, 29, 141, 0.12)', color: isManualFallbackUrgent ? 'var(--ship-red)' : 'var(--preview-pink)' }}>
                  {isManualFallbackUrgent ? 'Cloud Fallback Required' : 'Cloud Fallback'}
                </span>
                {taskId && <span className="badge badge-dark">Task {taskId}</span>}
              </div>
              <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>
                {isManualFallbackUrgent ? '手动云训练兜底方案' : '手动云训练预案'}
              </h2>
              <p style={{ fontSize: 13, color: 'var(--gray-500)', lineHeight: 1.7, margin: 0 }}>
                {isManualFallbackUrgent
                  ? '当前云训练状态异常。先在本地生成手动训练包，确认 ssh/scp 可用后，再去启动按时计费的云端实例。'
                  : '如果系统后续连不上云训练环境，不要先开付费实例。先在本地生成手动训练包，再确认可以上传和登录云电脑。'}
              </p>
            </div>
          </div>

          <div style={{ marginBottom: 16, padding: '12px 14px', borderRadius: 10, background: 'rgba(255,255,255,0.88)', boxShadow: 'var(--shadow-border)' }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: isManualFallbackUrgent ? 'var(--ship-red)' : 'var(--preview-pink)', marginBottom: 6 }}>
              本地先生成训练包
            </div>
            <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 12, lineHeight: 1.7, fontFamily: 'var(--font-mono)', color: 'var(--gray-900)' }}>
              {manualBundleCommand}
            </pre>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 12, marginBottom: 16 }}>
            <div style={{ padding: '12px 14px', borderRadius: 10, background: '#fff', boxShadow: 'var(--shadow-border)' }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--develop-blue)', marginBottom: 8 }}>会生成这些文件</div>
              <div style={{ fontSize: 12, color: 'var(--gray-600)', lineHeight: 1.8 }}>
                <div>{manualBundleDir}/dataset.zip</div>
                <div>{manualBundleDir}/cloud_scripts/train.py</div>
                <div>{manualBundleDir}/cloud_scripts/export.py</div>
                <div>{manualBundleDir}/cloud_scripts/health_check.py</div>
                <div>{manualBundleDir}/README.md</div>
              </div>
            </div>
            <div style={{ padding: '12px 14px', borderRadius: 10, background: '#fff', boxShadow: 'var(--shadow-border)' }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--develop-blue)', marginBottom: 8 }}>先做的检查</div>
              <div style={{ fontSize: 12, color: 'var(--gray-600)', lineHeight: 1.8 }}>
                <div>1. 数据集目录已在 {datasetDir}</div>
                <div>2. 本机可执行 `python`、`ssh`、`scp`</div>
                <div>3. 上传和登录链路打通后，再启动付费云实例</div>
              </div>
            </div>
          </div>

          <div style={{ marginBottom: 16, padding: '12px 14px', borderRadius: 10, background: '#fff', boxShadow: 'var(--shadow-border)' }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--develop-blue)', marginBottom: 6 }}>上传到云电脑</div>
            <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 12, lineHeight: 1.7, fontFamily: 'var(--font-mono)', color: 'var(--gray-900)' }}>
{`scp ${manualBundleDir}/dataset.zip root@<host>:/root/workspace/
scp -r ${manualBundleDir}/cloud_scripts root@<host>:/root/workspace/`}
            </pre>
          </div>

          <div style={{ marginBottom: 16, padding: '12px 14px', borderRadius: 10, background: '#fff', boxShadow: 'var(--shadow-border)' }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--develop-blue)', marginBottom: 6 }}>云端启动训练</div>
            <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 12, lineHeight: 1.7, fontFamily: 'var(--font-mono)', color: 'var(--gray-900)' }}>
              {cloudTrainCommand}
            </pre>
          </div>

          <div style={{ fontSize: 12, color: 'var(--gray-600)', lineHeight: 1.9 }}>
            <div>完整手册：`MANUAL_CLOUD_TRAINING.md`</div>
            <div>训练完成后，用 `python cloud_scripts/export.py --weights /root/workspace/training_output/exp/weights/best.pt --format {exportFormats[0]}` 导出模型。</div>
            <div>如果需要持续查看训练状态，可在云端运行 `python cloud_scripts/health_check.py` 或查看 `training_output/exp/results.csv`。</div>
          </div>
        </div>
      )}

      {autoDLRecovery && (
        <div
          className="card-section"
          style={{
            marginTop: 24,
            background: 'rgba(255, 91, 79, 0.06)',
            border: '2px solid var(--ship-red)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--ship-red)" strokeWidth="2">
              <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
              <line x1="12" y1="9" x2="12" y2="13"/>
              <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
            <h2 style={{ fontSize: 18, fontWeight: 700, margin: 0, color: 'var(--ship-red)' }}>
              AutoDL 训练异常 · 手动救回教程
            </h2>
          </div>
          <p style={{ fontSize: 13, color: 'var(--gray-700)', lineHeight: 1.7, marginBottom: 16 }}>
            实例 <strong>{autoDLRecovery.instance_id}</strong> 可能仍在运行，系统<strong>没有自动关机</strong>以避免浪费。请按下方步骤手动处理，或到 AutoDL 控制台直接关机。
          </p>

          {autoDLRecovery.error_msg && (
            <div style={{ padding: '10px 12px', background: '#fff', borderRadius: 6, marginBottom: 16, fontSize: 12, color: 'var(--gray-700)', fontFamily: 'var(--font-mono)' }}>
              <strong style={{ color: 'var(--ship-red)' }}>错误信息：</strong> {autoDLRecovery.error_msg}
            </div>
          )}

          {recoverySteps.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {recoverySteps.map((step) => (
                <div key={step.step} style={{ padding: '14px 16px', background: '#fff', borderRadius: 8, boxShadow: 'var(--shadow-border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                    <div style={{
                      width: 24, height: 24, borderRadius: '50%',
                      background: 'var(--ship-red)', color: '#fff',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: 12, fontWeight: 700,
                    }}>{step.step}</div>
                    <strong style={{ fontSize: 14, color: 'var(--gray-900)' }}>{step.title}</strong>
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--gray-600)', lineHeight: 1.6, marginBottom: step.action ? 10 : 0 }}>
                    {step.description}
                  </div>
                  {step.action && (
                    <pre style={{
                      margin: 0, padding: '10px 12px',
                      background: 'var(--gray-900)', color: '#a8e5a0',
                      borderRadius: 6, fontSize: 12, lineHeight: 1.7,
                      fontFamily: 'var(--font-mono)',
                      whiteSpace: 'pre-wrap', wordBreak: 'break-all',
                    }}>
                      {step.action}
                    </pre>
                  )}
                  {step.password && (
                    <div style={{ marginTop: 8, padding: '8px 12px', background: '#fff8e1', borderRadius: 6, fontSize: 12, fontFamily: 'var(--font-mono)' }}>
                      <strong>SSH 密码：</strong>
                      <code style={{ marginLeft: 8, userSelect: 'all' }}>{step.password}</code>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div style={{ fontSize: 12, color: 'var(--gray-500)' }}>
              {loadingRecovery ? '正在加载救回教程...' : '点击下方按钮加载教程'}
            </div>
          )}

          {autoDLRecovery.autodl_console_url && (
            <div style={{ marginTop: 16, padding: '12px 14px', background: '#fff8e1', borderRadius: 8, borderLeft: '3px solid #f59e0b' }}>
              <strong style={{ fontSize: 12, color: '#92400e' }}>⚠️ 重要提示</strong>
              <div style={{ fontSize: 12, color: 'var(--gray-700)', lineHeight: 1.7, marginTop: 4 }}>
                AutoDL 按使用时长计费。处理完请务必到{' '}
                <a href={autoDLRecovery.autodl_console_url} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--develop-blue)', textDecoration: 'underline' }}>
                  AutoDL 控制台
                </a>{' '}手动关机！
              </div>
            </div>
          )}
        </div>
      )}

      {/* Cancel */}
      <div style={{ marginTop: 24 }}>
        <button className="btn btn-danger" onClick={handleCancel} style={{ padding: '10px 20px' }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
          取消训练
        </button>
      </div>
    </div>
  )
}
