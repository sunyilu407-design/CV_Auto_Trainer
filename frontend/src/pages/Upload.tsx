import { useState, useRef, useCallback, useEffect } from 'react'
import { useTaskStore, DEVICE_PROFILES, type DeviceProfileId } from '../store/taskStore'
import { filesApi, taskApi, vlmApi, trainingApi } from '../api/backend'
import AnnotationCanvas from '../components/AnnotationCanvas'

export default function Upload() {
  const {
    setTaskMeta,
    setStage,
    setVLMResult,
    setVLMStatus,
    setAlgorithmPlan,
    resetNegotiation,
    sampleImages,
    datasetImages,
    sampleImageBoxes,
    setSampleImages,
    setDatasetImages,
    setSampleImageBoxes,
    setLabelingImageDir,
    userDescription,
    setUserDescription,
    deviceProfileId,
    setDeviceProfileId,
    trainConfig,
    setTrainConfig,
    algorithmPlan,
  } = useTaskStore()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeImageIndex, setActiveImageIndex] = useState(0)
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoInfo, setVideoInfo] = useState<{ fps: number; duration_seconds: number; width: number; height: number; frame_count?: number } | null>(null)
  const [videoUploading, setVideoUploading] = useState(false)
  const [annotationFiles, setAnnotationFiles] = useState<File[]>([])
  const sampleFileInputRef = useRef<HTMLInputElement>(null)
  const datasetFileInputRef = useRef<HTMLInputElement>(null)
  const videoFileInputRef = useRef<HTMLInputElement>(null)

  // ── Incremental mode: separate file tracking for badcase images ─────────
  const [incrImages, setIncrImages] = useState<File[]>([])
  const [incrUploading, setIncrUploading] = useState(false)
  const [incrError, setIncrError] = useState<string | null>(null)
  const [incrConfirmData, setIncrConfirmData] = useState<{
    new_images: number; old_images: number; auto_labeled: number;
    duplicates_removed: number; base_model_path: string;
    merged_dataset_dir: string; data_yaml: string;
    recommended_config: { epochs: number; lr0: number; patience: number; imgsz: number }
  } | null>(null)
  const incrFileInputRef = useRef<HTMLInputElement>(null)

  const handleIncrFilesSelected = useCallback((files: FileList | null) => {
    if (!files) return
    const allFiles = Array.from(files)
    const imageExts = new Set(['jpg', 'jpeg', 'png'])
    const newImgs = allFiles.filter((f) => {
      const ext = f.name.split('.').pop()?.toLowerCase()
      return ext && imageExts.has(ext)
    })
    setIncrImages((prev) => {
      const existing = new Set(prev.map((x) => `${x.name}-${x.size}`))
      const unique = newImgs.filter((x) => !existing.has(`${x.name}-${x.size}`))
      return [...prev, ...unique]
    })
  }, [])

  // ── Incremental mode: upload then call startIncremental ─────────────────
  const handleIncrementalUpload = async () => {
    if (!taskId || incrImages.length === 0) return
    setIncrUploading(true)
    setIncrError(null)
    try {
      // 1. Upload badcase images to uploads/{taskId}/incremental_images/
      for (const file of incrImages) {
        const formData = new FormData()
        formData.append('file', file)
        await filesApi.upload(taskId, formData, 'incremental_images')
      }
      // 2. Backend merges old+new, auto-labels new images with existing best.pt
      const classNames = (algorithmPlan?.algorithm_plan?.targets ?? []).map(
        (t: { class_name?: string }) => t.class_name ?? 'target'
      )
      const result = await trainingApi.startIncremental(taskId, {
        auto_label_new: true,
        class_names: classNames,
      })
      // 3. Show confirmation dialog before navigating to train_config
      setIncrConfirmData({
        new_images: result.new_images,
        old_images: result.old_images,
        auto_labeled: result.auto_labeled,
        duplicates_removed: result.duplicates_removed,
        base_model_path: result.base_model_path,
        merged_dataset_dir: result.merged_dataset_dir,
        data_yaml: result.data_yaml,
        recommended_config: result.recommended_config,
      })
    } catch (e: unknown) {
      setIncrError(e instanceof Error ? e.message : '增量数据处理失败')
    } finally {
      setIncrUploading(false)
    }
  }

  const confirmIncremental = () => {
    if (!incrConfirmData) return
    setTrainConfig({
      incrementalMode: true,
      baseModelPath: incrConfirmData.base_model_path,
      model: incrConfirmData.base_model_path,
      incrementalDatasetDir: incrConfirmData.merged_dataset_dir,
      incrementalDataYaml: incrConfirmData.data_yaml,
      epochs: incrConfirmData.recommended_config.epochs,
      lr0: incrConfirmData.recommended_config.lr0,
      patience: incrConfirmData.recommended_config.patience,
      imgsz: incrConfirmData.recommended_config.imgsz,
    })
    setIncrConfirmData(null)
    setStage('train_config')
  }

  // ── Standard upload ────────────────────────────────────────────────

  const activeBoxes = sampleImageBoxes.find((item) => item.imageIndex === activeImageIndex)?.boxes ?? []

  const handleSampleFilesSelected = useCallback((files: FileList | null) => {
    if (!files) return
    const newFiles = Array.from(files).slice(0, 3 - sampleImages.length)
    setSampleImages([...sampleImages, ...newFiles])
  }, [sampleImages, setSampleImages])

  const handleDatasetFilesSelected = useCallback((files: FileList | null) => {
    if (!files) return
    const allFiles = Array.from(files)
    const imageExts = new Set(['jpg', 'jpeg', 'png'])
    const txtFiles: File[] = []
    const imageFiles: File[] = []

    for (const file of allFiles) {
      const ext = file.name.split('.').pop()?.toLowerCase()
      if (ext === 'txt') {
        txtFiles.push(file)
      } else if (ext && imageExts.has(ext)) {
        imageFiles.push(file)
      }
    }

    const existingImageNames = new Set(datasetImages.map((f) => `${f.name}-${f.size}`))
    const newImages = imageFiles.filter((f) => !existingImageNames.has(`${f.name}-${f.size}`))

    setDatasetImages([...datasetImages, ...newImages])
    if (txtFiles.length > 0) {
      setAnnotationFiles((prev) => {
        const existing = new Set(prev.map((f) => `${f.name}-${f.size}`))
        const unique = txtFiles.filter((f) => !existing.has(`${f.name}-${f.size}`))
        return [...prev, ...unique]
      })
    }
  }, [datasetImages, setDatasetImages])

  const handleVideoSelected = useCallback(async (files: FileList | null) => {
    if (!files || files.length === 0) return
    const file = files[0]
    setVideoFile(file)
    setVideoInfo(null)
  }, [])

  const uploadVideoAndExtractFrames = async (currentTaskId: string) => {
    if (!videoFile) return
    setVideoUploading(true)
    try {
      const formData = new FormData()
      formData.append('video', videoFile)
      const result = await filesApi.uploadVideo(currentTaskId, formData, 'training')
      setVideoInfo({
        fps: result.video_info.fps,
        duration_seconds: result.video_info.duration_seconds,
        width: result.video_info.width,
        height: result.video_info.height,
        frame_count: result.frame_count,
      })
      if (datasetImages.length === 0) {
        setLabelingImageDir(`../backend/uploads/${currentTaskId}/video_frames`)
      }
    } finally {
      setVideoUploading(false)
    }
  }

  const createTaskAndUploadDataset = async () => {
    const taskName = `dataset-${new Date().toISOString()}`
    const task = await taskApi.create(taskName)
    const currentTaskId = task.id
    setTaskMeta(currentTaskId, task.name || taskName)

    for (const file of datasetImages) {
      const formData = new FormData()
      formData.append('file', file)
      await filesApi.upload(currentTaskId, formData, 'images')
    }

    // 上传预标注的 YOLO .txt 文件
    for (const file of annotationFiles) {
      const formData = new FormData()
      formData.append('file', file)
      await filesApi.upload(currentTaskId, formData, 'images')
    }
    if (datasetImages.length > 0) {
      setLabelingImageDir(`../backend/uploads/${currentTaskId}/images`)
    }

    return currentTaskId
  }

  const handleParse = async () => {
    if ((datasetImages.length === 0 && !videoFile) || !userDescription.trim()) {
      setError('请上传待处理图片并输入业务需求')
      return
    }

    setLoading(true)
    setError(null)
    let taskReadyForTextFallback = false

    try {
      setAlgorithmPlan(null)
      const currentTaskId = await createTaskAndUploadDataset()
      if (videoFile) {
        await uploadVideoAndExtractFrames(currentTaskId)
      }
      taskReadyForTextFallback = true

      const imagesBase64 = await Promise.all(
        sampleImages.map((f) =>
          new Promise<string>((resolve, reject) => {
            const reader = new FileReader()
            reader.onload = () => {
              const result = reader.result as string
              resolve(result.split(',')[1])
            }
            reader.onerror = reject
            reader.readAsDataURL(f)
          })
        )
      )

      const allSampleBoxes = sampleImageBoxes.flatMap((item) => item.boxes)
      const result = await vlmApi.parse(imagesBase64, userDescription, allSampleBoxes)
      const rawResponse = typeof result.raw_vlm_response === 'string' ? result.raw_vlm_response : ''
      const confidence = typeof result.confidence === 'number' ? result.confidence : null
      const classes = (result.classes as Array<Record<string, unknown>>).map((item) => ({
        class_name: String(item.class_name ?? ''),
        prompt: String(item.prompt ?? ''),
        negative_prompt: String(item.negative_prompt ?? ''),
        color_hint: (item.color_hint as string | null | undefined) ?? null,
        display_name_zh: String(item.display_name_zh ?? item.class_name ?? ''),
        display_prompt_zh: String(item.display_prompt_zh ?? item.prompt ?? ''),
        display_negative_prompt_zh: String(item.display_negative_prompt_zh ?? item.negative_prompt ?? ''),
        display_color_hint_zh:
          item.display_color_hint_zh == null ? (item.color_hint as string | null | undefined) ?? null : String(item.display_color_hint_zh),
      }))

      if (result.status === 'success') {
        setVLMResult({
          classes,
          raw_vlm_response: rawResponse,
          confidence,
        })
        setVLMStatus('success', null)
      } else {
        setVLMResult(null)
        setVLMStatus('failed', result.message)
        setError(result.message)
      }

      // 清空旧的协商状态（后端已在 update_vlm_result 时清理 DB 对话）
      resetNegotiation()
      setStage('intent_confirm')
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : '视觉解析失败'
      setVLMResult(null)
      setVLMStatus('failed', message)
      if (taskReadyForTextFallback) {
        resetNegotiation()
        setStage('intent_confirm')
      } else {
        setError(message)
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      {/* Page Header */}
      <div className="page-header" style={{ marginBottom: 32 }}>
        <div className="flex items-center gap-3 mb-4">
          <div className="badge badge-blue" style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Stage 1</div>
          <div style={{ width: 32, height: 32, borderRadius: 8, background: 'var(--gray-900)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>
        </div>
        <h1 className="page-title">输入业务需求</h1>
        <p className="page-subtitle">系统会先理解你的业务需求，再结合可选样板图补充视觉细节，生成策略草案</p>
      </div>

      {/* ── Incremental mode: badcase upload ─────────────────────────────────── */}
      {trainConfig.incrementalMode && taskId && (
        <div className="card-section" style={{ marginBottom: 24, background: 'rgba(245,158,11,0.04)', border: '1px solid rgba(245,158,11,0.3)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
            <span style={{ fontSize: 18 }}>⚡</span>
            <div>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#92400e' }}>增量训练模式</div>
              <div style={{ fontSize: 12, color: '#a16207', marginTop: 2 }}>
                上传新的 badcase 图片，系统会用已有模型自动标注，再合并旧数据生成增量数据集
              </div>
            </div>
          </div>

          {/* Selected images list */}
          {incrImages.length > 0 && (
            <div style={{ marginBottom: 12, fontSize: 13, color: '#a16207' }}>
              已选择 <strong>{incrImages.length}</strong> 张增量图片
              <span style={{ marginLeft: 12, color: 'var(--gray-400)', cursor: 'pointer' }}
                onClick={() => setIncrImages([])}>
                清除
              </span>
            </div>
          )}

          {/* Error */}
          {incrError && (
            <div style={{ marginBottom: 12, padding: '8px 12px', background: '#fff5f5', borderRadius: 6, fontSize: 13, color: 'var(--ship-red)' }}>
                {incrError}
            </div>
          )}

          {/* Actions */}
          <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
            <button
              onClick={() => incrFileInputRef.current?.click()}
              style={{
                padding: '10px 18px', borderRadius: 8, border: '1.5px solid rgba(245,158,11,0.5)',
                background: 'rgba(245,158,11,0.06)', cursor: 'pointer', fontSize: 13, fontWeight: 600, color: '#92400e',
              }}
            >
              📁 上传 Badcase 图片
            </button>
            <input
              ref={incrFileInputRef}
              type="file"
              accept="image/jpeg,image/png"
              multiple
              style={{ display: 'none' }}
              onChange={(e) => handleIncrFilesSelected(e.target.files)}
            />

            <button
              onClick={handleIncrementalUpload}
              disabled={incrUploading || incrImages.length === 0}
              style={{
                padding: '10px 24px', borderRadius: 8, border: 'none',
                background: incrImages.length === 0 ? 'var(--gray-200)' : '#f59e0b',
                color: incrImages.length === 0 ? 'var(--gray-400)' : '#fff',
                cursor: incrImages.length === 0 ? 'not-allowed' : 'pointer',
                fontSize: 13, fontWeight: 700,
              }}
            >
              {incrUploading ? '处理中...' : '开始增量训练准备 →'}
            </button>

            <button
              onClick={() => {
                setTrainConfig({ incrementalMode: false, incrementalDatasetDir: null, incrementalDataYaml: null })
              }}
              style={{ padding: '8px 16px', borderRadius: 8, border: '1px solid var(--gray-200)', background: 'transparent', cursor: 'pointer', fontSize: 12, color: 'var(--gray-500)' }}
            >
              取消增量训练
            </button>
          </div>
        </div>
      )}

      {/* ── Incremental confirmation: show merge stats before proceeding ─────────── */}
      {incrConfirmData && (
        <div className="card-section" style={{ marginBottom: 24, border: '1px solid rgba(245,158,11,0.4)', background: 'rgba(245,158,11,0.03)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
            <span style={{ fontSize: 18 }}>⚡</span>
            <span style={{ fontSize: 14, fontWeight: 700, color: '#92400e' }}>增量数据合并完成</span>
            <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 4, background: 'rgba(245,158,11,0.12)', color: '#92400e' }}>
              v{incrConfirmData.old_images > 0 ? '→v' : '1'}
            </span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 14 }}>
            {[
              { label: '旧图片', value: incrConfirmData.old_images, color: '#6b7280' },
              { label: '新增图片', value: incrConfirmData.new_images, color: '#2563eb' },
              { label: '自动标注', value: incrConfirmData.auto_labeled, color: '#16a34a' },
              { label: '去重移除', value: incrConfirmData.duplicates_removed, color: '#9333ea' },
            ].map(({ label, value, color }) => (
              <div key={label} style={{ textAlign: 'center', padding: '10px 8px', background: '#fff', borderRadius: 8, border: '1px solid var(--gray-100)' }}>
                <div style={{ fontSize: 22, fontWeight: 800, color }}>{value}</div>
                <div style={{ fontSize: 11, color: 'var(--gray-400)', marginTop: 2 }}>{label}</div>
              </div>
            ))}
          </div>
          <div style={{ fontSize: 12, color: '#92400e', marginBottom: 14, lineHeight: 1.6 }}>
            基础模型：<code style={{ fontFamily: 'var(--font-mono)' }}>{incrConfirmData.base_model_path.split('/').pop()}</code>
            &nbsp;·&nbsp;推荐训练参数：Epochs {incrConfirmData.recommended_config.epochs}
            &nbsp;·&nbsp;学习率 {incrConfirmData.recommended_config.lr0}
            &nbsp;·&nbsp;早停 {incrConfirmData.recommended_config.patience}
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <button
              onClick={confirmIncremental}
              style={{ padding: '10px 24px', borderRadius: 8, border: 'none', background: '#f59e0b', color: '#fff', fontWeight: 700, cursor: 'pointer', fontSize: 13 }}
            >
              确认并进入训练配置 →
            </button>
            <button
              onClick={() => setIncrConfirmData(null)}
              style={{ padding: '10px 16px', borderRadius: 8, border: '1px solid var(--gray-200)', background: 'transparent', cursor: 'pointer', fontSize: 12, color: 'var(--gray-500)' }}
            >
              返回修改
            </button>
          </div>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
        {/* Left: Image Upload */}
        <div>
          <div className="card-section" style={{ padding: 0, overflow: 'hidden' }}>
            {sampleImages.length === 0 ? (
              <div
                onClick={() => sampleFileInputRef.current?.click()}
                style={{
                  padding: '48px 24px',
                  textAlign: 'center',
                  cursor: 'pointer',
                  transition: 'background 0.15s ease',
                  borderBottom: '1px solid var(--gray-100)',
                }}
                onMouseEnter={(e) => ((e.currentTarget as HTMLElement).style.background = 'var(--gray-50)')}
                onMouseLeave={(e) => ((e.currentTarget as HTMLElement).style.background = 'transparent')}
              >
                <input ref={sampleFileInputRef} type="file" accept="image/jpeg,image/png" multiple onChange={(e) => handleSampleFilesSelected(e.target.files)} style={{ display: 'none' }} />
                <div style={{ width: 48, height: 48, borderRadius: 12, background: 'var(--gray-100)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px' }}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--gray-500)" strokeWidth="1.5">
                    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                </div>
                <p style={{ fontSize: 14, color: 'var(--gray-600)', marginBottom: 4 }}>添加可选样板图</p>
                <p style={{ fontSize: 12, color: 'var(--gray-400)' }}>JPG/PNG，最大 10MB，最多 3 张，用于补充视觉细节</p>
              </div>
            ) : (
              <div style={{ padding: 20 }}>
                <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
                  {sampleImages.map((img, i) => (
                    <div
                      key={i}
                      onClick={() => setActiveImageIndex(i)}
                      style={{
                        width: 64,
                        height: 64,
                        borderRadius: 8,
                        overflow: 'hidden',
                        cursor: 'pointer',
                        border: `2px solid ${activeImageIndex === i ? 'var(--develop-blue)' : 'var(--gray-100)'}`,
                        boxShadow: activeImageIndex === i ? '0 0 0 2px rgba(10,114,239,0.15)' : 'none',
                        transition: 'all 0.15s ease',
                        position: 'relative',
                      }}
                    >
                      <img src={URL.createObjectURL(img)} alt={`样板图 ${i + 1}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                      <div style={{ position: 'absolute', top: 4, left: 4, background: 'rgba(0,0,0,0.5)', color: '#fff', fontSize: 10, padding: '1px 5px', borderRadius: 4 }}>
                        {i + 1}
                      </div>
                    </div>
                  ))}
                  {sampleImages.length < 3 && (
                    <>
                      <input ref={sampleFileInputRef} type="file" accept="image/jpeg,image/png" multiple onChange={(e) => handleSampleFilesSelected(e.target.files)} style={{ display: 'none' }} />
                      <div
                        onClick={() => sampleFileInputRef.current?.click()}
                        style={{
                          width: 64,
                          height: 64,
                          borderRadius: 8,
                          border: '2px dashed var(--gray-200)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          cursor: 'pointer',
                          transition: 'all 0.15s ease',
                          color: 'var(--gray-400)',
                        }}
                        onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.borderColor = 'var(--develop-blue)'; (e.currentTarget as HTMLElement).style.color = 'var(--develop-blue)' }}
                        onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.borderColor = 'var(--gray-200)'; (e.currentTarget as HTMLElement).style.color = 'var(--gray-400)' }}
                      >
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
                      </div>
                    </>
                  )}
                </div>

                <div style={{ borderRadius: 8, overflow: 'hidden', border: '1px solid var(--gray-100)' }}>
                  <AnnotationCanvas
                    imageUrl={URL.createObjectURL(sampleImages[activeImageIndex])}
                    initialBoxes={activeBoxes}
                    onBoxesChange={(boxes) => setSampleImageBoxes(activeImageIndex, boxes)}
                  />
                </div>
                <p style={{ fontSize: 11, color: 'var(--gray-400)', marginTop: 8, textAlign: 'center' }}>可在图上框出关键目标，帮助系统理解外观、尺度和位置关系</p>
              </div>
            )}
          </div>
        </div>

        {/* Right: Description */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="card-section">
            <h2 style={{ fontSize: 15, fontWeight: 600, marginBottom: 12, color: 'var(--gray-900)' }}>业务需求描述</h2>
            <textarea
              className="input"
              value={userDescription}
              onChange={(e) => setUserDescription(e.target.value)}
              placeholder={'例如：\n"识别仓位是否被货箱占用，持续10秒后输出占位事件"\n"识别人员进入A区、离开A区，并在从A区进入B区时输出跨区事件"'}
              style={{ height: 180, fontFamily: 'var(--font-mono)', fontSize: 13, lineHeight: 1.6 }}
            />

            <div style={{ marginTop: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
                <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--gray-900)', margin: 0 }}>部署设备</h3>
                <span style={{ fontSize: 11, color: 'var(--gray-400)' }}>用于自动选择合适模型</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 8 }}>
                {(Object.keys(DEVICE_PROFILES) as DeviceProfileId[]).map((id) => {
                  const profile = DEVICE_PROFILES[id]
                  const active = deviceProfileId === id
                  return (
                    <div
                      key={id}
                      onClick={() => setDeviceProfileId(id)}
                      style={{
                        padding: '10px 12px',
                        borderRadius: 8,
                        cursor: 'pointer',
                        border: active ? '1px solid var(--develop-blue)' : '1px solid var(--gray-100)',
                        background: active ? 'rgba(10,114,239,0.04)' : 'var(--gray-50)',
                        transition: 'all 0.15s ease',
                      }}
                    >
                      <div style={{ fontSize: 12, fontWeight: 600, color: active ? 'var(--develop-blue)' : 'var(--gray-900)' }}>
                        {profile.label}
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--gray-500)', marginTop: 2, lineHeight: 1.5 }}>
                        {profile.description}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            <div style={{ marginTop: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
                <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--gray-900)', margin: 0 }}>待打标数据</h3>
                <span style={{ fontSize: 12, color: 'var(--gray-400)' }}>
                  {datasetImages.length > 0 && `${datasetImages.length} 张图片`}
                  {datasetImages.length > 0 && videoFile && ' + '}
                  {videoFile && `1 个视频`}
                </span>
              </div>
              <input
                ref={datasetFileInputRef}
                type="file"
                accept="image/jpeg,image/png,text/plain,.txt"
                multiple
                onChange={(e) => handleDatasetFilesSelected(e.target.files)}
                style={{ display: 'none' }}
              />
              <input
                ref={videoFileInputRef}
                type="file"
                accept="video/mp4,video/avi,video/mov,video/mkv,video/webm"
                onChange={(e) => handleVideoSelected(e.target.files)}
                style={{ display: 'none' }}
              />
              <div style={{ display: 'flex', gap: 10 }}>
                <div
                  onClick={() => datasetFileInputRef.current?.click()}
                  style={{
                    flex: 1,
                    padding: datasetImages.length > 0 ? '14px 16px' : '24px 16px',
                    borderRadius: 8,
                    border: '1px dashed var(--gray-200)',
                    background: 'var(--gray-50)',
                    cursor: 'pointer',
                  }}
                >
                  {datasetImages.length > 0 ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {datasetImages.slice(0, 5).map((file) => (
                        <div key={`${file.name}-${file.size}-${file.lastModified}`} style={{ fontSize: 12, color: 'var(--gray-600)' }}>
                          {file.name}
                        </div>
                      ))}
                      {datasetImages.length > 5 && (
                        <div style={{ fontSize: 12, color: 'var(--gray-400)' }}>还有 {datasetImages.length - 5} 张未展开</div>
                      )}
                    </div>
                  ) : (
                    <div style={{ textAlign: 'center' }}>
                      <p style={{ fontSize: 13, color: 'var(--gray-600)', margin: 0 }}>上传图片</p>
                      <p style={{ fontSize: 11, color: 'var(--gray-400)', margin: '6px 0 0' }}>JPG/PNG 格式</p>
                    </div>
                  )}
                </div>
                <div
                  onClick={() => videoFileInputRef.current?.click()}
                  style={{
                    flex: 1,
                    padding: videoFile ? '14px 16px' : '24px 16px',
                    borderRadius: 8,
                    border: videoFile ? '1px solid var(--develop-blue)' : '1px dashed var(--gray-200)',
                    background: videoFile ? 'rgba(10,114,239,0.04)' : 'var(--gray-50)',
                    cursor: 'pointer',
                  }}
                >
                  {videoFile ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--gray-900)' }}>{videoFile.name}</div>
                      <div style={{ fontSize: 11, color: 'var(--gray-400)' }}>{(videoFile.size / 1024 / 1024).toFixed(1)} MB</div>
                      {videoInfo && (
                        <div style={{ fontSize: 11, color: 'var(--develop-blue)' }}>
                          {videoInfo.duration_seconds}s · {videoInfo.width}×{videoInfo.height} · {videoInfo.frame_count ?? '?'} 帧
                        </div>
                      )}
                    </div>
                  ) : (
                    <div style={{ textAlign: 'center' }}>
                      <p style={{ fontSize: 13, color: 'var(--gray-600)', margin: 0 }}>上传视频</p>
                      <p style={{ fontSize: 11, color: 'var(--gray-400)', margin: '6px 0 0' }}>MP4/AVI/MOV 格式，自动拆帧</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {error && (
              <div style={{ marginTop: 12, padding: '10px 12px', background: '#fff5f5', border: '1px solid #fed7d7', borderRadius: 6, fontSize: 13, color: 'var(--ship-red)' }}>
                <strong>视觉解析未完成</strong>
                <div style={{ marginTop: 4 }}>{error}</div>
                <div style={{ marginTop: 6, color: 'var(--gray-600)' }}>
                  系统将先根据你的文字需求生成初步策略草案，你仍可在下一步继续确认。
                </div>
              </div>
            )}

            <button
              className="btn btn-primary w-full"
              onClick={handleParse}
              disabled={loading || videoUploading || (datasetImages.length === 0 && !videoFile) || !userDescription.trim()}
              style={{ marginTop: 16, justifyContent: 'center', height: 40, fontSize: 14, fontWeight: 600 }}
            >
              {loading ? (
                <>
                  <div className="spinner" />
                  理解中...
                </>
              ) : (
                <>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                  生成需求理解
                </>
              )}
            </button>
          </div>

          {/* Hint */}
          <div className="card-section" style={{ background: 'var(--gray-50)', boxShadow: 'none', border: '1px solid var(--gray-100)' }}>
            <div style={{ display: 'flex', gap: 10 }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--gray-400)" strokeWidth="2" style={{ flexShrink: 0, marginTop: 2 }}><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
              <div>
                <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--gray-600)', marginBottom: 2 }}>提示</p>
                <p style={{ fontSize: 12, color: 'var(--gray-500)', lineHeight: 1.5 }}>样板图是可选视觉参考。即使视觉解析失败，系统也会先基于你的文字需求生成策略草案；如果补充 2~3 个不同角度/大小的示例框，通常能让监测对象描述更完整。</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
