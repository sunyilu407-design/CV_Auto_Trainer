# Intent Understanding Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Stage 1 continue when VLM visual parsing fails, and shift Stage 1/Stage 2 copy from detector-training language to requirement-understanding and strategy-generation language.

**Architecture:** Normalize VLM parse failures into a structured response in the backend, store explicit VLM status in the frontend, and let the UI render either visual candidates or a fallback requirement summary. Keep algorithm planning on the same API and data path, but allow it to run with `vlm_result: null`.

**Tech Stack:** FastAPI, Pydantic, jsonschema, React, TypeScript, Zustand, Vite, Python integration tests.

---

### Task 1: Add Backend Fallback Parse Contract

**Files:**
- Modify: `backend/services/vlm_adapter.py`
- Modify: `backend/routers/vlm.py`
- Test: `tests/test_integration.py`

- [ ] **Step 1: Write the failing backend integration test**

```python
def test_vlm_parse_returns_structured_fallback_on_invalid_json():
    print("\n=== 测试：VLM 解析失败返回降级结构 ===")
    backend_dir = PROJECT_ROOT / "backend"
    old_cwd = Path.cwd()
    try:
        os.chdir(backend_dir)
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["CV_AUTO_TRAINER_DB_URL"] = f"sqlite:///{Path(tmpdir) / 'test_vlm_fallback.db'}"
            ensure_test_auth_env()

            backend_path = str(backend_dir)
            if backend_path not in sys.path:
                sys.path.insert(0, backend_path)

            from fastapi.testclient import TestClient

            for name in list(sys.modules):
                if name == "main" or name.startswith(("routers", "models", "services")):
                    sys.modules.pop(name, None)
            main = importlib.import_module("main")
            routers_vlm = importlib.import_module("routers.vlm")

            class FakeAdapter:
                def __init__(self, *args, **kwargs):
                    pass

                def parse_intent(self, images_base64, user_text, sample_boxes=None):
                    return {
                        "status": "failed",
                        "message": "VLM 已响应，但返回格式不符合系统要求，当前将先根据文字需求生成草案",
                        "retryable": True,
                        "raw_vlm_response": "<html>invalid</html>",
                    }

            with patch.object(routers_vlm, "VLMAdapter", FakeAdapter):
                with TestClient(main.app) as client:
                    login_resp = client.post(
                        "/api/auth/login",
                        json={"username": TEST_ADMIN_USERNAME, "password": TEST_ADMIN_PASSWORD},
                    )
                    headers = auth_headers(login_resp.json()["data"]["token"])
                    parse_resp = client.post(
                        "/api/vlm/parse",
                        json={
                            "images_base64": ["ZmFrZS1pbWFnZQ=="],
                            "user_text": "识别仓位是否被货箱占用，持续10秒后输出占位事件",
                            "sample_boxes": [],
                        },
                        headers=headers,
                    )
                    payload = parse_resp.json()

                    if parse_resp.status_code != 200 or payload.get("code") != 0:
                        print(f"✗ 解析接口未返回成功包裹: {parse_resp.status_code} {payload}")
                        return False

                    data = payload.get("data") or {}
                    if data.get("status") != "failed":
                        print(f"✗ 未返回 failed 状态: {data}")
                        return False
                    if not data.get("retryable"):
                        print(f"✗ retryable 未按预期返回: {data}")
                        return False
                    if "根据文字需求生成草案" not in (data.get("message") or ""):
                        print(f"✗ 失败提示不够可操作: {data}")
                        return False

        print("✓ VLM 解析失败降级结构正常")
        return True
    except Exception as e:
        print(f"✗ VLM 解析失败降级测试异常: {e}")
        return False
    finally:
        os.environ.pop("CV_AUTO_TRAINER_DB_URL", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_USERNAME", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_PASSWORD", None)
        os.environ.pop("CV_AUTO_TRAINER_SECRET_KEY", None)
        os.chdir(old_cwd)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python tests/test_integration.py vlm-parse-fallback`
Expected: FAIL because `/api/vlm/parse` still raises an HTTP 500 or returns no structured `status/message/retryable` payload on VLM parse failure.

- [ ] **Step 3: Implement normalized fallback mapping in `backend/services/vlm_adapter.py`**

```python
def parse_intent(... ) -> dict:
    last_err = None
    last_raw = ""
    for attempt in range(max_retry):
        try:
            raw = self._call_api(images_base64, user_text, sample_boxes)
            last_raw = raw
            result = self._parse_and_validate(raw)
            result["status"] = "success"
            result["message"] = ""
            result["retryable"] = False
            return result
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_err = e
            continue
        except httpx.HTTPError as e:
            return self._build_failed_result(
                message="VLM 服务暂时无法完成视觉理解，当前将先根据文字需求生成草案",
                retryable=True,
                raw_vlm_response=last_raw,
            )
    return self._build_failed_result_from_error(last_err, last_raw)
```

- [ ] **Step 4: Return fallback payload from `backend/routers/vlm.py` instead of raising**

```python
result = adapter.parse_intent(
    images_base64=payload.images_base64,
    user_text=payload.user_text,
    sample_boxes=payload.sample_boxes or [],
)
return {"code": 0, "msg": "ok", "data": result}
```

- [ ] **Step 5: Register the new integration test**

```python
tests = {
    ...
    "vlm-parse-fallback": test_vlm_parse_returns_structured_fallback_on_invalid_json,
}
groups["backend"] = [
    ...
    "vlm-parse-fallback",
]
```

- [ ] **Step 6: Run backend tests to verify they pass**

Run: `./.venv/bin/python tests/test_integration.py vlm-parse-fallback vlm-parse-settings vlm-display-fallbacks`
Expected: PASS for all selected tests.


### Task 2: Add Frontend Fallback State And Algorithm Plan Compatibility

**Files:**
- Modify: `frontend/src/api/backend.ts`
- Modify: `frontend/src/store/taskStore.ts`
- Modify: `frontend/src/pages/Upload.tsx`
- Modify: `frontend/src/pages/AlgorithmPlan.tsx`
- Test: `frontend/src/pages/Upload.tsx`

- [ ] **Step 1: Add parse result types in `frontend/src/api/backend.ts`**

```ts
export interface VLMParseSuccess {
  status: 'success'
  message: string
  retryable: boolean
  raw_vlm_response?: string
  classes: unknown[]
  confidence?: number | null
}

export interface VLMParseFailure {
  status: 'failed'
  message: string
  retryable: boolean
  raw_vlm_response?: string
  classes?: unknown[]
  confidence?: number | null
}

export type VLMParseResult = VLMParseSuccess | VLMParseFailure
```

- [ ] **Step 2: Update `vlmApi.parse()` and `algorithmApi.generatePlan()` signatures**

```ts
parse: (...) => request<VLMParseResult>('/vlm/parse', { ... })

generatePlan: (payload: {
  task_id: string
  user_description: string
  vlm_result: { classes: unknown[] } | null
  runtime_capability?: RuntimeCapability
}) => request<AlgorithmPlanRecord>('/algorithm/plan', { ... })
```

- [ ] **Step 3: Extend task store state for fallback mode**

```ts
export type VLMStatus = 'idle' | 'success' | 'failed'

vlmStatus: VLMStatus
vlmErrorMessage: string | null
vlmFallbackMode: boolean

setVLMStatus: (status: VLMStatus, message?: string | null) => void
```

- [ ] **Step 4: Update `Upload.tsx` to continue on fallback**

```ts
const result = await vlmApi.parse(imagesBase64, userDescription, allSampleBoxes)
await ensureTaskAndUploadDataset()

if (result.status === 'success') {
  setVLMResult({
    classes: normalizedClasses,
    raw_vlm_response: result.raw_vlm_response ?? '',
    confidence: typeof result.confidence === 'number' ? result.confidence : null,
  })
  setVLMStatus('success', null)
} else {
  setVLMResult(null)
  setVLMStatus('failed', result.message)
}

setAlgorithmPlan(null)
setStage('intent_confirm')
```

- [ ] **Step 5: Update `AlgorithmPlan.tsx` to treat text-only context as valid**

```ts
if (!taskId) {
  return <FallbackState />
}

const generated = await algorithmApi.generatePlan({
  task_id: taskId,
  user_description: userDescription,
  vlm_result: vlmResult ? { classes: vlmResult.classes } : null,
  runtime_capability: runtimeCapability,
})
```

- [ ] **Step 6: Run frontend build to verify type compatibility**

Run: `cd frontend && npm run build`
Expected: PASS with no TypeScript errors about `VLMParseResult`, store fields, or nullable `vlm_result`.


### Task 3: Rewrite Stage 1 / Stage 2 Copy And Fallback UI

**Files:**
- Modify: `frontend/src/pages/Upload.tsx`
- Modify: `frontend/src/pages/IntentConfirm.tsx`
- Modify: `frontend/src/pages/AlgorithmPlan.tsx`
- Test: `frontend/src/pages/IntentConfirm.tsx`

- [ ] **Step 1: Rewrite Stage 1 copy in `frontend/src/pages/Upload.tsx`**

```tsx
<h1 className="page-title">输入业务需求</h1>
<p className="page-subtitle">
  系统会先理解你的业务需求，再结合样板图补充视觉细节，生成策略草案
</p>

<textarea
  placeholder={'例如：\n"识别仓位是否被货箱占用，持续10秒后输出占位事件"\n"识别人员进入A区、离开A区，并在从A区进入B区时输出跨区事件"'}
/>
```

- [ ] **Step 2: Show fallback warning instead of hard error in `Upload.tsx`**

```tsx
{error && (
  <div ...>
    <strong>视觉解析未完成</strong>
    <div style={{ marginTop: 4 }}>{error}</div>
    <div style={{ marginTop: 6 }}>
      系统将先根据你的文字需求生成初步策略草案，你仍可在下一步继续确认。
    </div>
  </div>
)}
```

- [ ] **Step 3: Replace “确认检测意图” with “确认需求理解” in `IntentConfirm.tsx`**

```tsx
const { vlmResult, vlmStatus, vlmErrorMessage, userDescription, algorithmPlan, setStage } = useTaskStore()

<h1 className="page-title">确认需求理解</h1>
<p className="page-subtitle">
  系统正在整理目标、区域、事件和视觉线索，并将其扩展为策略与算法草案
</p>
```

- [ ] **Step 4: Add dual-mode body in `IntentConfirm.tsx`**

```tsx
{vlmStatus === 'failed' ? (
  <FallbackSummary
    userDescription={userDescription}
    errorMessage={vlmErrorMessage}
    scenarioLabel={algorithmPlan?.algorithm_plan.scenario_type ?? '待生成'}
  />
) : (
  <VisualCandidatesEditor classes={vlmResult?.classes ?? []} />
)}
```

- [ ] **Step 5: Add text-only hint in `AlgorithmPlan.tsx`**

```tsx
{vlmStatus === 'failed' && (
  <div className="card-section" ...>
    当前草案主要基于文字需求生成，后续仍可补充视觉参考以完善监测对象细节。
  </div>
)}
```

- [ ] **Step 6: Run final verification**

Run: `./.venv/bin/python tests/test_integration.py vlm-parse-fallback vlm-parse-settings vlm-display-fallbacks`
Expected: PASS

Run: `cd frontend && npm run build`
Expected: PASS


## Self-Review

- Spec coverage: Task 1 covers structured backend fallback and normalized messaging; Task 2 covers explicit frontend fallback state and nullable `vlm_result`; Task 3 covers copy changes, fallback messaging, and algorithm plan guidance.
- Placeholder scan: No TODO/TBD markers or “implement later” placeholders remain in tasks.
- Type consistency: The plan uses one parse result shape (`VLMParseResult`), one status enum (`VLMStatus`), and one nullable algorithm payload (`vlm_result: { classes: unknown[] } | null`) across all tasks.
