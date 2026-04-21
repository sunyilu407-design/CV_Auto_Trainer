# Single Machine Production Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把系统收口成 macOS / Windows 单机正式部署版本，前端由后端托管，云训练状态持久化，训练完成后可直接下载真实产物和算法 bundle。

**Architecture:** 继续保留前后端与 Worker 的现有单机结构，但去掉开发模式依赖。后端托管前端静态产物，Worker 地址和 CORS 改为配置化，云训练状态写入数据库而不是内存，交付页改为读取真实产物清单。

**Tech Stack:** FastAPI, SQLAlchemy, React 18, Zustand, Vite, Python integration tests, PostgreSQL-compatible configuration

---

### Task 1: 用失败测试锁定生产托管与训练状态持久化

**Files:**
- Modify: `tests/test_integration.py`
- Test: `tests/test_integration.py`

- [ ] **Step 1: Write a failing backend integration test for persisted cloud training state**

```python
def test_training_status_persistence():
    print("\n=== 测试：训练状态持久化 ===")
    backend_dir = PROJECT_ROOT / "backend"
    old_cwd = Path.cwd()
    try:
        os.chdir(backend_dir)
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["CV_AUTO_TRAINER_DB_URL"] = f"sqlite:///{Path(tmpdir) / 'test_training_state.db'}"
            ensure_test_auth_env()

            backend_path = str(backend_dir)
            if backend_path in sys.path:
                sys.path.remove(backend_path)
            sys.path.insert(0, backend_path)

            from fastapi.testclient import TestClient

            for name in list(sys.modules):
                if name == "main" or name.startswith(("routers", "models", "services")):
                    sys.modules.pop(name, None)
            main = importlib.import_module("main")

            with patch("services.train_dispatcher.TrainDispatcher.dispatch", return_value={"best.pt": "/tmp/mock/best.pt"}):
                with TestClient(main.app) as client:
                    login_resp = client.post("/api/auth/login", json={"username": TEST_ADMIN_USERNAME, "password": TEST_ADMIN_PASSWORD})
                    token = login_resp.json()["data"]["token"]
                    headers = auth_headers(token)
                    task_id = client.post("/api/tasks", json={"name": "云训练任务"}, headers=headers).json()["id"]

                    start_resp = client.post(
                        "/api/training/start",
                        json={
                            "task_id": task_id,
                            "model": "yolo11s.pt",
                            "epochs": 10,
                            "imgsz": 640,
                            "lr0": 0.01,
                            "patience": 5,
                            "conf": 0.25,
                            "iou": 0.7,
                            "export_formats": ["onnx"],
                            "train_mode": "cloud",
                            "gpu_type": "RTX 4090",
                            "resume_last": False,
                        },
                        headers=headers,
                    )
                    if start_resp.status_code != 200:
                        print(f"✗ 启动云训练失败: {start_resp.status_code} {start_resp.json()}")
                        return False

                    deadline = time.time() + 5
                    status_payload = None
                    while time.time() < deadline:
                        status_resp = client.get(f"/api/training/{task_id}/status", headers=headers)
                        status_payload = status_resp.json()["data"]
                        if status_payload["state"] == "done":
                            break
                        time.sleep(0.1)

                    if not status_payload or status_payload["state"] != "done":
                        print(f"✗ 云训练状态未持久化为 done: {status_payload}")
                        return False
                    if status_payload.get("artifact_paths", {}).get("best.pt") != "/tmp/mock/best.pt":
                        print(f"✗ artifact_paths 未返回: {status_payload}")
                        return False

        print("✓ 训练状态持久化正常")
        return True
    except Exception as e:
        print(f"✗ 训练状态持久化测试异常: {e}")
        return False
    finally:
        os.environ.pop("CV_AUTO_TRAINER_DB_URL", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_USERNAME", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_PASSWORD", None)
        os.environ.pop("CV_AUTO_TRAINER_SECRET_KEY", None)
        os.chdir(old_cwd)
```

- [ ] **Step 2: Run the new training persistence test and verify it fails**

Run: `./.venv/bin/python tests/test_integration.py training-status-persistence`
Expected: FAIL because training status is still in-memory and does not expose persisted artifact paths

- [ ] **Step 3: Write a failing backend test for production static hosting**

```python
def test_backend_serves_frontend_dist():
    print("\n=== 测试：后端生产静态托管 ===")
    backend_dir = PROJECT_ROOT / "backend"
    old_cwd = Path.cwd()
    try:
        os.chdir(backend_dir)
        with tempfile.TemporaryDirectory() as tmpdir:
            dist_dir = Path(tmpdir) / "dist"
            dist_dir.mkdir()
            (dist_dir / "index.html").write_text("<html><body>cv-auto-trainer-app</body></html>", encoding="utf-8")
            os.environ["CV_AUTO_TRAINER_FRONTEND_DIST"] = str(dist_dir)
            ensure_test_auth_env()

            backend_path = str(backend_dir)
            if backend_path in sys.path:
                sys.path.remove(backend_path)
            sys.path.insert(0, backend_path)

            from fastapi.testclient import TestClient

            for name in list(sys.modules):
                if name == "main" or name.startswith(("routers", "models", "services")):
                    sys.modules.pop(name, None)
            main = importlib.import_module("main")

            with TestClient(main.app) as client:
                resp = client.get("/app")
                if resp.status_code != 200 or "cv-auto-trainer-app" not in resp.text:
                    print(f"✗ 后端未返回前端 dist 内容: {resp.status_code} {resp.text}")
                    return False

        print("✓ 后端生产静态托管正常")
        return True
    except Exception as e:
        print(f"✗ 后端生产静态托管测试异常: {e}")
        return False
    finally:
        os.environ.pop("CV_AUTO_TRAINER_FRONTEND_DIST", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_USERNAME", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_PASSWORD", None)
        os.environ.pop("CV_AUTO_TRAINER_SECRET_KEY", None)
        os.chdir(old_cwd)
```

- [ ] **Step 4: Run the static hosting test and verify it fails**

Run: `./.venv/bin/python tests/test_integration.py backend-frontend-dist`
Expected: FAIL because backend does not yet serve `frontend/dist`

### Task 2: 实现训练状态持久化与云训练闭环

**Files:**
- Modify: `backend/models/db.py`
- Modify: `backend/models/database.py`
- Modify: `backend/routers/training.py`
- Modify: `backend/services/train_dispatcher.py`
- Test: `tests/test_integration.py`

- [ ] **Step 1: Add persisted training state fields to `Task`**

```python
    training_state = Column(String, default="idle")
    training_progress = Column(JSON)
    training_started_at = Column(DateTime)
    training_finished_at = Column(DateTime)
```

- [ ] **Step 2: Extend lightweight migration logic for new columns**

```python
    required_columns = {
        "algorithm_plan": "ALTER TABLE tasks ADD COLUMN algorithm_plan JSON",
        "algorithm_plan_status": "ALTER TABLE tasks ADD COLUMN algorithm_plan_status VARCHAR",
        "pipeline_config": "ALTER TABLE tasks ADD COLUMN pipeline_config JSON",
        "owner_user_id": "ALTER TABLE tasks ADD COLUMN owner_user_id INTEGER",
        "training_state": "ALTER TABLE tasks ADD COLUMN training_state VARCHAR",
        "training_progress": "ALTER TABLE tasks ADD COLUMN training_progress JSON",
        "training_started_at": "ALTER TABLE tasks ADD COLUMN training_started_at DATETIME",
        "training_finished_at": "ALTER TABLE tasks ADD COLUMN training_finished_at DATETIME",
    }
```

- [ ] **Step 3: Replace in-memory training state writes with DB-backed task state**

```python
    task = get_task_for_user(db, payload.task_id, current_user)
    task.train_config = payload.model_dump()
    task.training_state = "queued"
    task.training_progress = {
        "current_epoch": 0,
        "total_epochs": payload.epochs,
        "current_map": 0.0,
        "done": False,
    }
    task.training_started_at = datetime.utcnow()
    task.training_finished_at = None
    db.commit()
```

- [ ] **Step 4: Update dispatcher progress callbacks and final result writes**

```python
        def progress_callback(status: dict):
            session = Session.object_session(task) or self.db
            task.training_state = "running"
            task.training_progress = status
            session.commit()

        ...
            task.status = "done"
            task.training_state = "done"
            task.training_progress = {
                **(train_config.get("progress_snapshot") or {}),
                "done": True,
            }
            task.training_finished_at = datetime.utcnow()
            task.artifact_paths = result
```

- [ ] **Step 5: Fix cloud settings lookup to use task owner user id**

```python
        settings = get_settings(self.db, task.owner_user_id)
```

- [ ] **Step 6: Run persisted training tests until they pass**

Run: `./.venv/bin/python tests/test_integration.py training-status-persistence`
Expected: PASS

### Task 3: 实现前端生产托管与部署配置化

**Files:**
- Modify: `backend/main.py`
- Modify: `frontend/vite.config.ts`
- Modify: `frontend/src/api/worker.ts`
- Test: `tests/test_integration.py`

- [ ] **Step 1: Add backend static hosting for configured dist directory**

```python
frontend_dist = os.getenv("CV_AUTO_TRAINER_FRONTEND_DIST")
if frontend_dist:
    dist_path = Path(frontend_dist)
    if dist_path.exists():
        app.mount("/assets", StaticFiles(directory=dist_path / "assets"), name="frontend-assets")

        @app.get("/{full_path:path}")
        def frontend_app(full_path: str):
            if full_path.startswith("api"):
                raise HTTPException(status_code=404, detail="Not found")
            index_path = dist_path / "index.html"
            if index_path.exists():
                return FileResponse(index_path)
            raise HTTPException(status_code=404, detail="Frontend dist not found")
```

- [ ] **Step 2: Add Vite env support for worker websocket URL**

```typescript
const workerTarget = process.env.CV_AUTO_TRAINER_WORKER_URL ?? 'ws://127.0.0.1:7860/ws'
```

- [ ] **Step 3: Switch frontend worker client to env-based URL**

```typescript
const WS_BASE =
  (import.meta.env.VITE_WORKER_WS_URL as string | undefined)?.trim() ||
  `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.hostname}:7860/ws`
```

- [ ] **Step 4: Run static hosting test until it passes**

Run: `./.venv/bin/python tests/test_integration.py backend-frontend-dist`
Expected: PASS

### Task 4: 收口前端云训练监控与交付产物展示

**Files:**
- Modify: `frontend/src/store/taskStore.ts`
- Modify: `frontend/src/pages/TrainConfig.tsx`
- Modify: `frontend/src/pages/TrainingMonitor.tsx`
- Modify: `frontend/src/pages/Delivery.tsx`
- Modify: `frontend/src/api/backend.ts`
- Modify: `backend/routers/files.py`
- Test: `tests/test_integration.py`

- [ ] **Step 1: Add artifact and training state hydration support to the task store**

```typescript
  setArtifacts: (artifacts: Record<string, string>) => void
```

- [ ] **Step 2: Ensure cloud training monitor uses the real `taskId`**

```typescript
const { taskId, trainingProgress, trainConfig, setStage, setTrainingProgress, setArtifacts } = useTaskStore()
...
const status = await trainingApi.getStatus(taskId)
...
if (status.artifact_paths) {
  setArtifacts(status.artifact_paths)
}
```

- [ ] **Step 3: Make delivery page load real artifact list instead of fixed assumptions only**

```typescript
const [artifactList, setArtifactList] = useState<Array<{ name: string; path: string; size: number }>>([])
useEffect(() => {
  if (!taskId) return
  filesApi.getArtifacts(taskId).then(setArtifactList).catch(() => {})
}, [taskId, packageReady])
```

- [ ] **Step 4: Make file listing prefer task artifact paths and backend artifact directory**

```python
    files = []
    task = get_task_for_user(db, task_id, current_user)
    for name, path in (task.artifact_paths or {}).items():
        if path and Path(path).exists():
            files.append({"name": name, "path": str(path), "size": Path(path).stat().st_size})
```

- [ ] **Step 5: Run backend regression after UI-contract and artifact listing changes**

Run: `./.venv/bin/python tests/test_integration.py backend`
Expected: PASS

### Task 5: 补 macOS / Windows 单机部署文档与脚本

**Files:**
- Modify: `README.md`
- Modify: `MAC.md`
- Create: `WINDOWS.md`
- Create: `scripts/start_backend_macos.sh`
- Create: `scripts/start_worker_macos.sh`
- Create: `scripts/start_backend_windows.ps1`
- Create: `scripts/start_worker_windows.ps1`

- [ ] **Step 1: Update root deployment docs for production single-machine mode**

```markdown
## 正式部署（单机）

1. 配置 PostgreSQL 与环境变量
2. 构建前端 `npm run build`
3. 设置 `CV_AUTO_TRAINER_FRONTEND_DIST`
4. 启动 backend
5. 启动 worker
6. 访问后端托管的前端页面
```

- [ ] **Step 2: Add platform-specific startup scripts**

```bash
#!/usr/bin/env bash
export CV_AUTO_TRAINER_DB_URL=postgresql://...
export CV_AUTO_TRAINER_FRONTEND_DIST=...
export CV_AUTO_TRAINER_SECRET_KEY=...
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

- [ ] **Step 3: Run the full integration suite**

Run: `./.venv/bin/python tests/test_integration.py full`
Expected: PASS
