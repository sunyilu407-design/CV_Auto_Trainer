# Windows 单机部署指南

本指南面向正式单机使用：

- 本地做上传、画框、质检、增强
- 系统内部发起云端训练
- 训练完成后直接下载模型和算法工程包

## 1. 安装基础环境

- Python 3.11+
- Node.js 18+
- PostgreSQL 15+

## 2. 创建数据库

使用 PostgreSQL 客户端或 pgAdmin 创建：

```text
database: cv_auto_trainer
```

## 3. 构建前端

```powershell
cd frontend
npm install
npm run build
```

## 4. 设置环境变量

PowerShell 示例：

```powershell
$env:CV_AUTO_TRAINER_DB_URL = "postgresql://postgres:postgres@127.0.0.1:5432/cv_auto_trainer"
$env:CV_AUTO_TRAINER_SECRET_KEY = "replace-this-with-a-stable-secret"
$env:CV_AUTO_TRAINER_ADMIN_USERNAME = "admin"
$env:CV_AUTO_TRAINER_ADMIN_PASSWORD = "change-me"
$env:CV_AUTO_TRAINER_FRONTEND_DIST = (Resolve-Path ".\frontend\dist").Path
$env:CV_AUTO_TRAINER_CORS_ORIGINS = "http://127.0.0.1:8000,http://localhost:8000"
```

## 5. 启动后端与 Worker

```powershell
.\scripts\start_backend_windows.ps1
.\scripts\start_worker_windows.ps1
```

访问：

```text
http://127.0.0.1:8000/
```

## 6. 常驻建议

- 后端：NSSM 托管 `powershell.exe -File scripts\start_backend_windows.ps1`
- Worker：NSSM 托管 `powershell.exe -File scripts\start_worker_windows.ps1`
- 如不使用 NSSM，可用“任务计划程序”在登录后自动启动
