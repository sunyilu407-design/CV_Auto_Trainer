import sys
from pathlib import Path
from typing import Dict, Optional


def export_task_algorithm_package(
    task_id: str,
    pipeline_config: Dict,
    artifacts: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    project_root = Path(__file__).resolve().parents[2]
    worker_path = project_root / "worker"
    if str(worker_path) not in sys.path:
        sys.path.insert(0, str(worker_path))

    from pipeline.package_exporter import export_algorithm_package

    output_dir = project_root / "backend" / "artifacts" / task_id
    return export_algorithm_package(
        task_id=task_id,
        pipeline_config=pipeline_config,
        artifacts=artifacts or {},
        output_dir=output_dir,
    )
