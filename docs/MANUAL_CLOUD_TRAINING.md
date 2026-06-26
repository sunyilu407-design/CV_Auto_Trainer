# Manual Cloud Training Guide

Use this guide when the app cannot connect to the rented cloud GPU machine directly, but the user still wants to train without wasting rental time.

## What the system already gives you

- the prepared YOLO dataset directory under `backend/uploads/<task_id>/dataset`
- cloud-side helper scripts under `cloud_scripts/`
- a local helper that packages both into a ready-to-upload bundle

## Step 1: Generate the manual training bundle locally

From the repo root:

```bash
python scripts/prepare_manual_cloud_training.py \
  --dataset-dir backend/uploads/<task_id>/dataset \
  --output-dir /tmp/cv_manual_training \
  --model yolo11s.pt \
  --epochs 100 \
  --imgsz 640 \
  --export-format onnx
```

This generates:

- `manual_cloud_training/dataset.zip`
- `manual_cloud_training/cloud_scripts/*.py`
- `manual_cloud_training/README.md`

## Step 2: Upload to the cloud GPU machine

Example:

```bash
scp /tmp/cv_manual_training/manual_cloud_training/dataset.zip root@<host>:/root/workspace/
scp -r /tmp/cv_manual_training/manual_cloud_training/cloud_scripts root@<host>:/root/workspace/
```

## Step 3: Start training manually on the cloud machine

```bash
ssh root@<host>
cd /root/workspace
unzip -q dataset.zip -d dataset
screen -dmS train bash -c "cd /root/workspace && python cloud_scripts/train.py --data /root/workspace/dataset/data.yaml --model yolo11s.pt --epochs 100 --imgsz 640 --lr0 0.01 --patience 20 --project /root/workspace/training_output"
```

## Step 4: Check progress

```bash
cd /root/workspace
python cloud_scripts/health_check.py
tail -f /root/workspace/training_output/exp/results.csv
```

## Step 5: Export deployment formats

After training completes:

```bash
cd /root/workspace
python cloud_scripts/export.py --weights /root/workspace/training_output/exp/weights/best.pt --format onnx
```

## Step 6: Download results back

```bash
scp root@<host>:/root/workspace/training_output/exp/weights/best.pt ./best.pt
scp root@<host>:/root/workspace/training_output/exp/results.csv ./results.csv
scp root@<host>:/root/workspace/training_output/exp/weights/best.onnx ./best.onnx
```

## Recommended operating rule

Before renting a cloud machine for the user:

1. first make sure the local dataset is ready
2. generate the manual bundle locally
3. verify you can upload with `scp` and log in with `ssh`
4. only then let the user start the paid cloud instance

This avoids wasting rental time on local preparation or connection debugging.
