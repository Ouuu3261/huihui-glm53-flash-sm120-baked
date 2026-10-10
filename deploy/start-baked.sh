#!/usr/bin/env bash
set -euo pipefail
export STATE_DIR=${STATE_DIR:-/workspace/huihui-glm53-tp2}
mkdir -p "$STATE_DIR"
exec 9>"$STATE_DIR/bootstrap.lock"
flock -n 9 || { echo 'Deployment already running'; exit 0; }
export BOOTSTRAP_LOCK_HELD=1
for name in bootstrap-vast.sh serve-vast.sh apply_runtime_fixes.py select_communication.py verify_baked_model.py; do
  cp "/opt/huihui-deploy/$name" "$STATE_DIR/$name"
done
exec bash "$STATE_DIR/bootstrap-vast.sh"
