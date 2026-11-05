#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Usage: shoot_app.sh APP_DIR APP_FILE PORT  then pairs of: OUT.png "click-text" WIDTH HEIGHT
dir=$1; app=$2; port=$3; shift 3
cd "$dir"
streamlit run "$app" --server.headless true --server.port $port --client.toolbarMode viewer --browser.gatherUsageStats false > /tmp/st_$port.log 2>&1 &
pid=$!
timeout 300 bash -c "until curl -s localhost:$port >/dev/null; do sleep 2; done"
while [ $# -ge 4 ]; do
  python3 "$ROOT/scripts/screenshot.py" http://localhost:$port "$ROOT/figures/$1" "$2" $3 $4
  echo "saved $1"; shift 4
done
kill $pid
