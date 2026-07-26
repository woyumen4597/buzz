#!/usr/bin/env bash
# Buzz 启动脚本:走 PyPI 清华镜像、关遥测、后台运行、日志同步落盘
# 用法: ./start.sh

set -uo pipefail
cd "$(dirname "$0")"

# --- 环境:PyPI 清华镜像 + 关遥测 ---
export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export BUZZ_DISABLE_TELEMETRY=1

# --- 不走代理:清掉所有 *_proxy 变量,避免 OpenAI httpx 触发 SOCKS 闪退 ---
# (系统常被 V2Ray/Clash 设了 ALL_PROXY=socks5://...,Buzz 不需要联网)
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY \
      http_proxy https_proxy all_proxy \
      NO_PROXY no_proxy

# --- 日志:stdout/stderr(含闪退 traceback)全落盘 ---
LOG_DIR="$HOME/Library/Logs/Buzz"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run.log"
: > "$RUN_LOG"   # 清空,看本次干净的输出

PID_FILE="$LOG_DIR/buzz.pid"

# 已有进程在跑就免重复起
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Buzz already running, pid=$(cat "$PID_FILE")"
  exit 0
fi

# nohup 后台启动,所有输出重定向到 run.log
nohup uv run python main.py >>"$RUN_LOG" 2>&1 &
APP_PID=$!
echo "$APP_PID" >"$PID_FILE"

# 彩色打印日志位置(终端能渲染 ANSI;存到文件也无伤大雅)
GREEN=$'\033[1;32m'
CYAN=$'\033[1;36m'
YEL=$'\033[1;33m'
BOLD=$'\033[1m'
RST=$'\033[0m'

echo
echo "${GREEN}Buzz started${RST} ${BOLD}pid=$APP_PID${RST}"
echo "${YEL}━━━━━━ 日志位置 ━━━━━━${RST}"
echo "  ${CYAN}${RUN_LOG}${RST}"
echo "  ${CYAN}${LOG_DIR}/logs.txt${RST}    ${BOLD}(Buzz 自带,GUI 运行期 DEBUG)${RST}"
echo "${YEL}━━━━━━━━━━━━━━━━━━━${RST}"
echo
echo "实时跟看: ${BOLD}tail -f \"$RUN_LOG\"${RST}"
echo "查进程  : ${BOLD}cat \"$PID_FILE\"${RST}"
echo "停掉    : ${BOLD}kill \"\$(cat \"$PID_FILE\")\"${RST}"
