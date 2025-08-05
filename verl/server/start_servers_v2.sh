#!/bin/bash

: "${SERVER_HOST:?Please set SERVER_HOST in environment.sh}"
CRAWL_PAGE_PORT=9000
WEBSEARCH_PORT=9001

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$DIR/logs/$SERVER_HOST";   mkdir -p "$LOG_DIR"
PID_DIR="$DIR/pids/$SERVER_HOST";   mkdir -p "$PID_DIR"

cmd=$1
if [[ ! "$cmd" =~ ^(start|stop|status|test)$ ]]; then
  echo "Usage: $0 [start|stop|status|test]"
  echo "  start  : Start training mode servers (SerperV2, CrawlPageV2)"
  echo "  stop   : Stop all servers"
  echo "  status : Check server status"
  echo "  test   : Test server functionality"
  exit 1
fi

# =====================================================================================================================
#                                      start
# =====================================================================================================================

if [[ "$cmd" == "start" ]]; then
  echo "Starting training mode servers..."
  
  # CrawlPageV2
  pidf="$PID_DIR/${SERVER_HOST}_CrawlPageV2_$CRAWL_PAGE_PORT.pid"
  logf="$LOG_DIR/CrawlPageV2_$CRAWL_PAGE_PORT.log"
  if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "CrawlPageV2 is already running (PID $(cat "$pidf"))"
  else
    echo "Starting CrawlPageV2 on port $CRAWL_PAGE_PORT..."
    nohup python -u "$DIR/v2/crawl_page_server_v2.py" > "$logf" 2>&1 &
    echo $! > "$pidf"
  fi

  # SerperCacheV2
  pidf="$PID_DIR/${SERVER_HOST}_SerperCacheV2_$WEBSEARCH_PORT.pid"
  logf="$LOG_DIR/SerperCacheV2_$WEBSEARCH_PORT.log"
  if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "SerperCacheV2 is already running (PID $(cat "$pidf"))"
  else
    echo "Starting SerperCacheV2 on port $WEBSEARCH_PORT..."
    nohup python -u "$DIR/v2/cache_serper_server_v2_train.py" > "$logf" 2>&1 &
    echo $! > "$pidf"
  fi

# =====================================================================================================================
#                                      test
# =====================================================================================================================

elif [[ "$cmd" == "test" ]]; then
  echo "--------------------Starting test for serper cache v2 ------------------"
  python -u "$DIR/server_tests/test_cache_serper_server_v2.py"           "http://$SERVER_HOST:$WEBSEARCH_PORT/search"
  echo "-------------------------Test completed--------------------------"
  echo "--------------------Starting test for crawl page v2 -------------------"
  python -u "$DIR/server_tests/test_crawl_page_simple_v2.py"           "http://$SERVER_HOST:$CRAWL_PAGE_PORT/crawl_page"
  echo "-------------------------Test completed--------------------------"

# =====================================================================================================================
#                                      stop
# =====================================================================================================================

elif [[ "$cmd" == "stop" ]]; then
  echo "Stopping all servers..."
  
  # CrawlPageV2
  pidf="$PID_DIR/${SERVER_HOST}_CrawlPageV2_$CRAWL_PAGE_PORT.pid"
  stopped=0
  
  if [[ -f "$pidf" ]]; then
    pid=$(cat "$pidf" 2>/dev/null)
    if [[ -n "$pid" ]]; then
      if kill -0 "$pid" 2>/dev/null; then
        if kill "$pid" 2>/dev/null; then
          echo "Stopped CrawlPageV2 (PID $pid)"
          stopped=1
        else
          echo "Warning: Unable to stop CrawlPageV2 using PID $pid, trying other methods..."
        fi
      fi
    fi
    rm -f "$pidf"
  fi
  
  port_processes=($(lsof -t -i:"$CRAWL_PAGE_PORT" 2>/dev/null))
  if [[ ${#port_processes[@]} -gt 0 ]]; then
    for p in "${port_processes[@]}"; do
      if kill -0 "$p" 2>/dev/null; then
        if kill "$p" 2>/dev/null; then
          echo "Stopped CrawlPageV2 by port (PID $p)"
          stopped=1
        else
          echo "Warning: Unable to stop process $p on port $CRAWL_PAGE_PORT"
        fi
      fi
    done
  fi
  
  if [[ $stopped -eq 0 ]]; then
    echo "CrawlPageV2 is not running, and port $CRAWL_PAGE_PORT is not in use"
  fi

  # SerperCacheV2
  pidf="$PID_DIR/${SERVER_HOST}_SerperCacheV2_$WEBSEARCH_PORT.pid"
  stopped=0
  
  if [[ -f "$pidf" ]]; then
    pid=$(cat "$pidf" 2>/dev/null)
    if [[ -n "$pid" ]]; then
      if kill -0 "$pid" 2>/dev/null; then
        if kill "$pid" 2>/dev/null; then
          echo "Stopped SerperCacheV2 (PID $pid)"
          stopped=1
        else
          echo "Warning: Unable to stop SerperCacheV2 using PID $pid, trying other methods..."
        fi
      fi
    fi
    rm -f "$pidf"
  fi
  
  port_processes=($(lsof -t -i:"$WEBSEARCH_PORT" 2>/dev/null))
  if [[ ${#port_processes[@]} -gt 0 ]]; then
    for p in "${port_processes[@]}"; do
      if kill -0 "$p" 2>/dev/null; then
        if kill "$p" 2>/dev/null; then
          echo "Stopped SerperCacheV2 by port (PID $p)"
          stopped=1
        else
          echo "Warning: Unable to stop process $p on port $WEBSEARCH_PORT"
        fi
      fi
    done
  fi
  
  if [[ $stopped -eq 0 ]]; then
    echo "SerperCacheV2 is not running, and port $WEBSEARCH_PORT is not in use"
  fi

# =====================================================================================================================
#                                      status
# =====================================================================================================================

else
  # CrawlPageV2
  pidf="$PID_DIR/${SERVER_HOST}_CrawlPageV2_$CRAWL_PAGE_PORT.pid"
  if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "CrawlPageV2 is running (PID $(cat "$pidf"))"
  elif lsof -i:"$CRAWL_PAGE_PORT" &>/dev/null; then
    echo "CrawlPageV2 port $CRAWL_PAGE_PORT is in use, but PID file is invalid"
  else
    echo "CrawlPageV2 is not running, and port $CRAWL_PAGE_PORT is not in use"
  fi

  # SerperCacheV2
  pidf="$PID_DIR/${SERVER_HOST}_SerperCacheV2_$WEBSEARCH_PORT.pid"
  if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "SerperCacheV2 is running (PID $(cat "$pidf"))"
  elif lsof -i:"$WEBSEARCH_PORT" &>/dev/null; then
    echo "SerperCacheV2 port $WEBSEARCH_PORT is in use, but PID file is invalid"
  else
    echo "SerperCacheV2 is not running, and port $WEBSEARCH_PORT is not in use"
  fi
fi