## Services

- **CrawlPageV2**: A web crawling service that takes a URL and returns the page content.
- **SerperCacheV2**: Caches results from the Serper search engine to reduce latency and avoid duplicate searches.

## Environment Setup

Get the local IP address of the machine where the services will be started, and set the following variable in `environment.sh`:
```bash
export SERVER_HOST="YOUR_SERVER_IP"
```
Replace `YOUR_SERVER_IP` with the IP address of your server.

## Usage

Use the `start_servers_v2.sh` script to manage the servers.

- **Start Servers**:
  ```bash
  ./start_servers_v2.sh start
  ```
  This command starts the `CrawlPageV2` and `SerperCacheV2` services.

- **Stop Servers**:
  ```bash
  ./start_servers_v2.sh stop
  ```

- **Check Server Status**:
  ```bash
  ./start_servers_v2.sh status
  ```

- **Run Tests**:
  ```bash
  ./start_servers_v2.sh test
  ```
  This command runs the test scripts in the `server_tests` directory to verify that the services are working correctly.

## Directory Structure

```
.
├── start_servers_v2.sh   # Shell script to manage servers
├── server_tests/               # Test scripts directory
│   ├── test_cache_serper_server_v2.py
│   ├── test_crawl_page_simple_v2.py
│   └── ...
├── v2/                   # V2 server implementation
│   ├── cache_serper_server_v2_train.py
│   ├── crawl_page_server_v2.py
│   └── keys.py
├── wiki_server/                # Wiki RAG Server
│   ├── launch_rag_server.sh
│   └── wiki_rag_server.py
└── README.md                   # This document
```
