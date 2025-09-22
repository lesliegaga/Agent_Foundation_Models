export SERVER_HOST=$(hostname -I | awk '{print $1}')
export CRAWL_PAGE_PORT=9000
export WEBSEARCH_PORT=9001

# =====================================================================================================================
#                                      wiki_rag_server
# =====================================================================================================================
export WIKI_RAG_SERVER_URL="http://$SERVER_HOST:8000/retrieve"
# =====================================================================================================================
#                                      GRM
# =====================================================================================================================
# for llm as judge
export GRM_BASE_URL="http://$SERVER_HOST:8101/v1"
export GRM_API_KEY="1234"
export GRM_MODEL_NAME="qwen3_235b"
# =====================================================================================================================
#                                      Jina
# =====================================================================================================================
# for crawl page
export JINA_BASE_URL="https://s.jina.ai/"
export JINA_API_KEY="jina_9a8fc69d2a434d8d83647806d4a44f8ctGKe3Mj9u9k5DL_N5c7KLXFJSyFG"
# =====================================================================================================================
#                                      Serper
# =====================================================================================================================
# for web search
export WEB_SEARCH_METHOD_TYPE="serapi"
export WEB_SEARCH_SERP_NUM="10"
export WEB_SEARCH_SERPER_API_KEY="f43e9afc7ce856912b3ce3629c2ab9e1107041bb"
# =====================================================================================================================
#                                      Summary Model
# =====================================================================================================================
# for summary of crawl page content
export SUMMARY_OPENAI_API_BASE_URL="http://$SERVER_HOST:8101/v1"
export SUMMARY_OPENAI_API_KEY="1234"
export SUMMARY_MODEL="qwen3_235b"
# =====================================================================================================================
#                                      Code testcases
# =====================================================================================================================
# for evaluate code val datasets
export LIVECODEBENCH_DATA_PATH="$(pwd)/data/code_agent/livecodebench_testcases"

export UNI_API_URLS="http://$SERVER_HOST:8101/v1" # or other qwen3 api provider
export UNI_API_KEY="1234"