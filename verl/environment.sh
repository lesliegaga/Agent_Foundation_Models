export SERVER_HOST=10.77.225.105
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
export GRM_BASE_URL="https://api.uniapi.vip/v1"
export GRM_API_KEY="api key"
export GRM_MODEL_NAME="gpt-4.1-mini"
# =====================================================================================================================
#                                      Jina
# =====================================================================================================================
# for crawl page
export JINA_BASE_URL="https://s.jina.ai/"
export JINA_API_KEY="api key"
# =====================================================================================================================
#                                      Serper
# =====================================================================================================================
# for web search
export WEB_SEARCH_METHOD_TYPE="serapi"
export WEB_SEARCH_SERP_NUM="10"
export WEB_SEARCH_SERPER_API_KEY="api key"
# =====================================================================================================================
#                                      Summary Model
# =====================================================================================================================
# for summary of crawl page content
export SUMMARY_OPENAI_API_BASE_URL="api url"
export SUMMARY_OPENAI_API_KEY="api key"
export SUMMARY_MODEL="qwen2.5-72b-instruct"
# =====================================================================================================================
#                                      Code testcases
# =====================================================================================================================
# for evaluate code val datasets
export LIVECODEBENCH_DATA_PATH="data/livecodebench_testcases"

export UNI_API_URLS="https://api.uniapi.vip/v1" # or other qwen3 api provider
export UNI_API_KEY="api key"