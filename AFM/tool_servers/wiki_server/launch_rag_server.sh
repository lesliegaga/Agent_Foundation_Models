# nohup bash launch_rag_server.sh > launch_rag_server_8000.log 2>&1 &

file_path=/mnt/tongyan.zjy/data/afm/wiki
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=/mnt/tongyan.zjy/openlm/model/intfloat/e5-base-v2
port=8000

python -u wiki_rag_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu \
                                            --port $port
