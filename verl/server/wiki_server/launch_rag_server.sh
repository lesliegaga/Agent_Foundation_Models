# nohup bash launch_rag_server.sh > launch_rag_server_8000.log 2>&1 &

file_path=path_to_your_index_data_dir
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=path_to_your_embedding_file
port=8000

python server/wiki_rag_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu \
                                            --port $port
