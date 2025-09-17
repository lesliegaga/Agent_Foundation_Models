from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
# ds_sft = load_dataset("PersonalAILab/AFM-WebAgent-SFT-Dataset")
# ds_sft.save_to_disk("./AFM-WebAgent-SFT-Dataset")
#
# ds_rl = load_dataset("PersonalAILab/AFM-WebAgent-RL-Dataset")
# ds_rl.save_to_disk("./AFM-WebAgent-RL-Dataset")


from datasets import download_dataset
download_dataset("amap_search_rag/AFM-WebAgent-SFT-Dataset",local_dir="./AFM-WebAgent-SFT-Dataset")
download_dataset("amap_search_rag/AFM-WebAgent-RL-Dataset",local_dir="./AFM-WebAgent-RL-Dataset")