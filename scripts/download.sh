# 激活 searchr1 环境
# conda activate searchr1

# 下载索引和语料库
save_path=/var/lib/container/dataset/yxqiu/projects/Search-R1/data
mkdir -p $save_path
python scripts/download.py --save_path $save_path

# 合并索引文件
cat $save_path/part_* > $save_path/e5_Flat.index

# 解压语料库
gzip -d $save_path/wiki-18.jsonl.gz

# 处理 NQ 数据集
python scripts/data_process/nq_search.py