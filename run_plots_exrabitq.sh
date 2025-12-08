#!/bin/bash
set -u

PROJECT="Extended-RaBitQ-try-run"
# 确保这里指向你的通用绘图脚本路径
PYTHON_SCRIPT="/home/pzp/code/fast_graph/tools/plot_multi_curve_opt.py" 
PLOT_OUTPUT_DIR="/home/pzp/code/fast_graph/${PROJECT}/plots"
mkdir -p "$PLOT_OUTPUT_DIR"

DATASET_CONFIGS=(
    "word2vec:L2:/data/pzp/datasets/word2vec"
    "sift10m:L2:/data/pzp/datasets/sift10m"
    "gist:L2:/data/pzp/datasets/gist"
    "glove2.2m:L2:/data/pzp/datasets/glove2.2m"
    "tiny5m:L2:/data/pzp/datasets/tiny5m"
    "imagenet:L2:/data/pzp/datasets/imagenet"
)

# 注意：ExRaBitQ 的 Degree 参数实际映射为 K (Centroids)
# 在 run_seq 脚本中我们生成的文件是 deg${K}，所以这里遍历 K 列表
DEGREE_LIST="1024 4096"
K_LIST="1 10 100"

echo "========================================"
echo "开始 ExRaBitQ 绘图任务"
echo "========================================"

for config in "${DATASET_CONFIGS[@]}"; do
    dataset_name=$(echo "$config" | cut -d':' -f1)
    distance=$(echo "$config" | cut -d':' -f2)
    full_path=$(echo "$config" | cut -d':' -f3)
    parent_path=$(dirname "$full_path")

    echo ">> Dataset: $dataset_name"

    for degree in $DEGREE_LIST; do
        for k in $K_LIST; do
            echo "   -> Plotting: Clusters(Deg)=$degree, K=$k"
            
            # 关键修改：
            # 1. 传入 dataset-path
            # 2. Python 脚本需要稍微适配来读取 exrabitq_ 前缀
            #    (目前的 python 脚本里 get_csv_path 是硬编码 symqg_ 前缀的)
            #    建议修改 plot_multi_curve_opt.py 增加 --prefix 参数 (参照 HNSW 的做法)
            
            # 如果 python 脚本已支持 --prefix:
             python3 "$PYTHON_SCRIPT" \
                --dataset "$dataset_name" \
                --distance "$distance" \
                --degree "$degree" \
                --k "$k" \
                --dataset-path "$parent_path" \
                --output-dir "$PLOT_OUTPUT_DIR"
            
            # 注意：原 plot_multi_curve_opt.py 似乎写死了 symqg_ 前缀。
            # 你可能需要在 Python 脚本中加入简单的逻辑判断，
            # 或者在 run_seq_exrabitq.sh 中把输出文件命名为 symqg_... 
            # (虽然有点 tricky，但能复用脚本)
            # 我在 run_seq_exrabitq.sh 中已经将输出改为了 exrabitq_...
            # 因此你需要去修改 plot_multi_curve_opt.py 增加 prefix 参数支持。
        done
    done
done