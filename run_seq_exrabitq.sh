#!/bin/bash
set -euo pipefail

# ==============================================
# 1. 全局通用配置
# ==============================================
# K (Clusters) 列表，对应 SYMQG 的 Degree
K_CLUSTERS_LIST="1024 4096"
# B (Codebooks) 列表，对应 SYMQG 的 EF_BUILD
B_BITS_LIST="8 16"
# GT配置
GT_LIST="gt1 gt10 gt100"
GT_KNN_MAP="gt1=1,gt10=10,gt100=100"

NUM_RUNS="$1"

# 可执行文件路径 (请根据实际编译位置修改)
SEARCHING_EXECUTABLE="/home/pzp/code/fast_graph/Extended-RaBitQ-try-run/bin/test_search"
INDEXING_EXECUTABLE="/home/pzp/code/fast_graph/Extended-RaBitQ-try-run/bin/create_index"
LOG_DIR="/home/pzp/code/fast_graph/Extended-RaBitQ-try-run/experiment_logs"
mkdir -p "$LOG_DIR"

# 数据集配置
DATASET_CONFIGS=(
    "spacev100m:L2:/data/pzp/datasets/spacev100m"
    "word2vec:L2:/data/pzp/datasets/word2vec"
    "sift10m:L2:/data/pzp/datasets/sift10m"
    "gist:L2:/data/pzp/datasets/gist"
    "glove2.2m:L2:/data/pzp/datasets/glove2.2m"
    "tiny5m:L2:/data/pzp/datasets/tiny5m"
    "imagenet:L2:/data/pzp/datasets/imagenet"
)

# ==============================================
# 3. 工具函数
# ==============================================
get_dataset_config() {
    local dataset="$1"
    local config_key="$2"
    local config_line=$(printf '%s\n' "${DATASET_CONFIGS[@]}" | grep "^${dataset}:")
    if [ -z "$config_line" ]; then echo "Dataset not configured: $dataset" >&2; exit 1; fi
    local distance=$(echo "$config_line" | cut -d':' -f2)
    local path=$(echo "$config_line" | cut -d':' -f3)
    case "$config_key" in
        "distance") echo "$distance" ;;
        "path") echo "$path" ;;
    esac
}

get_knn_by_gt() {
    local gt_type="$1"
    echo "$GT_KNN_MAP" | tr ',' '\n' | grep "^${gt_type}=" | cut -d'=' -f2
}

# ==============================================
# 4. 核心任务函数
# ==============================================
process_index_combination() {
    local dataset="$1"
    local K="$2"   # Clusters
    local B="$3"   # Bits
    local distance="$4"
    local dataset_path="$5"
    local data_bin="$6"
    local query_bin="$7"
    
    # 自动推导 centroids 和 cids 文件路径
    # 假设命名规则: ${dataset}_centroid_${K}.fvecs
    # 注意：如果 centroids 也是 i8bin 格式，这里可能需要相应调整 C++ 代码读取部分
    # 目前假设 centroids/cids 依然是标准的 fvecs/ivecs
    local centroids_file="${dataset_path}/${dataset}_centroid_${K}.fvecs"
    local cids_file="${dataset_path}/${dataset}_cluster_id_${K}.ivecs"

    if [ ! -f "$centroids_file" ] || [ ! -f "$cids_file" ]; then
        echo "⚠️  跳过：缺少预训练文件 (K=${K})"
        echo "    Need: $centroids_file"
        echo "    Need: $cids_file"
        return 0
    fi

    # 索引命名
    index_path="${dataset_path}/exrabitq_${dataset}_K${K}_B${B}_${distance}.index"
    build_log="${LOG_DIR}/build_${dataset}_K${K}_B${B}.log"

    echo "========================================"
    echo "构建索引：Dataset=${dataset} | K=${K} | B=${B}"
    echo "========================================"
    
    # 调用 indexing executable
    # 参数: data_file index_file K B centroids_file cids_file
    start_time=$(date +%s%N)
    "$INDEXING_EXECUTABLE" \
        "$data_bin" \
        "$index_path" \
        "$K" \
        "$B" \
        "$centroids_file" \
        "$cids_file" >> "$build_log" 2>&1
        
    if [ ! -f "$index_path" ]; then
        echo "❌ 索引构建失败" >&2
        return 1
    fi
    end_time=$(date +%s%N)
    duration_sec=$(echo "scale=1; ($end_time - $start_time)/1000000000" | bc)
    echo "[Build Time] ${duration_sec}s" >> "$build_log"

    # 执行搜索任务
    local task_list=()
    for gt_type in $GT_LIST; do
        local knn=$(get_knn_by_gt "$gt_type")
        local gt_path="${dataset_path}/${gt_type}" # Symlink to .ibin
        if [ ! -f "$gt_path" ]; then
            # 尝试 fallback 到 .ivecs
            gt_path="${dataset_path}/${dataset}_groundtruth.ivecs" 
            if [ ! -f "$gt_path" ]; then
                echo "❌ GT Missing: $gt_path"; continue
            fi
        fi
        task_list+=("${dataset}|${K}|${B}|${gt_type}|${knn}|${distance}|${dataset_path}|${index_path}|${data_bin}|${query_bin}|${gt_path}")
    done

    # 串行执行搜索
    for task in "${task_list[@]}"; do
        IFS='|' read -r ds k b gt knn dist dpath idx dbin qbin gtpath <<< "$task"
        run_search_task "$ds" "$k" "$b" "$gt" "$knn" "$dist" "$dpath" "$idx" "$dbin" "$qbin" "$gtpath"
    done

    # 删除索引
    rm -f "$index_path"
    echo "✅ 索引已删除: $index_path"
}

run_search_task() {
    local dataset="$1"
    local K="$2"
    local B="$3"
    local gt_type="$4"
    local knn="$5"
    local distance="$6"
    local dataset_path="$7"
    local index_path="$8"
    local data_bin="$9"
    local query_bin="${10}"
    local gt_path="${11}"

    # 输出文件名使用 exrabitq_ 前缀
    OUTPUT_CSV="${dataset_path}/exrabitq_${dataset}_deg${K}_ef${B}_${distance}_${gt_type}_smoothed.csv"

    echo "→ 搜索: K=$K, B=$B, GT=$gt_type"
    
    local tmp_log="${LOG_DIR}/search_tmp_${dataset}_${K}_${B}_${gt_type}.log"
    rm -f "$tmp_log"

    for ((i=1; i<=NUM_RUNS; i++)); do
        # search args: search <deg> <dist> <index> <data> <query> <gt> <k> [rounds]
        # deg/dist 只是为了兼容参数位，不影响 exrabitq 逻辑 (C++代码中实际上K已经包含在index里了)
        "$SEARCHING_EXECUTABLE" "search" "$K" "$distance" \
            "$index_path" "$data_bin" "$query_bin" "$gt_path" "$knn" "1" >> "$tmp_log" 2>&1
    done

    # 聚合 CSV (匹配 C++ 输出格式: ef,recall,qps,latency,memory)
    # 正则匹配数字开头的 CSV 行
    grep -hE '^[0-9]+,[0-9.]+,[0-9.]+,[0-9.]+,[0-9]+$' "$tmp_log" > "${OUTPUT_CSV}.tmp"
    
    if [ -s "${OUTPUT_CSV}.tmp" ]; then
        echo "ef,recall@$knn,qps,mean_latency_ms,memory_usage_mb" > "$OUTPUT_CSV"
        # 简单均值聚合
        EF_VALUES=$(cut -d',' -f1 "${OUTPUT_CSV}.tmp" | sort -n | uniq)
        for ef in $EF_VALUES; do
            lines=$(awk -F',' -v e="$ef" '$1 == e' "${OUTPUT_CSV}.tmp")
            num=$(echo "$lines" | wc -l)
            # 计算平均 (recall, qps, latency, memory)
            awk -F',' -v n="$num" -v ef="$ef" '{r+=$2; q+=$3; l+=$4; m+=$5} END {printf "%d,%.6f,%.6f,%.6f,%.0f\n", ef, r/n, q/n, l/n, m/n}' <<< "$lines" >> "$OUTPUT_CSV"
        done
        echo "✅ 结果保存: $OUTPUT_CSV"
    else
        echo "❌ 无有效数据生成"
        cat "$tmp_log" # 打印日志以便调试
    fi
    rm -f "${OUTPUT_CSV}.tmp" "$tmp_log"
}

# ==============================================
# 5. 主流程
# ==============================================
if [ $# -ne 1 ]; then echo "Usage: $0 <num_runs>"; exit 1; fi

for config in "${DATASET_CONFIGS[@]}"; do
    dataset=$(echo "$config" | cut -d':' -f1)
    distance=$(get_dataset_config "$dataset" "distance")
    dataset_path=$(get_dataset_config "$dataset" "path")
    
    echo "=== Processing Dataset: $dataset ==="
    
    # --- 自动检测数据文件 ---
    data_bin=""
    query_bin=""

    # 1. Base Data
    if [ -f "${dataset_path}/data.i8bin" ]; then
        data_bin="${dataset_path}/data.i8bin"
        echo "→ Base: data.i8bin (int8)"
    elif [ -f "${dataset_path}/data.fbin" ]; then
        data_bin="${dataset_path}/data.fbin"
        echo "→ Base: data.fbin (float32)"
    else
        echo "❌ No data.i8bin or data.fbin found in ${dataset_path}"
        exit 1
    fi

    # 2. Query Data
    if [ -f "${dataset_path}/query.i8bin" ]; then
        query_bin="${dataset_path}/query.i8bin"
        echo "→ Query: query.i8bin (int8)"
    elif [ -f "${dataset_path}/query.fbin" ]; then
        query_bin="${dataset_path}/query.fbin"
        echo "→ Query: query.fbin (float32)"
    else
        echo "❌ No query.i8bin or query.fbin found in ${dataset_path}"
        exit 1
    fi
    
    # 循环遍历参数组合
    for K in $K_CLUSTERS_LIST; do
        for B in $B_BITS_LIST; do
            process_index_combination "$dataset" "$K" "$B" "$distance" "$dataset_path" "$data_bin" "$query_bin"
            # 简单的冷却时间，避免 IO 瞬间过载
            sleep 2
        done
    done
done