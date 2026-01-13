import os
import numpy as np
import faiss
import struct
import argparse
from tqdm import tqdm

# --- 环境配置：拉满 112 线程 ---
faiss.omp_set_num_threads(112)

def read_header(filename):
    """读取 fbin/i8bin 的 N 和 D"""
    with open(filename, "rb") as f:
        n, d = struct.unpack('<II', f.read(8))
    return int(n), int(d)

def get_mmap_data(filename, n, d, dtype):
    """使用内存映射打开文件，实现零拷贝读取"""
    return np.memmap(filename, dtype=dtype, mode='r', offset=8, shape=(n, d))

def write_bin(filename, data):
    """高效写入 fbin/ibin 格式 (N, D) + Data"""
    n = data.shape[0]
    d = data.shape[1] if len(data.shape) > 1 else 1
    with open(filename, "wb") as f:
        f.write(struct.pack('<II', n, d))
        data.tofile(f)

def compute_clusters(data_path, k, dataset_name, seed=1234):
    output_dir = os.path.dirname(data_path)
    # 路径严格适配顶层调用脚本
    centroids_path = os.path.join(output_dir, f"{dataset_name}_centroid_{k}.fbin")
    cids_path = os.path.join(output_dir, f"{dataset_name}_cluster_id_{k}.ibin")
    dists_path = os.path.join(output_dir, f"{dataset_name}_dist_to_centroid_{k}.fbin")

    if os.path.exists(centroids_path) and os.path.exists(cids_path) and os.path.exists(dists_path):
        print(f"✅ [Skip] Clusters for {dataset_name} K={k} already exist.")
        return

    # 1. 检测数据类型
    n, d = read_header(data_path)
    is_i8 = data_path.endswith(".i8bin")
    dtype = np.int8 if is_i8 else np.float32
    print(f"🚀 Processing {dataset_name}: N={n:,}, D={d}, Type={'int8' if is_i8 else 'float32'}")

    # 2. 确定采样大小 (Faiss 标准: k * 256)
    train_size = min(n, max(k * 256, 1_000_000))

    data_mmap = get_mmap_data(data_path, n, d, dtype)

    # 3. 采样数据
    if train_size >= n:
        print(f"   📥 Loading full dataset for training...")
        train_data = data_mmap[:].astype(np.float32) if is_i8 else data_mmap[:]
    else:
        print(f"   🎲 Sampling {train_size:,} points for training (Standard: k * 256)...")
        rs = np.random.RandomState(seed)
        indices = rs.choice(n, size=train_size, replace=False)
        indices.sort()
        train_data = data_mmap[indices]
        if is_i8:
            train_data = train_data.astype(np.float32)

    # 4. 训练 K-Means
    print(f"   🔥 Training K-Means (K={k})...")
    # 【改动】去掉 min_points_per_centroid 和 max_points_per_centroid
    # 保留 nredo=5（尝试5次找全局最优）和 niter=35（保证收敛）
    # 这将允许桶的分布处于“自然”状态（即便负载不均衡也会保留）
    kmeans = faiss.Kmeans(d, k, niter=35, verbose=True, seed=seed, gpu=False, nredo=1)
    kmeans.train(train_data)
    
    write_bin(centroids_path, kmeans.centroids)
    del train_data # 释放内存

    # 5. 分配 ID 并计算点到质心的 L2 距离
    print(f"   🏷️  Assigning Cluster IDs and Distances...")
    index = faiss.IndexFlatL2(d)
    index.add(kmeans.centroids)

    batch_size = 1_000_000
    with open(cids_path, "wb") as f_cids, open(dists_path, "wb") as f_dists:
        f_cids.write(struct.pack('<II', n, 1))
        f_dists.write(struct.pack('<II', n, 1))
        
        for start in tqdm(range(0, n, batch_size)):
            end = min(start + batch_size, n)
            batch_vecs = data_mmap[start:end]
            if is_i8: batch_vecs = batch_vecs.astype(np.float32)
            
            D, I = index.search(batch_vecs, 1)
            
            # 保存聚类 ID (uint32)
            I.astype(np.uint32).tofile(f_cids)
            
            # 保存距离开方 (float32)，对应 ExRaBitQ 所需的残差辅助信息
            dist_to_centroid = np.sqrt(D)
            dist_to_centroid.astype(np.float32).tofile(f_dists)

    # 稳健地获取 imbalance 统计，防止 KeyError 导致任务失败
    imb = -1.0
    try:
        if len(kmeans.iteration_stats) > 0:
            last_stat = kmeans.iteration_stats[-1]
            if hasattr(last_stat, 'imbalance'):
                imb = last_stat.imbalance
            elif isinstance(last_stat, dict) and 'imbalance' in last_stat:
                imb = last_stat['imbalance']
    except: pass

    print(f"   ✨ Done. Files saved to {output_dir}. Natural Imbalance: {imb:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--k_list", nargs='+', type=int, default=[4096, 16384, 65536, 262144])
    args = parser.parse_args()
    
    for k in args.k_list:
        compute_clusters(args.data_path, k, args.dataset)