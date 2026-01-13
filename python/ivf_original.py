import os
import numpy as np
import faiss
import struct
import argparse
import time
from tqdm import tqdm

# --- 辅助函数：读写高性能二进制格式 (与新脚本一致) ---
def read_header(filename):
    with open(filename, "rb") as f:
        n, d = struct.unpack('<II', f.read(8))
    return int(n), int(d)

def write_bin(filename, data):
    n = data.shape[0]
    d = data.shape[1] if len(data.shape) > 1 else 1
    with open(filename, "wb") as f:
        f.write(struct.pack('<II', n, d))
        data.tofile(f)

def run_original_clustering(data_path, k, dataset_name):
    output_dir = os.path.dirname(data_path)
    
    # 路径严格适配你现有的 run_seq 脚本
    centroids_path = os.path.join(output_dir, f"{dataset_name}_centroid_{k}.fbin")
    cids_path = os.path.join(output_dir, f"{dataset_name}_cluster_id_{k}.ibin")
    dists_path = os.path.join(output_dir, f"{dataset_name}_dist_to_centroid_{k}.fbin")

    # 1. 加载数据 (原始逻辑：全量加载进内存)
    print(f"📦 [Original] Loading full dataset into RAM: {dataset_name}")
    n, d = read_header(data_path)
    # 注意：这里会触发全量读取，如果内存不足会报错，这正是原始代码的瓶颈
    X = np.fromfile(data_path, dtype=np.float32, offset=8).reshape(n, d)

    # 2. 原始聚类逻辑：使用 index_factory
    print(f"🔨 [Original] Training IVF{k} on FULL dataset (No Sampling)...")
    start_time = time.time()
    
    index = faiss.index_factory(d, f"IVF{k},Flat")
    index.verbose = True
    
    # 全量训练
    index.train(X)
    train_time = time.time() - start_time
    print(f"⏱️ Training took: {train_time:.2f}s")

    # 3. 提取质心
    centroids = index.quantizer.reconstruct_n(0, index.nlist)

    # 4. 原始分配逻辑：全量搜索 (原始代码不分批，容易 OOM，这里保持其语义)
    print(f"🏷️  [Original] Assigning clusters for FULL dataset...")
    assign_start = time.time()
    
    # 获取距离（平方）和 ID
    D, I = index.quantizer.search(X, 1)
    
    # 转换为距离开方
    dist_to_centroid = np.sqrt(D).astype(np.float32)
    cluster_id = I.astype(np.uint32)
    
    assign_time = time.time() - assign_start
    print(f"⏱️ Assignment took: {assign_time:.2f}s")

    # 5. 写入文件 (格式与高性能版一致，方便 C++ 读取)
    write_bin(centroids_path, centroids)
    write_bin(cids_path, cluster_id)
    write_bin(dists_path, dist_to_centroid)

    print(f"✨ [Original] Done. Total Pre-process Time: {train_time + assign_time:.2f}s")
    print(f"💾 Files saved: {centroids_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Original Clustering Method")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--k_list", nargs='+', type=int, default=[4096])
    args = parser.parse_args()
    
    for k in args.k_list:
        run_original_clustering(args.data_path, k, args.dataset)