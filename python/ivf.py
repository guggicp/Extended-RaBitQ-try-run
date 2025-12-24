import os
import numpy as np
import faiss
import struct
import argparse
from tqdm import tqdm

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
    # 统一命名规范：centroids_k{k}.fbin 和 cids_k{k}.ibin
    centroids_path = os.path.join(output_dir, f"centroids_k{k}.fbin")
    cids_path = os.path.join(output_dir, f"cids_k{k}.ibin")

    if os.path.exists(centroids_path) and os.path.exists(cids_path):
        print(f"✅ [Skip] Clusters for K={k} already exist.")
        return

    # 1. 检测数据类型并打开 mmap
    n, d = read_header(data_path)
    is_i8 = data_path.endswith(".i8bin")
    dtype = np.int8 if is_i8 else np.float32
    print(f"🚀 Processing {dataset_name}: N={n:,}, D={d}, Type={'int8' if is_i8 else 'float32'}")

    data_mmap = get_mmap_data(data_path, n, d, dtype)

    # 2. 确定训练集大小 (融合你的逻辑)
    # 你的逻辑: min(nvecs, max(1000000, k * 40))
    # 脚本 A 逻辑: 优先满足 k*40, 尽量达到 k*256, 上限 20M
    min_req = k * 40
    train_size = min(n, max(1_000_000, min_req))
    train_size = min(train_size, 20_000_000) # 封顶 2000万，防止 RAM 爆炸

    # 3. 随机采样训练数据
    print(f"   🎲 Sampling {train_size:,} points for training...")
    rs = np.random.RandomState(seed)
    indices = rs.choice(n, size=train_size, replace=False)
    indices.sort() # 排序后读取磁盘更顺滑
    train_data = data_mmap[indices]
    
    # 如果是 int8，必须转为 float32 才能给 FAISS 训练
    if is_i8:
        train_data = train_data.astype(np.float32)

    # 4. 训练 K-Means
    print(f"   🔥 Training K-Means (K={k})...")
    kmeans = faiss.Kmeans(d, k, niter=25, verbose=True, seed=seed, gpu=False)
    kmeans.train(train_data)
    
    write_bin(centroids_path, kmeans.centroids)
    del train_data # 释放内存

    # 5. 分配聚类 ID (Assign)
    print(f"   🏷️  Assigning Cluster IDs...")
    index = faiss.IndexFlatL2(d)
    index.add(kmeans.centroids)

    batch_size = 200_000
    with open(cids_path, "wb") as f_out:
        f_out.write(struct.pack('<II', n, 1)) # Header: N, 1
        for start in tqdm(range(0, n, batch_size)):
            end = min(start + batch_size, n)
            batch_vecs = data_mmap[start:end]
            
            # 如果是 int8，搜索前需要转换
            if is_i8:
                batch_vecs = batch_vecs.astype(np.float32)
            
            _, I = index.search(batch_vecs, 1)
            I.astype(np.uint32).tofile(f_out) # 存为 uint32 二进制

    print(f"   ✨ Done. Files saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--k_list", nargs='+', type=int, default=[1024, 4096])
    args = parser.parse_args()
    
    for k in args.k_list:
        compute_clusters(args.data_path, k, args.dataset)