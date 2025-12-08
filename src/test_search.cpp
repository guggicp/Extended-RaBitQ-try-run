#define HIGH_ACC_FAST_SCAN
#define EIGEN_DONT_PARALLELIZE
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <sys/resource.h> // 用于内存监控

#include "defines.hpp"
#include "index/IVF.hpp"
#include "utils/IO.hpp"
#include "utils/StopW.hpp"

// 获取内存使用 (MB)
size_t get_memory_usage_mb() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return static_cast<size_t>(usage.ru_maxrss / 1024);
    }
    return 0;
}

// 通用 fbin/ibin 读取辅助
void read_fbin(const std::string& path, std::vector<float>& data, size_t& nvecs, size_t& dims) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "无法打开文件: " << path << std::endl; exit(1); }
    uint32_t n, d;
    in.read((char*)&n, 4); in.read((char*)&d, 4);
    nvecs = n; dims = d;
    data.resize(nvecs * dims);
    in.read((char*)data.data(), data.size() * sizeof(float));
}

void read_ibin(const std::string& path, std::vector<int32_t>& data, size_t& nvecs, size_t& dims) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "无法打开文件: " << path << std::endl; exit(1); }
    uint32_t n, d;
    in.read((char*)&n, 4); in.read((char*)&d, 4);
    nvecs = n; dims = d;
    data.resize(nvecs * dims);
    in.read((char*)data.data(), data.size() * sizeof(int32_t));
}

void load_float_mat(const std::string& path, FloatRowMat& mat) {
    if (path.find(".fbin") != std::string::npos) {
        std::vector<float> d; size_t n, dim;
        read_fbin(path, d, n, dim);
        mat.resize(n, dim);
        std::memcpy(mat.data(), d.data(), n * dim * sizeof(float));
    } else {
        load_vecs<float, FloatRowMat>(path.c_str(), mat);
    }
}

void load_uint_mat(const std::string& path, UintRowMat& mat) {
    if (path.find(".ibin") != std::string::npos) {
        std::vector<int32_t> d; size_t n, dim;
        read_ibin(path, d, n, dim);
        mat.resize(n, dim);
        // 注意类型转换 int32 -> PID (uint32)
        for(size_t i=0; i<n*dim; ++i) mat.data()[i] = (PID)d[i];
    } else {
        load_vecs<PID, UintRowMat>(path.c_str(), mat);
    }
}

int main(int argc, char* argv[]) {
    // 统一参数格式:
    // search <degree_ignored> <distance_ignored> <index_path> <data_file> <query_file> <gt_file> <K> [rounds]
    // 注意：ExRaBitQ 实际上需要知道 B 值，通常 B 值已包含在 Index 中，或者作为参数传递。
    // 为了适配 run_seq 脚本，我们尽量对齐参数位置。
    // 如果 run_seq 调用： $EXECUTABLE "search" $degree $distance $index_path $data_fbin $query_path $gt_path $knn $rounds
    
    if (argc < 9) {
        std::cerr << "Usage: " << argv[0] << " search <deg> <dist> <index> <data> <query> <gt> <k> [rounds]\n";
        return 1;
    }

    std::string index_file = argv[4];
    std::string data_file = argv[5];
    std::string query_file = argv[6];
    std::string gt_file = argv[7];
    size_t TOPK = std::stoul(argv[8]);
    size_t ROUND = (argc >= 10) ? std::stoul(argv[9]) : 1;

    FloatRowMat data;
    FloatRowMat query;
    UintRowMat gt;

    // 1. 加载数据
    load_float_mat(data_file, data);
    load_float_mat(query_file, query);
    load_uint_mat(gt_file, gt);

    size_t N = data.rows();
    size_t DIM = data.cols();
    size_t NQ = query.rows();

    std::cout << "Data loaded: N=" << N << ", DIM=" << DIM << ", NQ=" << NQ << "\n";

    // 2. 加载索引
    size_t m1 = get_memory_usage_mb();
    IVF ivf;
    ivf.load(index_file.c_str());
    size_t m2 = get_memory_usage_mb();
    size_t memory_mb = m2 - m1;

    // 3. 准备 nprobes 列表 (相当于 ef 列表)
    // 自动生成 nprobes 序列
    std::vector<size_t> nprobes;
    std::vector<size_t> bases = {1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 40, 50, 80, 100};
    for(size_t b : bases) {
        if (b <= ivf.k()) nprobes.push_back(b);
    }
    // 添加更多大的 nprobe，如果 clusters 数量够大
    for (size_t i = 200; i <= 2000; i += 200) {
        if (i <= ivf.k()) nprobes.push_back(i);
    }

    StopW stopw;

    // 4. Query 预处理 (Padding & Rotation)
    FloatRowMat padded_query(NQ, ivf.padded_dim());
    padded_query.setZero();
    FloatRowMat rotated_query(NQ, ivf.padded_dim());
    for (size_t i = 0; i < NQ; ++i) {
        std::memcpy(&padded_query(i, 0), &query(i, 0), sizeof(float) * DIM);
    }
    Rotator& rp = ivf.rotator();
    stopw.reset();
    rp.rotate(padded_query, rotated_query);
    float rotate_time_us = stopw.getElapsedTimeMicro();
    float rotate_time_per_query_us = rotate_time_us / NQ;

    // 5. 执行搜索并输出符合格式的 Log
    // Header format: ef,recall@k,qps,mean_latency_ms,memory_usage_mb
    std::cout << "ef,recall@" << TOPK << ",qps,mean_latency_ms,memory_usage_mb" << std::endl;

    for (size_t nprobe : nprobes) {
        double total_recall = 0.0;
        double total_qps = 0.0;
        double total_latency_ms = 0.0;

        for (size_t r = 0; r < ROUND; ++r) {
            size_t total_correct = 0;
            float total_search_time_us = 0;
            
            PID results[TOPK]; // buffer

            for (size_t i = 0; i < NQ; i++) {
                stopw.reset();
                ivf.search(&rotated_query(i, 0), data.data(), TOPK, nprobe, results);
                float latency = stopw.getElapsedTimeMicro();
                total_search_time_us += latency;

                // Check GT
                for (size_t j = 0; j < TOPK; j++) {
                    for (size_t k = 0; k < TOPK; k++) {
                        if (gt(i, k) == results[j]) {
                            total_correct++;
                            break;
                        }
                    }
                }
            }
            
            // 计算指标 (包含 Rotation 时间)
            float total_time_us = total_search_time_us + rotate_time_us; // Add global rotation time
            float qps = NQ / (total_time_us / 1e6);
            float recall = static_cast<float>(total_correct) / (NQ * TOPK);
            float avg_lat_ms = (total_time_us / NQ) / 1000.0f;

            total_recall += recall;
            total_qps += qps;
            total_latency_ms += avg_lat_ms;
        }

        // 输出平均值
        std::cout << nprobe << "," 
                  << (total_recall / ROUND) << "," 
                  << (total_qps / ROUND) << "," 
                  << (total_latency_ms / ROUND) << "," 
                  << memory_mb << std::endl;
    }

    return 0;
}