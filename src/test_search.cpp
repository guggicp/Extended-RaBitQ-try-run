#define HIGH_ACC_FAST_SCAN
#define EIGEN_DONT_PARALLELIZE
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <sys/resource.h>
#include <algorithm>

#include "defines.hpp"
#include "index/IVF.hpp"
#include "utils/IO.hpp"
#include "utils/StopW.hpp"

// --- 1. 内存监控 ---
size_t get_memory_usage_mb() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return static_cast<size_t>(usage.ru_maxrss / 1024);
    }
    return 0;
}

// --- 2. 智能参数生成器 (适配 IVF) ---
class BeamSizeGenerator {
private:
    int e;
    size_t index;
    // 针对 IVF，nprobe 需要从很小的值开始 (1, 2, 3...)
    const std::vector<int> bases; 
    int k; // topk

public:
    BeamSizeGenerator(int k) 
        // 扩展 bases 以覆盖小的 nprobe
        : bases({1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80}), k(k) {
        // 强制从 e=0 (即 10^0=1) 开始，保证覆盖 nprobe=1,2,3
        e = 0; 
        index = 0;
    }

    int next() {
        int val = bases[index] * static_cast<int>(pow(10.0, static_cast<double>(e)));
        index++;
        if (index >= bases.size()) {
            e++;
            index = 0;
        }
        return val;
    }
};

// --- 3. 数据读取 (支持 .i8bin/.fbin/.fvecs) ---
void load_float_mat_generic(const std::string& path, FloatRowMat& mat) {
    if (path.find(".fbin") != std::string::npos) {
        std::ifstream in(path, std::ios::binary);
        if(!in) { std::cerr << "Open failed: " << path << std::endl; exit(1); }
        uint32_t n, d; in.read((char*)&n, 4); in.read((char*)&d, 4);
        mat.resize(n, d);
        in.read((char*)mat.data(), n * d * sizeof(float));
    } else if (path.find(".i8bin") != std::string::npos) {
        std::ifstream in(path, std::ios::binary);
        if(!in) { std::cerr << "Open failed: " << path << std::endl; exit(1); }
        uint32_t n, d; in.read((char*)&n, 4); in.read((char*)&d, 4);
        mat.resize(n, d);
        std::vector<int8_t> buf(n * d);
        in.read((char*)buf.data(), n * d * sizeof(int8_t));
        #pragma omp parallel for
        for(size_t i=0; i<n*d; ++i) mat.data()[i] = static_cast<float>(buf[i]);
    } else {
        load_vecs<float, FloatRowMat>(path.c_str(), mat);
    }
}

void load_uint_mat_generic(const std::string& path, UintRowMat& mat) {
    if (path.find(".ibin") != std::string::npos) {
        std::ifstream in(path, std::ios::binary);
        if(!in) { std::cerr << "Open failed: " << path << std::endl; exit(1); }
        uint32_t n, d; in.read((char*)&n, 4); in.read((char*)&d, 4);
        mat.resize(n, d);
        std::vector<int32_t> buf(n * d);
        in.read((char*)buf.data(), n * d * sizeof(int32_t));
        for(size_t i=0; i<n*d; ++i) mat.data()[i] = (PID)buf[i];
    } else {
        load_vecs<PID, UintRowMat>(path.c_str(), mat);
    }
}

// --- 4. 自动寻找最佳 nprobes (复刻 find_EFS 逻辑) ---
std::vector<int> find_nprobes(
    IVF& ivf, 
    const FloatRowMat& rotated_query, 
    const FloatRowMat& data, 
    const std::vector<std::unordered_set<PID>>& gt_sets,
    int topk, 
    const std::string& distance
) {
    std::vector<int> selected_nprobes;
    BeamSizeGenerator W(topk);
    float prev_recall = 0.0f;
    size_t NQ = rotated_query.rows();
    size_t total_num = NQ * topk;
    size_t max_nprobe = ivf.k(); // IVF 的聚类中心数

    std::cout << "[Info] Finding optimal nprobes for distance=" << distance << ", topk=" << topk << " ...\n";
    std::cout << "[Info] Candidates: ";

    std::unordered_set<int> visited;

    while (true) {
        int nprobe = W.next();

        // 边界保护：不能超过总聚类数
        if (nprobe > max_nprobe) {
            if (visited.find(max_nprobe) == visited.end()) nprobe = max_nprobe;
            else break;
        }
        if (visited.count(nprobe)) continue;
        visited.insert(nprobe);

        // 快速跑一次获取 Recall
        size_t total_correct = 0;
        StopW stopw;
        std::vector<PID> results(topk);
        
        for (size_t i = 0; i < NQ; ++i) {
            ivf.search(&rotated_query(i, 0), data.data(), topk, nprobe, results.data());
            for (PID id : results) {
                if (gt_sets[i].count(id)) total_correct++;
            }
        }
        float time_us = stopw.getElapsedTimeMicro();
        float recall = static_cast<float>(total_correct) / total_num;
        float qps = NQ / (time_us / 1e6);

        std::cout << nprobe << " ";
        std::cout.flush();
        
        selected_nprobes.push_back(nprobe);

        // 停止条件：Recall 极高 或 提升停滞 或 速度太慢
        if (recall > 0.998f || (recall - prev_recall) < 0.0005f || qps < 10.0f || nprobe == max_nprobe) {
            break;
        }
        prev_recall = recall;
    }
    std::cout << "\n[Info] Selection done. Found " << selected_nprobes.size() << " probes.\n";
    return selected_nprobes;
}

// --- 5. 主函数 ---
int main(int argc, char* argv[]) {
    // 兼容 shell 脚本的参数位置：
    // argv: [0]exe [1]"search" [2]Deg(K) [3]Dist [4]Index [5]Data [6]Query [7]GT [8]k [9]Rounds
    if (argc < 10) {
        std::cerr << "Usage: " << argv[0] << " search <deg> <dist> <index> <data> <query> <gt> <k> [rounds]\n";
        return 1;
    }

    std::string distance_str = argv[3];
    std::string index_file = argv[4];
    std::string data_file = argv[5];
    std::string query_file = argv[6];
    std::string gt_file = argv[7];
    size_t TOPK = std::stoul(argv[8]);
    size_t ROUND = std::stoul(argv[9]);

    // 加载数据
    FloatRowMat data;
    FloatRowMat query;
    UintRowMat gt_mat;
    load_float_mat_generic(data_file, data);
    load_float_mat_generic(query_file, query);
    load_uint_mat_generic(gt_file, gt_mat);

    size_t N = data.rows();
    size_t DIM = data.cols();
    size_t NQ = query.rows();
    std::cout << "Data loaded: N=" << N << ", DIM=" << DIM << ", NQ=" << NQ << "\n";

    // 转换 GT 为 Set 加速比对
    std::vector<std::unordered_set<PID>> gt_sets(NQ);
    for(size_t i=0; i<NQ; ++i) {
        for(size_t j=0; j<(size_t)gt_mat.cols(); ++j) gt_sets[i].insert(gt_mat(i, j));
    }

    // 加载索引
    size_t m1 = get_memory_usage_mb();
    IVF ivf;
    ivf.load(index_file.c_str());
    size_t m2 = get_memory_usage_mb();
    size_t memory_mb = m2 - m1;

    // 预处理 Query (Rotation)
    StopW stopw;
    FloatRowMat padded_query(NQ, ivf.padded_dim());
    padded_query.setZero();
    FloatRowMat rotated_query(NQ, ivf.padded_dim());
    for(size_t i=0; i<NQ; ++i) {
        std::memcpy(&padded_query(i, 0), &query(i, 0), sizeof(float) * DIM);
    }
    Rotator& rp = ivf.rotator();
    stopw.reset();
    rp.rotate(padded_query, rotated_query);
    float rotate_time_us = stopw.getElapsedTimeMicro(); // 旋转总耗时

    // 自动选择参数
    std::vector<int> nprobes = find_nprobes(ivf, rotated_query, data, gt_sets, TOPK, distance_str);

    // --- 正式测试 (Output CSV) ---
    // 这是为了欺骗 plot 脚本，ExRaBitQ 的 "ef" 就是 "nprobe"
    std::cout << "ef,recall@" << TOPK << ",qps,mean_latency_ms,memory_usage_mb" << std::endl;

    for (int nprobe : nprobes) {
        double sum_recall = 0.0;
        double sum_qps = 0.0;
        double sum_lat = 0.0;

        // 这里相当于原来的 horizontal_avg 逻辑：
        // 跑 ROUND 次，累加结果，最后取平均
        for (size_t r = 0; r < ROUND; ++r) {
            size_t total_correct = 0;
            float search_time_us = 0;
            std::vector<PID> results(TOPK);

            for (size_t i = 0; i < NQ; ++i) {
                stopw.reset();
                ivf.search(&rotated_query(i, 0), data.data(), TOPK, nprobe, results.data());
                search_time_us += stopw.getElapsedTimeMicro();

                for (PID id : results) {
                    if (gt_sets[i].count(id)) total_correct++;
                }
            }

            // 指标计算 (包含 Rotation 时间，公平对比)
            float total_time_us = search_time_us + rotate_time_us;
            float qps = NQ / (total_time_us / 1e6);
            float recall = static_cast<float>(total_correct) / (NQ * TOPK);
            float latency_ms = (total_time_us / NQ) / 1000.0f;

            sum_recall += recall;
            sum_qps += qps;
            sum_lat += latency_ms;
        }

        // 输出平均值
        std::cout << nprobe << "," 
                  << (sum_recall / ROUND) << "," 
                  << (sum_qps / ROUND) << "," 
                  << (sum_lat / ROUND) << "," 
                  << memory_mb << std::endl;
    }

    return 0;
}