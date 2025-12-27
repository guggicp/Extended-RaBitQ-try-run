#define HIGH_ACC_FAST_SCAN
#define EIGEN_DONT_PARALLELIZE
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <sys/resource.h>
#include <cassert>

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

// --- 2. 健壮的文件读取函数 ---

void load_fbin(const std::string& path, FloatRowMat& mat) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "Open failed: " << path << std::endl; exit(1); }
    uint32_t n, d;
    in.read((char*)&n, 4); in.read((char*)&d, 4);
    mat.resize(n, d);
    in.read((char*)mat.data(), n * d * sizeof(float));
    std::cout << "[IO] Loaded .fbin: " << path << " (" << n << "x" << d << ")\n";
}

void load_i8bin(const std::string& path, FloatRowMat& mat) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "Open failed: " << path << std::endl; exit(1); }
    uint32_t n, d;
    in.read((char*)&n, 4); in.read((char*)&d, 4);
    mat.resize(n, d);
    std::vector<int8_t> buf(n * d);
    in.read((char*)buf.data(), n * d * sizeof(int8_t));
    #pragma omp parallel for
    for(size_t i=0; i<n*d; ++i) mat.data()[i] = static_cast<float>(buf[i]);
    std::cout << "[IO] Loaded .i8bin: " << path << " (" << n << "x" << d << ")\n";
}

void load_fvecs(const std::string& path, FloatRowMat& mat) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "Open failed: " << path << std::endl; exit(1); }
    in.seekg(0, std::ios::end);
    size_t fsize = in.tellg();
    in.seekg(0, std::ios::beg);
    int32_t d; in.read((char*)&d, 4);
    size_t row_size = 4 + d * 4;
    size_t n = fsize / row_size;
    mat.resize(n, d);
    in.seekg(0, std::ios::beg);
    for(size_t i=0; i<n; ++i) {
        int32_t dd; in.read((char*)&dd, 4);
        in.read((char*)&mat(i, 0), d * sizeof(float));
    }
    std::cout << "[IO] Loaded .fvecs: " << path << " (" << n << "x" << d << ")\n";
}

void load_ivecs(const std::string& path, UintRowMat& mat) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "Open failed: " << path << std::endl; exit(1); }
    in.seekg(0, std::ios::end);
    size_t fsize = in.tellg();
    in.seekg(0, std::ios::beg);
    int32_t d; in.read((char*)&d, 4);
    size_t row_size = 4 + d * 4;
    size_t n = fsize / row_size;
    mat.resize(n, d);
    in.seekg(0, std::ios::beg);
    for(size_t i=0; i<n; ++i) {
        int32_t dd; in.read((char*)&dd, 4);
        if(dd != d) { std::cerr << "ivecs dim mismatch\n"; exit(1); }
        std::vector<int32_t> buf(d);
        in.read((char*)buf.data(), d * sizeof(int32_t));
        for(size_t j=0; j<(size_t)d; ++j) mat(i, j) = static_cast<PID>(buf[j]);
    }
    std::cout << "[IO] Loaded .ivecs: " << path << " (" << n << "x" << d << ")\n";
}

void load_ibin(const std::string& path, UintRowMat& mat) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "Open failed: " << path << std::endl; exit(1); }
    uint32_t n, d;
    in.read((char*)&n, 4); in.read((char*)&d, 4);
    mat.resize(n, d);
    std::vector<int32_t> buf(n * d);
    in.read((char*)buf.data(), n * d * sizeof(int32_t));
    for(size_t i=0; i<n*d; ++i) mat.data()[i] = static_cast<PID>(buf[i]);
    std::cout << "[IO] Loaded .ibin: " << path << " (" << n << "x" << d << ")\n";
}

void load_data_auto(const std::string& path, FloatRowMat& mat) {
    if (path.find(".fbin") != std::string::npos) load_fbin(path, mat);
    else if (path.find(".i8bin") != std::string::npos) load_i8bin(path, mat);
    else if (path.find(".fvecs") != std::string::npos) load_fvecs(path, mat);
    else load_fbin(path, mat); // Default
}

// 修复后的 GT 加载逻辑：接受 expected_nq 参数
void load_gt_auto(const std::string& path, UintRowMat& mat, size_t expected_nq) {
    // 1. 优先后缀判断
    if (path.find(".ibin") != std::string::npos) { load_ibin(path, mat); return; }
    if (path.find(".ivecs") != std::string::npos) { load_ivecs(path, mat); return; }

    // 2. 读取 Header 进行智能探测
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "Open failed: " << path << std::endl; exit(1); }
    int32_t val1, val2;
    in.read((char*)&val1, 4);
    in.read((char*)&val2, 4);
    in.close();

    // 核心修复：如果文件头的 N 等于查询数量，且 D 比较小，那就是 .ibin
    // word2vec: N=1000, K=1/10/100.
    // val1=1000 (match expected_nq), val2=1. -> .ibin
    if (val1 == (int32_t)expected_nq && val2 <= 2000) {
        std::cout << "[IO] Header (" << val1 << "," << val2 << ") matches Query Count -> .ibin\n";
        load_ibin(path, mat);
    } 
    // 旧的 heuristic: N > 2000 (对于 word2vec 失败)
    else if (val1 > 2000 && val2 <= 2000) {
        std::cout << "[IO] Header (" << val1 << "," << val2 << ") looks like large .ibin\n";
        load_ibin(path, mat);
    } 
    else {
        std::cout << "[IO] Header (" << val1 << "," << val2 << ") -> assuming .ivecs\n";
        load_ivecs(path, mat);
    }
}

// --- 3. 智能参数生成器 ---
class BeamSizeGenerator {
private:
    int e; size_t index; const std::vector<int> bases; int k;
public:
    BeamSizeGenerator(int k) : bases({1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80}), k(k) { e = 0; index = 0; }
    int next() {
        int val = bases[index] * static_cast<int>(pow(10.0, static_cast<double>(e)));
        index++;
        if (index >= bases.size()) { e++; index = 0; }
        return val;
    }
};

// --- 4. 自动寻找最佳 nprobes ---
std::vector<int> find_nprobes(IVF& ivf, const FloatRowMat& rotated_query, const FloatRowMat& data, 
                              const std::vector<std::unordered_set<PID>>& gt_sets, int topk, const std::string& distance) {
    std::vector<int> selected;
    BeamSizeGenerator W(topk);
    float prev_recall = 0.0f;
    size_t NQ = rotated_query.rows();
    size_t max_nprobe = ivf.k();
    
    std::cout << "[Info] Finding nprobes for topk=" << topk << "...\n[Info] Candidates: ";
    std::unordered_set<int> visited;

    while (true) {
        int nprobe = W.next();
        if (nprobe > max_nprobe) {
            if (visited.find(max_nprobe) == visited.end()) nprobe = max_nprobe; else break;
        }
        if (visited.count(nprobe)) continue;
        visited.insert(nprobe);

        size_t correct = 0;
        StopW stopw;
        std::vector<PID> results(topk);
        for (size_t i = 0; i < NQ; ++i) {
            ivf.search(&rotated_query(i, 0), data.data(), topk, nprobe, results.data());
            for (PID id : results) if (gt_sets[i].count(id)) correct++;
        }
        float recall = (float)correct / (NQ * topk);
        float qps = NQ / (stopw.getElapsedTimeMicro() / 1e6);
        std::cout << nprobe << " "; std::cout.flush();
        selected.push_back(nprobe);

        if (recall > 0.998f || (recall - prev_recall) < 0.0005f || qps < 10.0f || nprobe == max_nprobe) break;
        prev_recall = recall;
    }
    std::cout << "\n";
    return selected;
}

// --- 5. Main ---
int main(int argc, char* argv[]) {
    if (argc < 10) { std::cerr << "Usage error\n"; return 1; }
    std::string dist = argv[3], idx_file = argv[4], data_file = argv[5], q_file = argv[6], gt_file = argv[7];
    size_t TOPK = std::stoul(argv[8]), ROUND = std::stoul(argv[9]);
    int B = std::stoi(argv[2]);
    assert(B == 9 || B == 5 || B == 7 || B == 3 || B == 4 || B == 8);
    FloatRowMat data, query;
    UintRowMat gt_mat;

    std::cout << "--- Loading ---\n";
    load_data_auto(data_file, data);
    load_data_auto(q_file, query);
    
    // 关键修正：先获取 Query 数量，再加载 GT
    size_t NQ = query.rows();
    std::cout << "NQ detected: " << NQ << ". Loading GT...\n";
    load_gt_auto(gt_file, gt_mat, NQ); 

    // 校验 GT 维度
    if (gt_mat.rows() != NQ) {
        std::cerr << "Error: GT rows (" << gt_mat.rows() << ") != Query rows (" << NQ << ")\n";
        exit(1);
    }

    size_t N = data.rows(), DIM = data.cols();
    std::cout << "Data: " << N << "x" << DIM << "\n";

    std::vector<std::unordered_set<PID>> gt_sets(NQ);
    for(size_t i=0; i<NQ; ++i) 
        for(size_t j=0; j<(size_t)gt_mat.cols(); ++j) gt_sets[i].insert(gt_mat(i, j));

    size_t m1 = get_memory_usage_mb();
    IVF ivf; ivf.load(idx_file.c_str());
    size_t m2 = get_memory_usage_mb();

    FloatRowMat p_q(NQ, ivf.padded_dim()), r_q(NQ, ivf.padded_dim());
    p_q.setZero();
    for(size_t i=0; i<NQ; ++i) std::memcpy(&p_q(i, 0), &query(i, 0), sizeof(float) * DIM);
    
    StopW stopw;
    ivf.rotator().rotate(p_q, r_q);
    float rot_time = stopw.getElapsedTimeMicro();

    std::vector<int> nprobes = find_nprobes(ivf, r_q, data, gt_sets, TOPK, dist);
    
    std::cout << "ef,recall@" << TOPK << ",qps,mean_latency_ms,memory_usage_mb" << std::endl;

    const char* env_threads = getenv("OMP_NUM_THREADS");
    int num_t = env_threads ? atoi(env_threads) : 1;
    omp_set_num_threads(num_t);
    for (int np : nprobes) {
        double s_rec = 0, s_qps = 0, s_lat = 0;
        for (size_t r = 0; r < ROUND; ++r) {
            size_t total_corr = 0;
            stopw.reset(); // 开始全量计时
            
            #pragma omp parallel
            {
                std::vector<PID> local_res(TOPK); // 每个线程独立的 buffer
                size_t local_corr = 0;
                
                #pragma omp for nowait // 并行分发查询
                for (size_t i = 0; i < NQ; ++i) {
                    ivf.search(&r_q(i, 0), data.data(), TOPK, np, local_res.data());
                    for (PID id : local_res) {
                        if (gt_sets[i].count(id)) local_corr++;
                    }
                }
                
                #pragma omp atomic
                total_corr += local_corr;
            }
            
            float t_total_seconds = (stopw.getElapsedTimeMicro() + rot_time) / 1e6;
            
            s_rec += (float)total_corr / (NQ * TOPK);
            s_qps += NQ / t_total_seconds;
            s_lat += (t_total_seconds * 1000.0f) / NQ; // 这里的 Latency 会受并发影响变大
        }
        std::cout << np << "," << s_rec/ROUND * 100.0F << "," << s_qps/ROUND << "," << s_lat/ROUND << "," << (m2-m1) << "\n";
    }
    return 0;
}