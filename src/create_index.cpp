#define HIGH_ACC_FAST_SCAN
#define EIGEN_DONT_PARALLELIZE
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <sys/resource.h>
#include <cassert>

#include "index/IVF.hpp"
#include "utils/IO.hpp"
#include "utils/StopW.hpp"

// --- 1. 基础读取函数 ---

void load_ibin(const std::string& path, UintRowMat& mat) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "Open failed: " << path << std::endl; exit(1); }
    
    uint32_t n, d;
    in.read((char*)&n, 4); 
    in.read((char*)&d, 4); // 脚本 A 写入的是 n, 1
    
    mat.resize(n, d);
    // 直接批量读取到矩阵内存中，速度极快
    in.read((char*)mat.data(), (size_t)n * d * sizeof(uint32_t));
    
    if(!in) { std::cerr << "Read ibin failed: " << path << std::endl; exit(1); }
    std::cout << "[IO] Loaded .ibin: " << path << " (" << n << "x" << d << ")\n";
}


// 读取 .fbin (N * D bytes float)
void load_fbin(const std::string& path, FloatRowMat& mat) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "Open failed: " << path << std::endl; exit(1); }
    uint32_t n, d;
    in.read((char*)&n, 4); in.read((char*)&d, 4);
    mat.resize(n, d);
    in.read((char*)mat.data(), n * d * sizeof(float));
    if(!in) { std::cerr << "Read fbin failed (truncated?): " << path << std::endl; exit(1); }
    std::cout << "[IO] Loaded .fbin: " << path << " (" << n << "x" << d << ")\n";
}

// 读取 .i8bin (N * D bytes int8 -> float)
void load_i8bin(const std::string& path, FloatRowMat& mat) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "Open failed: " << path << std::endl; exit(1); }
    uint32_t n, d;
    in.read((char*)&n, 4); in.read((char*)&d, 4);
    mat.resize(n, d);
    std::vector<int8_t> buf(n * d);
    in.read((char*)buf.data(), n * d * sizeof(int8_t));
    if(!in) { std::cerr << "Read i8bin failed: " << path << std::endl; exit(1); }
    
    // Convert int8 -> float
    float* data_ptr = mat.data();
    for(size_t i=0; i<n*d; ++i) {
        data_ptr[i] = static_cast<float>(buf[i]);
    }
    std::cout << "[IO] Loaded .i8bin: " << path << " (" << n << "x" << d << ")\n";
}

// 读取 .fvecs (每行: [4byte dim] [dim * 4byte float])
// 通常用于 Centroids
void load_fvecs(const std::string& path, FloatRowMat& mat) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "Open failed: " << path << std::endl; exit(1); }
    
    // 获取文件大小以预估行数 (非必要，但为了 Resize 方便)
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    // 读取第一行获取维度
    int32_t d;
    in.read((char*)&d, 4);
    size_t row_size = 4 + d * 4;
    size_t n = file_size / row_size;

    mat.resize(n, d);
    in.seekg(0, std::ios::beg); // 回到开头

    for (size_t i = 0; i < n; ++i) {
        int32_t dim_check;
        in.read((char*)&dim_check, 4);
        if (dim_check != d) {
            std::cerr << "Error: fvecs dimension mismatch at row " << i << std::endl;
            exit(1);
        }
        in.read((char*)&mat(i, 0), d * sizeof(float));
    }
    std::cout << "[IO] Loaded .fvecs: " << path << " (" << n << "x" << d << ")\n";
}

// 读取 .ivecs (每行: [4byte dim] [dim * 4byte int]) -> 转换为 PID (uint32)
// 通常用于 Cluster IDs
void load_ivecs(const std::string& path, UintRowMat& mat) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "Open failed: " << path << std::endl; exit(1); }

    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    int32_t d;
    in.read((char*)&d, 4);
    size_t row_size = 4 + d * 4;
    size_t n = file_size / row_size;

    mat.resize(n, d);
    in.seekg(0, std::ios::beg);

    for (size_t i = 0; i < n; ++i) {
        int32_t dim_check;
        in.read((char*)&dim_check, 4);
        if (dim_check != d) { std::cerr << "Error: ivecs dim mismatch\n"; exit(1); }
        
        std::vector<int32_t> buf(d);
        in.read((char*)buf.data(), d * sizeof(int32_t));
        for(size_t j=0; j<(size_t)d; ++j) {
            mat(i, j) = static_cast<PID>(buf[j]);
        }
    }
    std::cout << "[IO] Loaded .ivecs: " << path << " (" << n << "x" << d << ")\n";
}

// --- 2. 智能分发加载器 ---
void load_data_auto(const std::string& path, FloatRowMat& mat) {
    if (path.find(".fbin") != std::string::npos) load_fbin(path, mat);
    else if (path.find(".i8bin") != std::string::npos) load_i8bin(path, mat);
    else if (path.find(".fvecs") != std::string::npos) load_fvecs(path, mat);
    else {
        std::cerr << "Unknown data extension: " << path << std::endl;
        exit(1);
    }
}

void load_centroids_auto(const std::string& path, FloatRowMat& mat) {
    if (path.find(".fbin") != std::string::npos) {
        load_fbin(path, mat);
    } else if (path.find(".fvecs") != std::string::npos) {
        load_fvecs(path, mat);
    } else {
        std::cerr << "Unknown centroids extension: " << path << std::endl;
        exit(1);
    }
}

void load_cids_auto(const std::string& path, UintRowMat& mat) {
    if (path.find(".ibin") != std::string::npos) {
        load_ibin(path, mat);
    } else if (path.find(".ivecs") != std::string::npos) {
        load_ivecs(path, mat);
    } else {
        std::cerr << "Unknown cids extension: " << path << std::endl;
        exit(1);
    }
}


// --- 3. Main ---
int main(int argc, char* argv[]) {
    // Usage: <data_file> <index_file> <K> <B> <centroids_file> <cids_file>
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " <data> <index> <K> <B> <centroids> <cids>\n";
        return 1;
    }

    std::string data_file = argv[1];
    std::string index_file = argv[2];
    size_t K = std::stoul(argv[3]);
    int B = std::stoi(argv[4]);
    std::string centroids_file = argv[5];
    std::string cids_file = argv[6];
    if (!(B == 9 || B == 5 || B == 7 || B == 3 || B == 4 || B == 8)) {
        std::cerr << "Warning: B=" << B << " is not in the original assert list (3,4,5,7,8,9).\n";
        // 不强制退出，防止脚本被意外中断，但给出警告
    }

    std::cout << "=== Create Index ===\n";
    std::cout << "Data: " << data_file << "\n";
    std::cout << "Params: K=" << K << ", B=" << B << "\n";

    FloatRowMat data;
    FloatRowMat centroids;
    UintRowMat cids;

    // 1. Load Data
    load_data_auto(data_file, data);

    // 2. Load Centroids (expect .fvecs)
    load_centroids_auto(centroids_file, centroids);

    // 3. Load CIDs (expect .ivecs)
    load_cids_auto(cids_file, cids);

    // 4. Verify Dimensions
    size_t N = data.rows();
    size_t DIM = data.cols();
    
    if (centroids.rows() != K) {
        std::cerr << "Error: Centroids count (" << centroids.rows() << ") != K (" << K << ")\n";
        exit(1);
    }
    if (centroids.cols() != DIM) {
        std::cerr << "Error: Centroids dim (" << centroids.cols() << ") != Data dim (" << DIM << ")\n";
        exit(1);
    }
    if (cids.rows() != N) {
        std::cerr << "Error: CIDs count (" << cids.rows() << ") != N (" << N << ")\n";
        exit(1);
    }

    std::cout << "Data check passed. Start construction...\n";

    StopW stopw;
    IVF ivf(N, DIM, K, B);
    
    // 注意：ExRaBitQ 的 construct 可能需要特定的内存布局
    // 这里的 data(), centroids.data() 都是指针，必须确保数据是 Row-Major 且连续
    // 我们使用的 FloatRowMat 应该是 std::vector 或者 Eigen::Matrix<..., RowMajor>
    // 上面的 load 函数保证了连续性
    ivf.construct(data.data(), centroids.data(), cids.data());
    
    float minutes = stopw.getElapsedTimeMili() / 1000.0f / 60.0f;
    std::cout << "IVF constructed in " << minutes << " minutes.\n";
    
    ivf.save(index_file.c_str());
    std::cout << "Index saved to " << index_file << "\n";

    return 0;
}