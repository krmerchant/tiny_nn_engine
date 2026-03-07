#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace tinyinfer {

struct MNISTSample {
    std::vector<float> image;  // 784 floats in [0,1]
    int label = -1;
};

struct MNISTDataset {
    std::vector<MNISTSample> samples;
    size_t size() const { return samples.size(); }
};

class MNISTLoader {
public:
    // Load images + labels from IDX binary files
    // images_path: e.g. "t10k-images-idx3-ubyte"
    // labels_path: e.g. "t10k-labels-idx1-ubyte"
    static MNISTDataset load(const std::string& images_path,
                             const std::string& labels_path);

    // Convenience: discover files under data_dir automatically
    static MNISTDataset load_test(const std::string& data_dir);
    static MNISTDataset load_train(const std::string& data_dir);

private:
    static uint32_t read_be_uint32(std::FILE* f);
};

}  // namespace tinyinfer
