#pragma once
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "tensor/tensor.h"

namespace tinyinfer {

// ---------------------------------------------------------------------------
// IDataset<Datum> — abstract base for indexed datasets
// ---------------------------------------------------------------------------

template <typename Datum>
class IDataset {
public:
    virtual const Datum& operator[](size_t idx) const = 0;
    virtual size_t size() const = 0;
    virtual ~IDataset() = default;
};

// ---------------------------------------------------------------------------
// MNISTSample
// ---------------------------------------------------------------------------

struct MNISTSample {
    Tensor image;   // 1D CPU tensor, shape {784}, values in [0, 1]
    int label = -1;
};

// ---------------------------------------------------------------------------
// MNISTDataset
// ---------------------------------------------------------------------------

class MNISTDataset : public IDataset<MNISTSample> {
public:
    // Load images + labels from IDX binary files directly
    MNISTDataset(const std::string& images_path,
                 const std::string& labels_path);

    const MNISTSample& operator[](size_t idx) const override;
    size_t size() const override { return samples_.size(); }

    // Convenience: discover files under data_dir automatically
    static MNISTDataset test(const std::string& data_dir);
    static MNISTDataset train(const std::string& data_dir);

private:
    static uint32_t read_be_uint32(std::FILE* f);
    std::vector<MNISTSample> samples_;
};

}  // namespace tinyinfer
