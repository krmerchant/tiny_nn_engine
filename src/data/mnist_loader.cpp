#include "data/mnist_loader.h"
#include <cstring>
#include <stdexcept>

namespace tinyinfer {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

uint32_t MNISTDataset::read_be_uint32(std::FILE* f) {
    uint8_t buf[4];
    if (std::fread(buf, 1, 4, f) != 4)
        throw std::runtime_error("MNISTDataset: unexpected EOF reading header");
    return (uint32_t(buf[0]) << 24) | (uint32_t(buf[1]) << 16) |
           (uint32_t(buf[2]) << 8)  |  uint32_t(buf[3]);
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

MNISTDataset::MNISTDataset(const std::string& images_path,
                            const std::string& labels_path) {
    // --- images ---
    std::FILE* img_f = std::fopen(images_path.c_str(), "rb");
    if (!img_f)
        throw std::runtime_error("MNISTDataset: cannot open " + images_path);

    uint32_t magic = read_be_uint32(img_f);
    if (magic != 0x00000803)
        throw std::runtime_error("MNISTDataset: bad image magic in " + images_path);
    uint32_t n_images = read_be_uint32(img_f);
    uint32_t rows     = read_be_uint32(img_f);
    uint32_t cols     = read_be_uint32(img_f);
    uint32_t n_pixels = rows * cols;

    // --- labels ---
    std::FILE* lbl_f = std::fopen(labels_path.c_str(), "rb");
    if (!lbl_f) {
        std::fclose(img_f);
        throw std::runtime_error("MNISTDataset: cannot open " + labels_path);
    }

    uint32_t lbl_magic = read_be_uint32(lbl_f);
    if (lbl_magic != 0x00000801)
        throw std::runtime_error("MNISTDataset: bad label magic in " + labels_path);
    uint32_t n_labels = read_be_uint32(lbl_f);

    if (n_images != n_labels)
        throw std::runtime_error("MNISTDataset: image/label count mismatch");

    samples_.reserve(n_images);

    std::vector<uint8_t> pixel_buf(n_pixels);
    for (uint32_t i = 0; i < n_images; ++i) {
        if (std::fread(pixel_buf.data(), 1, n_pixels, img_f) != n_pixels)
            throw std::runtime_error("MNISTDataset: unexpected EOF reading image data");

        uint8_t label_byte;
        if (std::fread(&label_byte, 1, 1, lbl_f) != 1)
            throw std::runtime_error("MNISTDataset: unexpected EOF reading label data");

        // Build a 1D CPU tensor of shape {n_pixels}
        std::vector<float> floats(n_pixels);
        for (uint32_t p = 0; p < n_pixels; ++p)
            floats[p] = pixel_buf[p] / 255.f;

        MNISTSample sample;
        sample.image = Tensor(floats, {static_cast<int64_t>(n_pixels)});
        sample.label = static_cast<int>(label_byte);
        samples_.push_back(std::move(sample));
    }

    std::fclose(img_f);
    std::fclose(lbl_f);
}

// ---------------------------------------------------------------------------
// operator[]
// ---------------------------------------------------------------------------

const MNISTSample& MNISTDataset::operator[](size_t idx) const {
    if (idx >= samples_.size())
        throw std::out_of_range("MNISTDataset: index out of range");
    return samples_[idx];
}

// ---------------------------------------------------------------------------
// Static factories
// ---------------------------------------------------------------------------

MNISTDataset MNISTDataset::test(const std::string& data_dir) {
    return MNISTDataset(data_dir + "/t10k-images-idx3-ubyte",
                        data_dir + "/t10k-labels-idx1-ubyte");
}

MNISTDataset MNISTDataset::train(const std::string& data_dir) {
    return MNISTDataset(data_dir + "/train-images-idx3-ubyte",
                        data_dir + "/train-labels-idx1-ubyte");
}

}  // namespace tinyinfer
