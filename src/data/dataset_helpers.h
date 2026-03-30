#pragma once

#include "mnist_loader.h"
#include "tensor/tensor.h"
namespace tinyinfer {

template <typename Datum>
Tensor create_batch(const IDataset<Datum> &data, int offset, int batch_size) {

  const int64_t feat =
      data[0].image.shape()[0]; // 784 -- assumes dataset all same size
  // Stack n images into [n, feat]
  Tensor batch({(int64_t)batch_size, feat});
  for (size_t j = 0; j < batch_size; ++j)
    std::memcpy(batch.data_ptr() + j * feat, data[offset + j].image.data_ptr(),
                feat * sizeof(float));

  return batch;
}
} // namespace tinyinfer
