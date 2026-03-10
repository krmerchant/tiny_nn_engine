#include "tensor/tensor.h"
#include "tensor/tensor_functions.h"
#include <cstdio>

int main() {
    tinyinfer::Tensor a = tinyinfer::zeros({4, 3, 2, 5});
    a.fill(2.0f);
    a.cuda();

    tinyinfer::Tensor b = tinyinfer::zeros({4, 3, 2, 5});
    b.fill(4.0f);
    b.cuda();

    tinyinfer::Tensor c = a + b;

    printf("c[1][2][0][3] = %f\n", c.at({1, 2, 0, 3}));

    return 0;
}
