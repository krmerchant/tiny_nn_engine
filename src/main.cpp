#include "model/model.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx>\n";
        return 1;
    }
    auto model = tinyinfer::Model::load(argv[1]);
    model.print_graph();
    return 0;
}
