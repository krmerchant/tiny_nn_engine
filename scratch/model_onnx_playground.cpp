#include "model/model.h"


int main(int argc, char** argv) {

    tinyinfer::Model model = tinyinfer::Model::load("/home/komelmerchant/Dropbox/JHUCourseTracking/IntroToGPU/tiny_nn_engine/tools/mnist_fc.onnx"); 
    model.print_graph();
    return 1;
}
