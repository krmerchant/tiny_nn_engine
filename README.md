# Tiny NNgine

A toy CUDA-based Neural Network Inference SDK.
## Description

The project is inspired at a high level by PyTorch. The primarily library componenets are: 

* `Tensor` 
	* This is responsible for both holding executing all tensor operations. For the scope of this project, I only support `GEMM`,`softmax` `matsum` operations. 
		* This Tensor object contains a `TensorStorage` interface class, which actually holds the data as a flat vector and dispatches operation to CPU and GPU depending on the derived type (eg. `CPUStorage` and `GPUStorage`) 
* `Model`class
	* Responsible for loading in a `model.onnx` file and generating the underlying `Graph` object. This `Graph` is a topologically sorted DAG, which each `Node` represent an operation in the NN model network
* `Executor` class
	* This is responsible for iterating through the `Model`'s `Graph` and executing each operation
* `Evaluator` class 
	* This is



## Getting Started

### Dependencies

This application is assumed to run on a linux machine with a CUDA GPU. I've only tested on the following NVIDIA archetecutres: 

Dependencies to install

### Build

Run CMake build command
```  sh 
  mkdir -p build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release
  make -j$(nproc) tinyinfer_demo
```

### Executing program

Generate a ONNX model
``` sh
cd tools/
python3 generate_mnist_onnx.py --output mnist_fc.onnx --data-dir ./data --epochs 10
```

 Demo application to load model, data and run inference
``` sh
  ./build/tinyinfer_demo \
    -m tools/mnist_fc.onnx \
    -i tools/data/MNIST/raw/t10k-images-idx3-ubyte \
    -l tools/data/MNIST/raw/t10k-labels-idx1-ubyte \
    -b 64 # batch size to use for inference

```
## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

* Komel Merchant 

## Version History



## License

This project is licensed under the AGPL License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)