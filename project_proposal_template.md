# Project Proposal

**Student Name:** Komel Merchant
**Date:**

---

## Tiny NNEngine  


---

## Objective

_In 2–4 sentences, describe what you are building and what problem or goal it addresses. Be specific about the computational task you plan to accelerate or implement on the GPU._

My plan is to create a basic TensorRT-Like Engine. The Program will primiarly be written in C++ while the core low-level computations will be carried out in CUDA. The Engine will be able to consume an ONNX file containing a simple pretrained model (I will aim for usin a simple MLP trained to predict digits from the MNIST dataset. This ONNX file will be generated from PyTorch.


---

## Programming Platform

- [x ] CUDA
- [ ] OpenCL


---

## Hypothesis / Expected Outcome

_What do you expect to achieve? Consider addressing:_
- _The performance gains or behavior you anticipate (e.g., speedup over a CPU baseline)_
   * The two important statistics I will be optimizing for are acurracy and speed. 
        * I expect my accuracy will be identical to that of the standard PyTorch model. This ensures correctness. 
        * I expect my speed to be better than PyTorch CPU JIT, but potentially less/equal perforamane to PyTorch CUDA JIT
- _Any technical challenges you expect to encounter_
    * I will need to read up a bit on how to create graph strucutre
- _How you will measure success_
    * Compute Accuracy accuracy across 

---

## Deliverables Overview

| Deliverable | Description |
|---|---|
| Code | Working GPU implementation of the proposed project |
| Video Presentation | Class presentation recorded as a video |
| Final Report | Written report covering design, implementation, and results |
| Code Review (Host) | Host one code review of your own code |
| Code Reviews (Peer) | Participate in at least two other students' code reviews |

---

_Note: This proposal is intentionally brief. Save detailed design decisions for your final report._
