# snn_inference
# Spiking Neural Network Policy on DROID Dataset with FPGA (HLS)

This project trains a small Spiking Neural Network (SNN) on the [DROID dataset](https://droid-dataset.github.io/),
exports the trained weights, and implements inference on an FPGA using Vitis HLS.

---

## Setup
### Python Environment
- Python 3.10
- PyTorch
- snntorch
- Numpy

-Install with:
```bash
pip install torch snntorch numpy

#FPGA Tools:
-AMD Vitis HLS 2025.1


#Training:

-Run the training script:
python train/train_snn.py

#This saves:

-snn_weights.pth
-normalization.json

#Export weights and headers for FPGA:

-python export/export_weights_csv.py
-python export/make_headers_from_csv.py


##FPGA (HLS) Inference:

-Open Vitis HLS.

-Add sources: snn_top.cpp, snn_top.h, and the generated headers.

-Add testbench: tb_snn.cpp.

-Set top function = snn_inference.

-Run C Simulation → prints the output.

-Run C Synthesis → shows latency & resource usage.


#Results:
#C Simulation Output:

0.466240  -0.023169  0.310902  -0.259587  -0.007338  -0.383760  0.108342


#Latency:

2687 cycles ≈ 26.9 µs @ 100 MHz

Interval: 2688 cycles

#Resource Utilization (Artix-7 target)

LUT: 13,820 (172%)

FF: 18,994 (118%)

DSP: 162 (405%)

BRAM: 17 (42%)

#Exceeds capacity of this small FPGA, but demonstrates full flow from training → export → FPGA inference
