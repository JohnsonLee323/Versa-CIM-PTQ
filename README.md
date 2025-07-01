# Versa-CIM-PTQ
Versa-CIM-PTQ is a post-training quantization (PTQ) framework specifically developed to support compute-in-memory (CIM) architectures. This framework offers the flexibility to adaptively adjust the quantization precision across different network layers and can automatically search for the optimal Exponent/Mantissa Ratio (EMR) under a given quantization precision based on the MSE distance between tensors.
# Install
## Requirement
First, make sure to have **Python ≥3.12** (tested with Python 3.12.9) and ensure the latest version of pip (tested with 25.0), next install the remaining dependencies using pip:
```bash
pip install -r requirements.txt
```
## Datasets
To run example testing, you should download ImageNet2012 dataset for resnet-50, wikitext2 for gpt-2.
Then you should add the path of dataset to the corresponding configuration *.yaml* file:
 ```bash
DATA:
   DATA_PATH = your_dataset_path
```
# Usage
## Evaluate
To evaluate quantized resnet-50 on imagenet2012, run
```bash
python ptq.py --cfg configs/resnet_imagenet.yaml
```
To evaluate quantized gpt-2 on wikitext2, run
```bash
python ptq.py --cfg configs/gpt2_wikitext2.yaml
```
## Configuration
you can change the configuration in `configs` folder, below I will introduce some important configurable parameters:
`FP32`: you can get full-precision 
`N_BITS`: The globl quantization precision for entire net
