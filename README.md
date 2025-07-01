 # Versa-CIM-PTQ
Versa-CIM-PTQ is a post-training quantization (PTQ) framework for Versatile compute-in-memory (VersaCIM) architecture. This framework offers the flexibility to adaptively adjust the quantization precision across different network layers. Based on the MSE distance between tensors, Versa-CIM-PTQ firstly balances the data range and resolutions with arbitrary Exponent-Mantissa Ratio (EMR), enabling the efficient low-precision FP with uphold accuracy. It significantly pushes the envelope of the trade-off between performance and accuracy.
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

`FP32`: Full-precision or quantized model

`N_BITS`: The globl quantization precision for entire modle (can be replaced by specific configuration)

`EXPONENT`: The default exponent bit for FP format

`EMMX`: If True, framework will automatically search for optimal EMR under a given quantization precision

`BIAS_MX`: If EMMX is set to False, this parameter will work to decide whether bias for fp format is searched by framework or default

`SPECIFIC_CFG_PATH`: Specific quantization precision configuration path in specific_cfg folder


In specific_cfg folder, you can set quantization precision for specific layers, and you can adjust module whether to quantized in `quantization/net_wrap.py`.


Our SNN for VAD is based on an open-souce cnn model (https://gitee.com/ooooooooya/VAD_tutorial) and a cnn-to-snn tool (SpikingJelly: https://github.com/fangwei123456/spikingjelly/tree/master).
