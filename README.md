# Discovery of Primary Prostate Cancer Biomarkers using Cross-Cancer Learning

### Introduction

This repository is for our submitted paper for Scientific Reports '[Discovery of Primary Prostate Cancer Biomarkers
using Cross-Cancer Learning]'. The code is modified from [DeePathology](https://github.com/SharifBioinf/DeePathology). 

### Installation
This repository is based on Tensorflow 2.2.0

For installing tensorflow, please follow the official instructions in [here](https://www.tensorflow.org/install/install_linux). The code is tested under Python 3.6 on Ubuntu 18.04.

### Data
Our prepared data can be downloaded from [CCL-Discovery(data)](https://drive.google.com/file/d/1evJ7J4M7U8TsU_lujKUBq6ROYS9d7pZR/view?usp=sharing). Put all files in this folder to `data_process` folder in the root directory.
### Usage

1. Setup the parameters accordingly in `option.py`
   
2. Train the model for our autoencoder to obtain SHAP scores.
    Run:
   ```shell
   cd code
   python mlc-ae.py --phase train
   ```

3. Test the model of autoencoder and draw the SHAP visualization.
    Run:
   ```shell
   cd code
   python mlc-ae.py --phase test
   ```  
   
4. Train the model for our evaluation classifier, in where we have attached sample score files.
    Run:
   ```shell
   cd code
   python eval-classifier.py --phase train
   ```
   
5. Test the model for our evaluation classifier.
    Run:
   ```shell
   cd code
   python eval-classifier.py --phase test
   ```
