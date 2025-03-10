# wab-3-log-anomaly-detection

### Purpose

This Repository is meant to go along with my Term Paper in which I am conducting a comparative study of log anomaly detection models using reconstruction and representation learning, respectively.

### Implemented Models / Papers

The first model is AutoLog[[1]](#references). An Autoencoder, which plays the role of the reconstuction learning part in my paper. It allows for Semi-Supervised Learning and detects anomalous Log Timeframes by setting a Threshold on the Reconstruction Error of the Model's output. (The author's official implementation can be found [here](https://github.com/ScalingLab/AutoLog)).

The second model is CLDTLog[[2]](#references). A fine-tuned version of the BERT[[3]](#references) Large Language Model developed by Google. It is trained in a Supervised setting, using Triplet Loss as well as Focal Loss in order to seperate log embeddings and address the class imbalance of normative vs anomalous samples.

Both Models are trained and tested on the public Loghub[[4]](#references) Blue Gene/L Supercomputer Log Dataset which contains log data labeled as either normative or anomalous. (Can be found [here](https://github.com/logpai/loghub/tree/master/BGL)).

### Research Goal

The goal of this paper is comparison of the AutoLog and CLDTLog models in terms of anomaly detection performance, measured by their Recall, Precision and F1 Score, as well as their efficiency, measured by the model's throughput, in real-world log data.

### Set-Up Instructions


> [!WARNING]
> Code was exclusively developed and tested in Python Version 3.13.2 and might not work on older Versions.
>
> The Installation Script `install_deps.bat` is executable on Windows. If you wish to run this project on another Operating System, please check out the Source code and manually replicate the performed steps.
>
> If you want to run the code yourself you will need an NVIDIA GPU. If you wish to train and evaluate the models on your CPU, you will have to alter the source code a bit yourself since tensor / model conversions to CUDA are hard-coded in some places in the code.

1. Clone this repository
    1. By cloning using git in the command-line: `git clone https://github.com/fietensen/wab-3-log-anomaly-detection.git`
    2. By downloading and extracting the ZIP-File when clicking on the `<> Code` Button at the top of this Page

2. Install Python (Version `3.13.2`) from [python.org](https://www.python.org/ftp/python/3.13.2/python-3.13.2-amd64.exe)

3. Install the NVIDIA Cuda Toolkit (Version `11`) from [developer.nvidia.com](https://developer.nvidia.com/cuda-downloads)

4. Install NVIDIA CuDNN (Version `10`) from [developer.nvidia.com](https://developer.nvidia.com/cudnn-downloads)

5. Install Python Dependencies and run the Model
    1. Press the `WIN+R` Keys on your Keyboard to open a Command Prompt
    2. Navigate to the cloned Repository by executing `cd C:\Path\To\wab-3-log-anomaly-detection\`
    3. Install prerequisites by executing `.\install_deps.bat`
    4. Activate the created virtual environment: `.\venv\Scripts\activate.bat`
    5. Run the model training and evaluation script: `python -m model_eval`

### References
[[1]](#references) Catillo, M., Pecchia, A., & Villano, U. (2022).  
**AutoLog: Anomaly Detection by Deep Autoencoding of System Logs**.  
*Expert Systems with Applications, 191*, 116263. [https://doi.org/10.1016/j.eswa.2021.116263](https://doi.org/10.1016/j.eswa.2021.116263)

[[2]](#references) Tian, G., Luktarhan, N., Wu, H., & Shi, Z. (2023).  
**CLDTLog: System Log Anomaly Detection Method Based on Contrastive Learning and Dual Objective Tasks**.  
*Sensors, 23*(11), 5042. [https://doi.org/10.3390/s23115042](https://doi.org/10.3390/s23115042)  


[[3]](#references) Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019).
**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**.
*arXiv preprint*, arXiv:1810.04805. [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

### Further Credits
This repository makes use of Code from these external sources (libraries excluded):

*@AktGPT* - Fast Online Triplet mining in Pytorch:
- https://github.com/aktgpt/onlinetripletmining

Implements classes for efficiently performing semi-hard Triplet Mining.

---

*@rjnclarke* - Fine-Tune an Embedding Model with Triplet Margin Loss in PyTorch:
-  https://medium.com/@rjnclarke/fine-tune-an-embedding-model-with-triplet-margin-loss-in-pytorch-62bf00865a6c

Implements a Batch Sampler ensuring Batches for CLDTLog always contain both, normative and anomalous samples so the Triplets can be mined in an online fashion. The Sampler was slightly altered for this implementation.

### Figures

![KDEPlot of Logging-Entity Scores for the AutoLog Model](https://github.com/fietensen/wab-3-log-anomaly-detection/blob/main/figures/autolog_scores_kde.png?raw=true)

*KDEPlot of Logging-Entity Scores for the AutoLog Model*

![Training and Validation Loss Curve for the AutoLog Model](https://github.com/fietensen/wab-3-log-anomaly-detection/blob/main/figures/autolog_loss_curve.png?raw=true)

*Training and Validation Loss Curve for the AutoLog Model (50 epochs)*

![Training and Validation Loss Curve for the CLDTLog Model](https://github.com/fietensen/wab-3-log-anomaly-detection/blob/main/figures/cldtlog_loss_curve.png?raw=true)

*Training and Validation Loss Curve for the CLDTLog Model (10 epochs)*

![Model Metrics Comparison](https://github.com/fietensen/wab-3-log-anomaly-detection/blob/main/figures/model_metric_comparison.png?raw=true)

*Comparison of AutoLog and CLDTLog Metrics: F1 Score, Precision & Recall*

![Model Throughput Comparison](https://github.com/fietensen/wab-3-log-anomaly-detection/blob/main/figures/model_throughput_comparison.png?raw=true)

*Comparison of AutoLog and CLDTLog Throughput: Per Sample in Milliseconds; Log Scale*