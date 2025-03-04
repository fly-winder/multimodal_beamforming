# Sensing-Assisted High Reliable Communication: A Transformer-Based Beamforming Approach

This repository contains the implementation of our research on multimodal learning-based beamforming. This work is based on the research paper: **[Sensing-Assisted High Reliable Communication: A Transformer-Based Beamforming Approach](https://ieeexplore.ieee.org/document/10539181)**.

## 📂 Repository Structure

- `Data_Augmentation/` - Implements data augmentation techniques for improving model generalization.
- `Data_Preprocessing/` - Scripts for preprocessing raw dataset before training.
- `Dataset/` - Directory to store the dataset.
- `log/test/` - Stores log files.
- `args.txt` - Stores argument configurations for training and testing.
- `config_seq.py` - Configuration file for model and training hyperparameters.
- `data.py` - Data loading and processing utilities.
- `main.py` - Main script for training and evaluating the model.
- `model.py` - Defines the neural network architecture.
- `scheduler.py` - Learning rate scheduler for training.

## 📥 Dataset Download

The dataset required for training and evaluation can be downloaded from the following link:

🔗 [Dataset Download](https://drive.google.com/drive/folders/1zvOOJpGodEnjqvAiAeXkzOdjWmz1semF)

After downloading, extract and place the dataset inside the `Dataset/` folder.


## 📌 Notes
- Ensure the dataset is correctly placed in the `Dataset/` folder before running the scripts.
- Logs and training results will be stored in the `log/test/` directory.


## 📜 Citation
If you find this work useful, please consider citing our paper:
```bibtex

@ARTICLE{10539181,
  author={Cui, Yuanhao and Nie, Jiali and Cao, Xiaowen and Yu, Tiankuo and Zou, Jiaqi and Mu, Junsheng and Jing, Xiaojun},
  journal={IEEE Journal of Selected Topics in Signal Processing}, 
  title={Sensing-Assisted High Reliable Communication: A Transformer-Based Beamforming Approach}, 
  year={2024},
  doi={10.1109/JSTSP.2024.3405859}
}
```
