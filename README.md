# Arrhythmias Analysis (ECG) — Machine Learning Research 

This repository contains the research paper **"Arrhythmias Analysis with Efficient Machine Learning Classification Algorithms"** by Holly Kwan, plus extracted R scripts used in the Appendix.

## 📄 Paper
- **Title:** Arrhythmias Analysis with efficient machine learning classification algorithms
- **Author:** Holly Kwan 
- **TXT:** 'Electrocardiogram.txt' 
- **PDF:** `Electrocardiogram.pdf`

## Files in this repository
```
Electrocardiogram.txt
Rscript/
  ├── svm_features.R
  ├── svm_signals.R
  ├── convnet_features.R
  └── convnet_signals.R
```

## How to run the R scripts (basic)
Open R or RStudio in the repository root and run (example):
```r
# install required packages (one-time)
install.packages(c('e1071','readr','class','dplyr','devtools','reticulate','remotes'))

# For ConvNet, in R run:
devtools::install_github('rstudio/keras')
remotes::install_github('rstudio/tensorflow')
library(keras)
# install_keras()  # run if you need to install TensorFlow/Keras backend

# Run SVM (Features)
source('code/svm_features.R')

# Run SVM (Signals)
source('code/svm_signals.R')

# ConvNet scripts contain example model code and require data preprocessing to shape arrays/matrices.
# See convnet_features.R and convnet_signals.R for details.
```

## Repo setup & push 
```bash
# create local folder and move files into it, or download files from this session to your machine
mkdir ecg-ml-paper
cd ecg-ml-paper

git init
git add .
git commit -m "Initial commit: research ECG paper + R scripts"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/REPO-NAME.git
git push -u origin main
```

## 🔗 Datasets Mentioned (as per paper)

> This repository mirrors the paper’s description; no new claims are introduced here.

## 📚 Citation
If you reference this work, please cite the paper PDF in this repository.

