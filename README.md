# RSRAE
Robust Subspace Recovery Layer for Unsupervised Anomaly Detection

## Getting Started
This repo implements the main experiments of our ICLR 2020 paper: https://openreview.net/forum?id=rylb3eBtwr

### Prerequisites
- python
- tensorflow
- keras
- scikit-learn

Run the following to get the results in the paper.
```
python experiments.py -t caltech101 -l l21 -q 1 -r 5 -m 10 -z 1
python experiments.py -t fashion -l l21 -q 1 -r 5 -m 10 -z 1
python experiments.py -t 20news -l l21 -q 1 -r 5 -m 10 -z 1
python experiments.py -t reuters -l l21 -q 1 -r 5 -m 10 -z 1
```

Links for preprocessed data:
Caltech101: https://drive.google.com/open?id=1qSGEy3EMwD7XcndMjJvb-TexAaDwWBy8
20 Newsgroup: https://drive.google.com/open?id=1Cc53BQF39XT0VWJhDTaRBUopQ6Vg75q_
Reuters: https://drive.google.com/open?id=11Jg7UewpVRwp40V5yTuKeeARS0K2XMzF
Tiny Imagenet: https://drive.google.com/open?id=1jfceyTY4SSBxxHE2mohy2rwSjmCZTgNE

## Citation

The following is the bibtex for citation.
```
@inproceedings{lai2020robust,
title={Robust Subspace Recovery Layer for Unsupervised Anomaly Detection},
author={Chieh-Hsin Lai and Dongmian Zou and Gilad Lerman},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=rylb3eBtwr},
}
```
