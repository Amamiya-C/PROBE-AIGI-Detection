# PROBE: Probing Robustness via Boundary Exploration

This repository contains the model weights for the paper:

**"Where Detectors Fail: Probing Generative Space for Generalizable AI-Generated Image Detection"**

*Zijie Cao, Weijie Tu, Yao Xiao, Weijian Deng, Weiyan Chen, Liang Lin, Pengxu Wei*

Accepted by **ICML 2026** 


## Abstract
Detecting AI-generated images (AIGI) remains challenging because detectors often fail to generalize to unseen generators. Although existing methods are trained on large datasets, their performance still degrades when generation settings change, indicating that data scale alone is insufficient and that limited coverage of generative variations during training is a key factor.
Studies on generative model editing show that small changes in internal representations can produce diverse and meaningful image variations, many of which are not explored under standard sampling.
Leveraging this insight, we propose PROBE (Probing Robustness via Boundary Exploration), a framework that improves detector generalization by actively exploring challenging regions of the generative process. Instead of treating the generator as a fixed data source, \name uses the detector as a critic to steer the generator through manifold-level modifications, producing realistic samples that are difficult to classify. 
These samples expose failure cases that are uncommon under standard data sampling strategies and are used to refine the detector.
Experimental results across multiple benchmarks indicate that PROBE enhances generalization to unseen generators, resulting in more generalizable AIGI detection performance.

## Model Weights

We release the following detector checkpoints fine-tuned with PROBE:

| Model | Backbone | Training Data | Avg. bAcc (7 benchmarks) |
|-------|----------|---------------|--------------------------|
| PROBE-ResNet50 | ResNet-50 | GenImage SD 1.4 + PROBE samples | 78.1% |
| PROBE-DINOv2 | DINOv2 | Reconstruction Training Set + PROBE samples | 93.9% |


## Evaluation Benchmarks

The models are evaluated on seven benchmarks covering both in-house and in-the-wild scenarios:

- **GenImage**
- **Synthbuster** 
- **AIGI-Quality-Paradox**
- **Chameleon** 
- **SynthWildX** 
- **WildRF** 
- **AIGI-Bench**

