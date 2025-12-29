# OCL_FEN
Feature Enhancement Network for Online Continual Learning.  A robust approach to mitigate catastrophic forgetting in non-stationary environments via feature enhancement.(Han et al., 2025).

#  Introduction
Abstract
Current artificial intelligence systems have shown excellent performance on the tasks at hand. However, catastrophic forgetting occurs when they learn new tasks. The field of continual learning (CL) investigates the ability to learn a sequence of tasks while being able to deal with earlier tasks well. Typically, current models learn a single latent representation based on the class label for each task in this context. To overcome this, we propose a novel approach, called Online Continual Learning via the Feature Enhancement Network (OCLFEN), which integrates an auto-encoder framework to learn example-level representation features using reconstruction loss from the data itself. This strategy enhances latent discriminative representations. Additionally, we incorporate relationship distillation loss to transfer similarity relationships between current and previous models, further mitigating catastrophic forgetting. Extensive experiments on various benchmarks and under different conditions demonstrate that OCLFEN proposed in this paper outperforms state-of-the-art algorithms, highlighting its effectiveness and robustness in continual learning scenarios.

#  Requirements
1)Python 3.6+

2)Torch 2.0+

#  Benchmarks
Data processing is based on the online_Continual_Learning_Datasets_Repository.

#  License
This project is released under the AGPL-2.0 license .
