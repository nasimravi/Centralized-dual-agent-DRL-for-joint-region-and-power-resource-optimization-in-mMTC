This repository contains the Python code for the paper titled "Centralized-dual-agent-DRL-for-joint-region-and-power-resource-optimization-in-mMTC", published in Computer Networks. The manuscript is available on (https://doi.org/10.1016/j.comnet.2026.112262).

# Abstract
Achieving energy-efficient transmission with high decoding reliability is a fundamental challenge for massive machine-type communication (mMTC) using grant-free Non-Orthogonal Multiple Access (NOMA), due to dense device activity, sporadic traffic, and strong uplink interference. This paper introduces a Centralized Dual-Agent Deep Reinforcement Learning (CDA-DRL) framework that jointly optimizes region partitioning and transmit power pool design in uplink grant-free NOMA. Two cooperative agents at the base station independently learn the number of spatial regions and the number of power levels, respectively, using recurrent Deep Q-Networks under a centralized training and decentralized execution paradigm. This factorized architecture reduces the action-space complexity and enables scalable learning. Simulation results demonstrate that CDA-DRL achieves more stable training, higher Successive Interference Cancellation (SIC) decoding success, and significantly improved energy efficiency, outperforming geometric gap-based baselines by up to 114% and fixed-power schemes by 33%.

## Citation
If you use this code in your work, please cite our paper: 

Ravi, Nasim, Nuno Lourenço, and Marilia Curado. "Centralized Dual-Agent DRL for Joint Region and Power Resource Optimization in mMTC." Computer Networks (2026): 112262.

bibtex:

@article{ravi2026centralized,
  title={Centralized Dual-Agent DRL for Joint Region and Power Resource Optimization in mMTC},
  author={Ravi, Nasim and Louren{\c{c}}o, Nuno and Curado, Marilia},
  journal={Computer Networks},
  pages={112262},
  year={2026},
  publisher={Elsevier}
}
