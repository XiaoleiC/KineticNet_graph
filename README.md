# Kinetic Theory-informed Graph Learning for Traffic Forecasting
## DGL Implementation of KineticNet (ours), DCRNN and GaAN paper.

Verify our proposed KineticNet. We also implement the GNN model proposed in the paper [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://arxiv.org/abs/1707.01926) and [GaAN:Gated Attention Networks for Learning on Large and Spatiotemporal Graphs](https://arxiv.org/pdf/1803.07294). Still under testing.

The graph dataset used in this example 
---------------------------------------
METR-LA dataset. Dataset summary:
- NumNodes: 207
- NumEdges: 1722
- NumFeats: 2
- TrainingSamples: 70%
- ValidationSamples: 20%
- TestSamples: 10%

PEMS-BAY dataset. Dataset Summary:

- NumNodes: 325
- NumEdges: 2694
- NumFeats: 2
- TrainingSamples: 70%
- ValidationSamples: 20%
- TestSamples: 10%

Performance on METR-LA
-------------------------
| Models/Datasets | Test MAE |
| :-------------- | --------:|
| DCRNN in DGL    | 2.91 |
| DCRNN paper     | 3.17 |
| GaAN in DGL     | 3.20 |
| GaAN paper      | 3.16 |
| KineticNet      | 4.00 |

*Currently the number of trainable parameters in KineticNet is only 1/7 of DCRNN and 1/20 of GaAN, I am fine-tuning.*

Notice that Any Graph Convolution module can be plugged into the recurrent discrete temporal dynamic graph template to test performance; simply replace DiffConv or GaAN.
