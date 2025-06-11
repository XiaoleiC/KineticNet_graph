# Kinetic Theory-informed Graph Learning for Traffic Forecasting


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
| KineticNet      | ???  |

Notice that Any Graph Convolution module can be plugged into the recurrent discrete temporal dynamic graph template to test performance; simply replace DiffConv or GaAN.
