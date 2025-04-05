# PG-DScheduler

Online Scheduling for Distributed DNN Training Tasks in resource-constrained edge clusters

## Requirements

```
Python>=.6.2
tensorflow>=1.13.1
```

## Dataset Construction

```
cd dnn2/dag
python main.py
```

## Quick Start

```
python train.py --num_init_dags 3 --num_stream_dags 15 --num_agents 16 --model_save_interval 100 --num_ep 2000 --model_folder ./models/init_3_stream_15_16_100_2000/

```

## Thanks

The code refers to the repo [decima](https://github.com/hongzimao/decima-sim)
