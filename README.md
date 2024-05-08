# Lux AI Challenge 🍄 🎮 DS598 Spring 2024

Team Gamma: Andy Yang, Wai Yuen Cheng, Tariq Georges

## Introduction
Our team approached this challenge by leveraging a U-Net-based neural network, specifically
designed to process spatial data and prioritize relevant information through skip connections. The
network was trained on replay data generously shared by the original competition-winning team, Toad Brigade. We also integrated our own replays, from Team GO and Team Q of our DS598 class tournament.
By analyzing and learning from this high-quality data, our model aimed to replicate and improve
upon the successful strategies employed by the winning team. 

## Train and test for yourself
As linked to in the report, the original training data can be downloaded from [Kaggle Datasets]([https://www.kaggle.com/datasets](https://www.kaggle.com/datasets/bomac1/luxai-replay-dataset)). For this repo, all replays are stored in the `full_episodes/top_agents` directory.
For training, if running on the BU SCC servers, please submit a job (with GPU) running the `run.sh` script. If running locally, pip install `requirements.txt`, navigate to `UNet_attention`, and run `train.py`.

## Result
Finally our team ended in fourth place, losing third place on the final day. Our highest obtained score was 1937.2.
Our agent trained with U-Net has become a smart game player!  

Based on: https://github.com/Epicato/lux-AI
