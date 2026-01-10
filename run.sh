#!/bin/sh

# 1. Performance Evaluation

python train.py Matrix-IPD DRIVE-TD identity
python train.py Matrix-IPD MATE-TD identity
python train.py Matrix-IPD LIO identity
python train.py Matrix-IPD IAC identity

python train.py CoinGame-2 DRIVE-TD identity
python train.py CoinGame-2 MATE-TD identity
python train.py CoinGame-2 LIO identity
python train.py CoinGame-2 IAC identity

python train.py CoinGame-4 DRIVE-TD identity
python train.py CoinGame-4 MATE-TD identity
python train.py CoinGame-4 LIO identity
python train.py CoinGame-4 IAC identity

python train.py Harvest-12 DRIVE-TD identity
python train.py Harvest-12 MATE-TD identity
python train.py Harvest-12 LIO identity
python train.py Harvest-12 IAC identity

# 2. Robustness against Reward Drift

# Linear Drift

python train.py Matrix-IPD DRIVE-TD linear
python train.py Matrix-IPD MATE-TD linear
python train.py Matrix-IPD LIO linear
python train.py Matrix-IPD IAC linear

python train.py CoinGame-2 DRIVE-TD linear
python train.py CoinGame-2 MATE-TD linear
python train.py CoinGame-2 LIO linear
python train.py CoinGame-2 IAC linear

python train.py CoinGame-4 DRIVE-TD linear
python train.py CoinGame-4 MATE-TD linear
python train.py CoinGame-4 LIO linear
python train.py CoinGame-4 IAC linear

python train.py Harvest-12 DRIVE-TD linear
python train.py Harvest-12 MATE-TD linear
python train.py Harvest-12 LIO linear
python train.py Harvest-12 IAC linear

# Exponential Decay

python train.py Matrix-IPD DRIVE-TD exponential_decay
python train.py Matrix-IPD MATE-TD exponential_decay
python train.py Matrix-IPD LIO exponential_decay
python train.py Matrix-IPD IAC exponential_decay

python train.py CoinGame-2 DRIVE-TD exponential_decay
python train.py CoinGame-2 MATE-TD exponential_decay
python train.py CoinGame-2 LIO exponential_decay
python train.py CoinGame-2 IAC exponential_decay

python train.py CoinGame-4 DRIVE-TD exponential_decay
python train.py CoinGame-4 MATE-TD exponential_decay
python train.py CoinGame-4 LIO exponential_decay
python train.py CoinGame-4 IAC exponential_decay

python train.py Harvest-12 DRIVE-TD exponential_decay
python train.py Harvest-12 MATE-TD exponential_decay
python train.py Harvest-12 LIO exponential_decay
python train.py Harvest-12 IAC exponential_decay

# Stepwise Increase

python train.py Matrix-IPD DRIVE-TD stepwise_increase
python train.py Matrix-IPD MATE-TD stepwise_increase
python train.py Matrix-IPD LIO stepwise_increase
python train.py Matrix-IPD IAC stepwise_increase

python train.py CoinGame-2 DRIVE-TD stepwise_increase
python train.py CoinGame-2 MATE-TD stepwise_increase
python train.py CoinGame-2 LIO stepwise_increase
python train.py CoinGame-2 IAC stepwise_increase

python train.py CoinGame-4 DRIVE-TD stepwise_increase
python train.py CoinGame-4 MATE-TD stepwise_increase
python train.py CoinGame-4 LIO stepwise_increase
python train.py CoinGame-4 IAC stepwise_increase

python train.py Harvest-12 DRIVE-TD stepwise_increase
python train.py Harvest-12 MATE-TD stepwise_increase
python train.py Harvest-12 LIO stepwise_increase
python train.py Harvest-12 IAC stepwise_increase

# Damped Cosine

python train.py Matrix-IPD DRIVE-TD cos_damped
python train.py Matrix-IPD MATE-TD cos_damped
python train.py Matrix-IPD LIO cos_damped
python train.py Matrix-IPD IAC cos_damped

python train.py CoinGame-2 DRIVE-TD cos_damped
python train.py CoinGame-2 MATE-TD cos_damped
python train.py CoinGame-2 LIO cos_damped
python train.py CoinGame-2 IAC cos_damped

python train.py CoinGame-4 DRIVE-TD cos_damped
python train.py CoinGame-4 MATE-TD cos_damped
python train.py CoinGame-4 LIO cos_damped
python train.py CoinGame-4 IAC cos_damped

python train.py Harvest-12 DRIVE-TD cos_damped
python train.py Harvest-12 MATE-TD cos_damped
python train.py Harvest-12 LIO cos_damped
python train.py Harvest-12 IAC cos_damped