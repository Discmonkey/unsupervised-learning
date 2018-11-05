#!/usr/bin/env bash

&
 kmeans &
 &
 &
CUDA_VISIBLE_DEVICES=0 python -m experiments.train_keras fire nmf


