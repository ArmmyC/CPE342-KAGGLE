#!/bin/bash
# Train the Task 4 models (5 folds each) on one CUDA GPU.
# The final answer blends el5 + cl5; eb5 is trained for the comparison table in task4.ipynb.
# Run from inside the task4 folder:  bash train_all.sh
# Outputs go to preds_raw/ ; then run pack_preds.py to build preds/*.npz used by task4.ipynb.
#
# Approximate time per fold on one NVIDIA L40 (48 GB):
#   cl5 (ConvNeXt V2 Large, 160x320, 18 epochs)        about 1.8 h
#   eb5 (EVA02 Base, 224x448, 18 epochs)               about 1.2 h
#   el5 (EVA02 Large, 224x448, 10 epochs, batch 16)    about 2.5 h
# On a Slurm cluster each fold can run as its own array task (T4_FOLDS=$SLURM_ARRAY_TASK_ID).
set -e
export T4_DATA=${T4_DATA:-../data/04_image_classification}
export T4_OUT=${T4_OUT:-preds_raw}
export T4_WORKERS=${T4_WORKERS:-6}

run () {  # tag model size lr batch epochs aug
  for f in 0 1 2 3 4; do
    T4_TAG=$1 T4_MODEL=$2 T4_SIZE=$3 T4_LR=$4 T4_BS=$5 T4_EPOCHS=$6 T4_AUG=$7 T4_FOLDS=$f python train.py
  done
}

run el5 eva02_large_patch14_448.mim_m38m_ft_in22k_in1k  224,448 2e-5 16 10 5
run cl5 convnextv2_large.fcmae_ft_in22k_in1k            160,320 1e-4 32 18 5
run eb5 eva02_base_patch14_448.mim_in22k_ft_in22k_in1k  224,448 5e-5 32 18 5
