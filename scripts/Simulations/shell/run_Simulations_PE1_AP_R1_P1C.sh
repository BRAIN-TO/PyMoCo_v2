#!/bin/bash -l
# export M_DIR=/home/nghiemb/Data/TWH/MPRAGE_ReferencePoses/InVivo/sub-01/dat
# export IN_DIR=${M_DIR}/corrupted/Test38
# export OUT_DIR=${IN_DIR}/neck_cropped
# export SUBMIT_DIR=/home/nghiemb/PyMoCo

export SUBMIT_DIR=/home/nghiemb/PyMoCo
export OUT_DIR=${SUBMIT_DIR}/data/cc/test/PE1_AP/Complex/R1/Paradigm_1C

python ${SUBMIT_DIR}/main_Simulations_PE1_AP_R1_P1C.py > ${OUT_DIR}/log_P1C_2025-04-14.txt & disown
