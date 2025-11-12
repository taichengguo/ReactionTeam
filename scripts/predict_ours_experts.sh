#!/bin/bash

MODEL=g2s_series_rel

# Multiple models + Main / Multiple dropouts
#CHECKPOINTS
EXP_NO=1
DATASET=USPTO480k_rare
#DATASET=USPTO_480k
DATASET=USPTO_STEREO
DATASET=USPTO_480kseed2025
DATASET=USPTO_480kseed42

ROOT_PATH=save
ROOT_PATH=save_real_shuffle


BS=20
T=1.0
NBEST=20
MPN_TYPE=dgat
#MPN_TYPE=dgcn

REPR_START=smiles
REPR_END=smiles

# 0.33
#EXP_NO=2
#ROOT_PATH=save_shuffle0.33/
#DATASET=USPTO_480k

#0.66
#EXP_NO=3
#ROOT_PATH=save_shuffle0.66/
#DATASET=USPTO_480k

PREFIX=${DATASET}_${MODEL}_${REPR_START}_${REPR_END}

#for i in 5 7 9 11 13 15 17 19
#for i in 5 7 9 11 12 13
for i in 5 7
#for i in 5
do
  case "$i" in
      5)
          variable=1
          echo "i is in {2, 4, 6, 8}, setting CUDA_VISIBLE_DEVICES to $variable"
          ;;
      7|9)
          variable=2
          echo "i is in {10, 12}, setting CUDA_VISIBLE_DEVICES to $variable"
          ;;
      11)
          variable=3
          echo "i is in {14, 16}, setting CUDA_VISIBLE_DEVICES to $variable"
          ;;
      *)
          variable=3
          echo "i does not match any specified range, setting CUDA_VISIBLE_DEVICES to $variable"
          ;;
  esac
  CHECKPOINT=./$ROOT_PATH/model.${i}.pt
  # predict
  CUDA_VISIBLE_DEVICES=$variable python predict.py \
    --do_predict \
    --do_score \
    --model="$MODEL" \
    --data_name="$DATASET" \
    --test_bin="./preprocessed/$PREFIX/test_0.npz" \
    --test_tgt="./data/$DATASET/tgt-test.txt" \
    --result_file="./results/${PREFIX}_${i}_${EXP_NO}_result_idx.txt" \
    --log_file="$PREFIX.predict.$EXP_NO.log" \
    --load_from="$CHECKPOINT" \
    --mpn_type="$MPN_TYPE" \
    --rel_pos="$REL_POS" \
    --seed=42 \
    --batch_type=tokens \
    --predict_batch_size=8000 \
    --beam_size="$BS" \
    --n_best="$NBEST" \
    --temperature="$T" \
    --predict_min_len=1 \
    --predict_max_len=512 \
    --log_iter=100 &

  for seed in 42 101 202
#  for seed in 42 101 202 303 2025
  do
    # dropout
    CUDA_VISIBLE_DEVICES=$variable python predict_dropout.py \
    --do_predict \
    --do_score \
    --do_dropout \
    --model="$MODEL" \
    --data_name="$DATASET" \
    --test_bin="./preprocessed/$PREFIX/test_0.npz" \
    --test_tgt="./data/$DATASET/tgt-test.txt" \
    --result_file="./results/${PREFIX}_${i}_${EXP_NO}_drop${seed}_result_idx.txt" \
    --log_file="$PREFIX.predict.$EXP_NO.log" \
    --load_from="$CHECKPOINT" \
    --mpn_type="$MPN_TYPE" \
    --rel_pos="$REL_POS" \
    --seed=$seed \
    --batch_type=tokens \
    --predict_batch_size=8000 \
    --beam_size="$BS" \
    --n_best="$NBEST" \
    --temperature="$T" \
    --predict_min_len=1 \
    --predict_max_len=512 \
    --log_iter=100 &
  done
done
