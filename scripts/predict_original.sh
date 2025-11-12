#!/bin/bash

MODEL=g2s_series_rel

# Multiple models + Main / Multiple dropouts
#CHECKPOINTS
EXP_NO=1
# 2  for 0.33 splitting
#EXP_NO=2
# 3  for 0.66 splitting
#EXP_NO=3

#DATASET=USPTO_480k
CHECKPOINT=./checkpoints/pretrained/USPTO_480k_dgat.pt
#CHECKPOINT=./checkpoints/pretrained/USPTO_480k_dgcn.pt
#CHECKPOINT=./checkpoints/pretrained/USPTO_STEREO_dgcn.pt
#CHECKPOINT=./checkpoints/USPTO_480k_g2s_series_rel_smiles_smiles.1.0.33/model.70000_13.pt
#CHECKPOINT=./checkpoints/USPTO_480k_g2s_series_rel_smiles_smiles.1.0.66/model.85000_16.pt
##CHECKPOINT=./save/model.2.pt
#CHECKPOINT=./checkpoints/USPTO_480kseed42_g2s_series_rel_smiles_smiles.1/model.185000_36.pt
#CHECKPOINT=./checkpoints/USPTO_480kseed2025_g2s_series_rel_smiles_smiles.1/model.185000_36.pt
#

BS=10
T=1.0
NBEST=50
MPN_TYPE=dgat

REPR_START=smiles
REPR_END=smiles

# Run for obtain dropout and original performance
#for DATASET in USPTO480k_rare USPTO_480k
#do
#  PREFIX=${DATASET}_${MODEL}_${REPR_START}_${REPR_END}
#
#  python predict_original.py \
#    --do_predict \
#    --do_score \
#    --model="$MODEL" \
#    --data_name="$DATASET" \
#    --test_bin="./preprocessed/$PREFIX/test_0.npz" \
#    --test_tgt="./data/$DATASET/tgt-test.txt" \
#    --result_file="./results/$PREFIX.$EXP_NO.result.txt" \
#    --log_file="$PREFIX.predict.$EXP_NO.log" \
#    --load_from="$CHECKPOINT" \
#    --mpn_type="$MPN_TYPE" \
#    --rel_pos="$REL_POS" \
#    --seed=42 \
#    --batch_type=tokens \
#    --predict_batch_size=5000 \
#    --beam_size="$BS" \
#    --n_best="$NBEST" \
#    --temperature="$T" \
#    --predict_min_len=1 \
#    --predict_max_len=512 \
#    --log_iter=100 >> log_original 2>&1 &
#
#
#  ## Ablation_study: Add dropout for original model
#  for seed in 42 101 202 303 2025 1 2222 1996 666 789 999 1001
#  do
#      case "$seed" in
#        42|1001)
#            variable=0
#            echo "i is in {2, 4, 6, 8}, setting CUDA_VISIBLE_DEVICES to $variable"
#            ;;
#        101|2025|2222)
#            variable=1
#            echo "i is in {10, 12}, setting CUDA_VISIBLE_DEVICES to $variable"
#            ;;
#        202|303|1996)
#            variable=2
#            echo "i is in {14, 16}, setting CUDA_VISIBLE_DEVICES to $variable"
#            ;;
#        *)
#            variable=3
#            echo "i does not match any specified range, setting CUDA_VISIBLE_DEVICES to $variable"
#            ;;
#    esac
#    # dropout
#    CUDA_VISIBLE_DEVICES=${variable} python predict_original_dropout.py \
#    --do_predict \
#    --do_score \
#    --do_dropout \
#    --model="$MODEL" \
#    --data_name="$DATASET" \
#    --test_bin="./preprocessed/$PREFIX/test_0.npz" \
#    --test_tgt="./data/$DATASET/tgt-test.txt" \
#    --result_file="./results/${PREFIX}.${EXP_NO}_drop${seed}_result_idx.txt" \
#    --log_file="$PREFIX.predict.$EXP_NO.log" \
#    --load_from="$CHECKPOINT" \
#    --mpn_type="$MPN_TYPE" \
#    --rel_pos="$REL_POS" \
#    --seed=$seed \
#    --batch_type=tokens \
#    --predict_batch_size=5000 \
#    --beam_size="$BS" \
#    --n_best="$NBEST" \
#    --temperature="$T" \
#    --predict_min_len=1 \
#    --predict_max_len=512 \
#    --log_iter=100 >> log_dropout 2>&1 &
#  done
#done
#
#

#DATASET=USPTO_STEREO
#DATASET=USPTO_480k
#DATASET=USPTO_480kseed42
DATASET=USPTO_480kseed2025
PREFIX=${DATASET}_${MODEL}_${REPR_START}_${REPR_END}
CUDA_VISIBLE_DEVICES=0 python predict_original.py \
  --do_predict \
  --do_score \
  --model="$MODEL" \
  --data_name="$DATASET" \
  --test_bin="./preprocessed/$PREFIX/test_0.npz" \
  --test_tgt="./data/$DATASET/tgt-test.txt" \
  --result_file="./results/$PREFIX.$EXP_NO.result.txt" \
  --log_file="$PREFIX.predict.$EXP_NO.log" \
  --load_from="$CHECKPOINT" \
  --mpn_type="$MPN_TYPE" \
  --rel_pos="$REL_POS" \
  --seed=42 \
  --batch_type=tokens \
  --predict_batch_size=100000 \
  --beam_size="$BS" \
  --n_best="$NBEST" \
  --temperature="$T" \
  --predict_min_len=1 \
  --predict_max_len=512 \
  --log_iter=100