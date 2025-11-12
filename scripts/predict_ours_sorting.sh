#!/bin/bash

#rm results/final_result.txt

MODEL=g2s_series_rel

EXP_NO=1
DATASET=USPTO_STEREO
#DATASET=USPTO_480k
#DATASET=USPTO480k_rare
CHECKPOINT=./checkpoints/pretrained/USPTO_480k_dgcn.pt

#DATASET=USPTO_480kseed42
DATASET=USPTO_480kseed2025

#CHECKPOINT=./checkpoints/pretrained/USPTO_STEREO_dgcn.pt

# 0.33
#CHECKPOINT=./checkpoints/USPTO_480k_g2s_series_rel_smiles_smiles.1.0.33/model.70000_13.pt
#EXP_NO=2

# 0.66
#CHECKPOINT=./checkpoints/USPTO_480k_g2s_series_rel_smiles_smiles.1.0.66/model.85000_16.pt
#EXP_NO=3


BS=30
T=1.0
NBEST=50
MPN_TYPE=dgcn

REPR_START=smiles
REPR_END=smiles

PREFIX=${DATASET}_${MODEL}_${REPR_START}_${REPR_END}

# Test the performance related to the num of expert
#for i in 0 1 2 3 4
#do
#  rm results/final_result.txt
#  CUDA_VISIBLE_DEVICES=0 python predict_sorting.py \
#  --do_predict \
#  --do_score \
#  --model="$MODEL" \
#  --data_name="$DATASET" \
#  --test_bin="./preprocessed/$PREFIX/test_0.npz" \
#  --test_tgt="./data/$DATASET/tgt-test.txt" \
#  --result_file="./results/final_result.txt" \
#  --log_file="$PREFIX.predict.$EXP_NO.log" \
#  --load_from="$CHECKPOINT" \
#  --mpn_type="$MPN_TYPE" \
#  --rel_pos="$REL_POS" \
#  --seed=42 \
#  --batch_type=tokens \
#  --predict_batch_size=50000 \
#  --beam_size="$BS" \
#  --n_best="$NBEST" \
#  --temperature="$T" \
#  --predict_min_len=1 \
#  --predict_max_len=512 \
#  --log_iter=100 \
#  --num_experts=$i  \
#  --num_dropout=0
#done

# Test the performance of each dropout
# Most experts different dropout 0-5
#for i in 0 1 2 3 4 5 6 7 8 9 10
#for i in 4 5 6 7 8 9 10
#do
#  rm results/final_result.txt
#  CUDA_VISIBLE_DEVICES=2 python predict_sorting.py \
#  --do_predict \
#  --do_score \
#  --model="$MODEL" \
#  --data_name="$DATASET" \
#  --test_bin="./preprocessed/$PREFIX/test_0.npz" \
#  --test_tgt="./data/$DATASET/tgt-test.txt" \
#  --result_file="./results/final_result.txt" \
#  --log_file="$PREFIX.predict.$EXP_NO.log" \
#  --load_from="$CHECKPOINT" \
#  --mpn_type="$MPN_TYPE" \
#  --rel_pos="$REL_POS" \
#  --seed=42 \
#  --batch_type=tokens \
#  --predict_batch_size=50000 \
#  --beam_size="$BS" \
#  --n_best="$NBEST" \
#  --temperature="$T" \
#  --predict_min_len=1 \
#  --predict_max_len=512 \
#  --log_iter=100 \
#  --num_experts=0  \
#  --num_dropout=$i
#done


for iter in 0
do
#  rm results/final_result.txt
  CUDA_VISIBLE_DEVICES=2 python predict_sorting.py \
    --do_predict \
    --do_score \
    --model="$MODEL" \
    --data_name="$DATASET" \
    --test_bin="./preprocessed/$PREFIX/test_0.npz" \
    --test_tgt="./data/$DATASET/tgt-test.txt" \
    --result_file="./results/$PREFIX.final_result.txt" \
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
    --log_iter=100 \
    --num_experts=6  \
    --num_dropout=10
done


#for iter in 0
##for iter in 0
#do
#rm results/final_result${EXP_NO}.txt
#CUDA_VISIBLE_DEVICES=${iter} python predict_sorting_splitting.py \
#  --do_predict \
#  --do_score \
#  --model="$MODEL" \
#  --data_name="$DATASET" \
#  --test_bin="./preprocessed/$PREFIX/test_0.npz" \
#  --test_tgt="./data/$DATASET/tgt-test.txt" \
#  --result_file="./results/final_result$EXP_NO.txt" \
#  --log_file="$PREFIX.predict.$EXP_NO.log" \
#  --load_from="$CHECKPOINT" \
#  --mpn_type="$MPN_TYPE" \
#  --rel_pos="$REL_POS" \
#  --seed=42 \
#  --batch_type=tokens \
#  --predict_batch_size=100000 \
#  --beam_size="$BS" \
#  --n_best="$NBEST" \
#  --temperature="$T" \
#  --predict_min_len=1 \
#  --predict_max_len=512 \
#  --log_iter=100 \
#  --num_experts=6  \
#  --num_dropout=10 \
#  --exp_no=$EXP_NO
#done


## Just test
#CHECKPOINT=./save/model.5.pt
#CUDA_VISIBLE_DEVICES=0 python predict_sorting.py \
#--do_predict \
#--do_score \
#--do_dropout \
#--model="$MODEL" \
#--data_name="$DATASET" \
#--test_bin="./preprocessed/$PREFIX/test_0.npz" \
#--test_tgt="./data/$DATASET/tgt-test.txt" \
#--result_file="./results/final_result_test.txt" \
#--log_file="$PREFIX.predict.$EXP_NO.log" \
#--load_from="$CHECKPOINT" \
#--mpn_type="$MPN_TYPE" \
#--rel_pos="$REL_POS" \
#--seed=111 \
#--batch_type=tokens \
#--predict_batch_size=50000 \
#--beam_size="$BS" \
#--n_best="$NBEST" \
#--temperature="$T" \
#--predict_min_len=1 \
#--predict_max_len=512 \
#--log_iter=100 \
#--num_experts=0  \
#--num_dropout=0





