import argparse
import logging
import pdb

import numpy as np
import os
import sys
import torch
from models.graph2seq_series_rel import Graph2SeqSeriesRel
from models.seq2seq import Seq2Seq
from torch.utils.data import DataLoader
from utils import parsing
from utils.data_utils import canonicalize_smiles, load_vocab, S2SDataset, G2SDataset
from utils.train_utils import log_tensor, param_count, set_seed, setup_logger
import numpy as np
import pandas as pd
import random
import time

def remove_duplicates_ordered(lst):
    seen = set()
    result = []
    for sub in lst:
        t = tuple(sub)
        if t not in seen:
            seen.add(t)
            result.append(sub)
    return result

def get_predict_parser():
    parser = argparse.ArgumentParser("predict")
    parsing.add_common_args(parser)
    parsing.add_preprocess_args(parser)
    parsing.add_train_args(parser)
    parsing.add_predict_args(parser)

    return parser


def main(args):
    parsing.log_args(args)
    selected_topn = 10

    if args.do_predict and os.path.exists(args.result_file):
        logging.info(f"Result file found at {args.result_file}, skipping prediction")

    elif args.do_predict and not os.path.exists(args.result_file):
        # os.makedirs(os.path.join("./results", args.data_name), exist_ok=True)

        # initialization ----------------- model
        assert os.path.exists(args.load_from), f"{args.load_from} does not exist!"
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        state = torch.load(args.load_from)
        pretrain_args = state["args"]
        pretrain_state_dict = state["state_dict"]

        # TODO original model wrong save previous error args => temporary using original model
        state_ori = torch.load("./checkpoints/pretrained/USPTO_480k_dgcn.pt")
        pretrain_args_ori = state_ori["args"]
        # pdb.set_trace()
        pretrain_args.vocab_file = pretrain_args_ori.vocab_file

        for attr in ["mpn_type", "rel_pos"]:
            try:
                getattr(pretrain_args, attr)
            except AttributeError:
                setattr(pretrain_args, attr, getattr(args, attr))

        assert args.model == pretrain_args.model, f"Pretrained model is {pretrain_args.model}!"
        if args.model == "s2s":
            model_class = Seq2Seq
            dataset_class = S2SDataset
        elif args.model == "g2s_series_rel":
            model_class = Graph2SeqSeriesRel
            dataset_class = G2SDataset
            args.compute_graph_distance = True
            assert args.compute_graph_distance
        else:
            raise ValueError(f"Model {args.model} not supported!")

        # initialization ----------------- vocab
        vocab = load_vocab(pretrain_args.vocab_file)
        vocab_tokens = [k for k, v in sorted(vocab.items(), key=lambda tup: tup[1])]

        model = model_class(pretrain_args, vocab)
        model.load_state_dict(pretrain_state_dict)
        logging.info(f"Loaded pretrained state_dict from {args.load_from}")

        model.to(device)
        model.eval()

        logging.info(model)
        logging.info(f"Number of parameters = {param_count(model)}")

        # initialization ----------------- data
        test_dataset = dataset_class(pretrain_args, file=args.test_bin)
        test_dataset.batch(
            batch_type=args.batch_type,
            batch_size=args.predict_batch_size
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda _batch: _batch[0],
            pin_memory=True
        )

        all_predictions = []
        sample2candidates = {}

        # TODO Read prediction tokens files
        all_candidates = {}
        file_idx = 0

        # main model file and 5 dropout: All select top-10
        # control num_dropouts
        dropout_list = [42, 101, 202, 303, 2025, 1, 2222, 1996, 666, 789, 999, 1001]
        selected_dropouts = dropout_list[:args.num_dropout]
        selected_topn = 20

        for topn in range(selected_topn):
            if args.data_name == 'USPTO480k_rare':
                filename = 'USPTO480k_rare_g2s_series_rel_smiles_smiles.1.result_token.txt'
            elif args.data_name == 'USPTO_STEREO':
                filename = 'USPTO_STEREO_g2s_series_rel_smiles_smiles.1.result_token.txt'
            else:
                filename = f"USPTO_480k_g2s_series_rel_smiles_smiles.{args.exp_no}.result_token.txt"
            with open(f"./results/{filename}", "r") as f_predict:
                file_candiates = []
                for i, line_predict in enumerate(f_predict):
                    split_data = line_predict.split(",")
                    smis_predict = [list(map(int, part.split())) for part in split_data]
                    file_candiates.append(smis_predict[topn])

            all_candidates[file_idx] = file_candiates
            file_idx += 1

            for seed in selected_dropouts:
                if args.data_name == 'USPTO480k_rare':
                    filename = 'USPTO480k_rare_g2s_series_rel_smiles_smiles.1_drop{}_result_idx_token.txt'.format(seed)
                elif args.data_name == 'USPTO_STEREO':
                    filename = 'USPTO_STEREO_g2s_series_rel_smiles_smiles.1_drop{}_result_idx_token.txt'.format(seed)
                else:
                    filename = 'USPTO_480k_g2s_series_rel_smiles_smiles.{}_drop{}_result_idx_token.txt'.format(args.exp_no, seed)
                try:
                    with open(f"./results/{filename}", "r") as f_predict:
                        file_candiates = []
                        for i, line_predict in enumerate(f_predict):
                            split_data = line_predict.split(",")
                            smis_predict = [list(map(int, part.split())) for part in split_data]
                            file_candiates.append(smis_predict[topn] if topn < len(smis_predict) else smis_predict[0])

                        all_candidates[file_idx] = file_candiates
                        file_idx += 1
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
                    continue


        # files
        expert_list = [5, 7]
        select_models = expert_list[:args.num_experts]
        # dropout_list = [42, 101, 202, 303, 2025, 1, 2222, 1996,  666, 789, 999, 1001, 9999, 998]
        # # get current time second as the seed for random.sample
        # random.seed(time.time())
        # selected_dropouts = random.sample(dropout_list, 5)
        # selected_dropouts = selected_dropouts[:args.num_dropout]
        selected_dropouts = [42]

        print(selected_dropouts)

        for topn in range(selected_topn):
            for modeli in select_models:
                if args.data_name == 'USPTO480k_rare':
                    filename = '{}_result_idx_token.txt'.format(modeli)
                    # filename = 'USPTO480k_rare_g2s_series_rel_smiles_smiles_{}_result_idx_token.txt'.format(modeli)
                elif args.data_name == 'USPTO_STEREO':
                    filename = 'USPTO_STEREO_g2s_series_rel_smiles_smiles_{}_result_idx_token.txt'.format(modeli)
                else:
                    filename = 'USPTO_480k_g2s_series_rel_smiles_smiles_{}_{}_result_idx_token.txt'.format(modeli, args.exp_no)

                try:
                    with open(f"./results/{filename}", "r") as f_predict:
                        file_candiates = []

                        for i,  line_predict in enumerate(f_predict):
                            split_data = line_predict.split(",")
                            smis_predict = [list(map(int, part.split())) for part in split_data]
                            file_candiates.append(smis_predict[topn] if topn < len(smis_predict) else smis_predict[0])

                        all_candidates[file_idx] = file_candiates
                        file_idx += 1
                except Exception as e:
                    print(modeli, topn)
                    print(e)
                    continue

                for seed in selected_dropouts:
                    try:
                        if args.data_name == 'USPTO480k_rare':
                            # filename = '{}_drop{}_result_idx_token.txt'.format(modeli, seed)
                            filename = 'USPTO480k_rare_g2s_series_rel_smiles_smiles_{}_drop{}_result_idx_token.txt'.format(modeli, seed)
                        elif args.data_name == 'USPTO_STEREO':
                            filename = 'USPTO_STEREO_g2s_series_rel_smiles_smiles_{}_drop{}_result_idx_token.txt'.format(modeli, seed)
                        else:
                            filename = 'USPTO_480k_g2s_series_rel_smiles_smiles_{}_{}_drop{}_result_idx_token.txt'.format(modeli, args.exp_no, seed)
                        with open(f"./results/{filename}", "r") as f_predict:
                            file_candiates = []

                            for i, line_predict in enumerate(f_predict):
                                split_data = line_predict.split(",")
                                smis_predict = [list(map(int, part.split())) for part in split_data]
                                file_candiates.append(smis_predict[topn] if topn < len(smis_predict) else smis_predict[0])

                            all_candidates[file_idx] = file_candiates
                            file_idx += 1
                    except Exception as e:
                        print(modeli, seed)
                        print(e)

        all_candidates_sample = {}
        # convert all_candidates to all_candidates_sample
        for _, candidates in all_candidates.items():
            for sample_idx, candidate in enumerate(candidates):
                all_candidates_sample[sample_idx] = all_candidates_sample.get(sample_idx, []) + [candidate]

        with torch.no_grad():
            for test_idx, test_batch in enumerate(test_loader):

                # TODO Obtain the pred_indexs of test_batch
                if test_idx % args.log_iter == 0:
                    logging.info(f"Doing inference on test step {test_idx}")
                    sys.stdout.flush()

                test_batch.to(device)
                # generate batch (test and corresponding target_tokens)
                # target_tokens = tensor (bz * max_length)
                for file_id in all_candidates.keys():
                    # print("Getting score of {}".format(file_id))
                    candidateN_token_ids = all_candidates[file_id]
                    candidateN_token_ids = [candidateN_token_ids[i] for i in test_batch.data_indices.tolist()]
                    results_scores = model.input_output_score(reaction_batch=test_batch,
                                                       pred_tgt_token_ids=candidateN_token_ids)

                    for idx, sample_index in enumerate(test_batch.data_indices.tolist()):
                        sample2candidates[sample_index] = sample2candidates.get(sample_index, []) + [results_scores[idx]]

                # sorting scores and return following smiles
                for sample_index in test_batch.data_indices.tolist():
                    candidate_scores = sample2candidates[sample_index]
                    # sorting candidate_scores and return indexs
                    sorted_indices = np.argsort(candidate_scores)
                    smis = []
                    # pdb.set_trace()
                    sample_all_candidates = all_candidates_sample[sample_index]

                    for candidate_index in sorted_indices:
                        predicted_idx = sample_all_candidates[candidate_index]
                        predicted_tokens = [vocab_tokens[idx] for idx in predicted_idx[:-1]]
                        smi = " ".join(predicted_tokens)
                        smis.append(smi)

                    # remove duplicates in order to keep the first one
                    smis = remove_duplicates_ordered(smis)
                    smis = ",".join(smis)
                    all_predictions.append(f"{smis}\n")

        with open(args.result_file, "w") as of:
            of.writelines(all_predictions)

    if args.do_score:
        correct = 0
        invalid = 0

        with open(args.test_tgt, "r") as f:
            total = sum(1 for _ in f)

        accuracies = np.zeros([total, args.n_best], dtype=np.float32)

        with open(args.test_tgt, "r") as f_tgt, open(args.result_file, "r") as f_predict:
            for i, (line_tgt, line_predict) in enumerate(zip(f_tgt, f_predict)):
                smi_tgt = "".join(line_tgt.split())
                smi_tgt = canonicalize_smiles(smi_tgt, trim=False)
                if not smi_tgt or smi_tgt == "CC":
                    continue

                # smi_predict = "".join(line_predict.split())
                line_predict = "".join(line_predict.split())
                smis_predict = line_predict.split(",")
                smis_predict = [canonicalize_smiles(smi, trim=False) for smi in smis_predict]
                if not smis_predict[0]:
                    invalid += 1
                smis_predict = [smi for smi in smis_predict if smi and not smi == "CC"]
                smis_predict = list(dict.fromkeys(smis_predict))

                for j, smi in enumerate(smis_predict):
                    if smi == smi_tgt:
                        accuracies[i, j:] = 1.0
                        break

        logging.info(f"Total: {total}, "
                     f"top 1 invalid: {invalid / total * 100: .2f} %")

        mean_accuracies = np.mean(accuracies, axis=0)

        print(f"Experts: {args.num_experts}, Dropouts: {args.num_dropout}")
        # append the experts and dropouts to a file
        with open(f"results_summary{selected_topn}_final.txt", "a") as of:
            of.write("=" * 30 + "\n")
            of.write(f"Experts: {args.num_experts}, Dropouts: {args.num_dropout}\n")
            for n in range(args.n_best):
                of.write(f"Top {n+1} accuracy: {mean_accuracies[n] * 100: .2f} %\n")

        for n in range(args.n_best):
            logging.info(f"Top {n+1} accuracy: {mean_accuracies[n] * 100: .2f} %")



if __name__ == "__main__":
    predict_parser = get_predict_parser()
    args = predict_parser.parse_args()

    # set random seed (just in case)
    set_seed(args.seed)

    # logger setup
    logger = setup_logger(args, warning_off=True)

    torch.set_printoptions(profile="full")
    main(args)
