import argparse
import logging
import pdb

import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from models.graph2seq_series_rel import Graph2SeqSeriesRel, apply_lora_to_decoder
from models.seq2seq import Seq2Seq
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader
from utils import parsing
from utils.data_utils import load_vocab, S2SDataset, G2SDataset
from utils.train_utils import get_lr, grad_norm, NoamLR, param_count, param_norm, set_seed, setup_logger


def get_train_parser():
    parser = argparse.ArgumentParser("train")
    parsing.add_common_args(parser)
    parsing.add_train_args(parser)
    parsing.add_predict_args(parser)

    return parser


def main(args):
    parsing.log_args(args)

    # initialization ----------------- vocab
    if not os.path.exists(args.vocab_file):
        raise ValueError(f"Vocab file {args.vocab_file} not found!")
    vocab = load_vocab(args.vocab_file)
    vocab_tokens = [k for k, v in sorted(vocab.items(), key=lambda tup: tup[1])]

    # initialization ----------------- model
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.model == "s2s":
        model_class = Seq2Seq
        dataset_class = S2SDataset
    elif args.model == "g2s_series_rel":
        model_class = Graph2SeqSeriesRel
        dataset_class = G2SDataset
        assert args.compute_graph_distance
    else:
        raise ValueError(f"Model {args.model} not supported!")

    model = model_class(args, vocab)
    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad:
            xavier_uniform_(p)

    if args.load_from:
        state = torch.load(args.load_from)
        pretrain_args = state["args"]
        pretrain_state_dict = state["state_dict"]

        boosting_interval = args.boosting_interval
        # args = pretrain_args
        args.boosting_interval = boosting_interval

        # print(args)
        # print(pretrain_args)
        # For 480k and stereo
        vocab = load_vocab(pretrain_args.vocab_file)
        vocab_tokens = [k for k, v in sorted(vocab.items(), key=lambda tup: tup[1])]

        model = model_class(pretrain_args, vocab)
        model.load_state_dict(pretrain_state_dict)
        logging.info(f"Loaded pretrained state_dict from {args.load_from}")

    model.to(device)
    model.train()

    logging.info(model)
    logging.info(f"Number of parameters = {param_count(model)}")

    # initialization ----------------- optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    scheduler = NoamLR(
        optimizer,
        model_size=args.decoder_hidden_size,
        warmup_steps=args.warmup_steps
    )

    # initialization ----------------- data
    train_dataset = dataset_class(args, file=args.train_bin)
    valid_dataset = dataset_class(args, file=args.valid_bin)

    total_step = 0
    accum = 0
    losses, accs = [], []

    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler(enabled=args.enable_amp)

    o_start = time.time()

    logging.info("Start training")

    correct_index_epoch = ()
    boosting_save = 0
    apply_lora = False

    for epoch in range(args.epoch):
        correct_data_index_this_epoch = []
        if epoch >= 12:
            args.boosting_interval = 1
            print("change boosting_interval to 1")

        # TODO Export domain lora experts:
        #  1. Remove learned indexes from train_dataset
        #  2. Save current model
        #  3. Freeze all wieghts except lora attention
        # each sample, a list, rank and select top10% percentil accuracy; later 3 compare number of data
        if epoch >= 5 and boosting_save >= args.boosting_interval:
            # Remove learned indexes from train_dataset
            boosting_save = 0
            print("calculate correct index of epoch {}".format(epoch))
            correct_index_epoch = ()
            for i in range(int(epoch - args.boosting_interval), epoch):
                print('./correct_index_epoch_{}.txt'.format(i))
                with open('./correct_index_epoch_{}.txt'.format(i), "r") as f:
                    tem = f.readlines()
                tem = [int(line.strip()) for line in tem]  # TODO
                if len(correct_index_epoch) > 0:
                    # correct_index_epoch = set(correct_index_epoch).union(set(tem))
                    # intersect
                    correct_index_epoch = set(correct_index_epoch).intersection(set(tem))
                else:
                    correct_index_epoch = set(tem)

            correct_index_epoch = list(correct_index_epoch)
            print("len previous correct index is {}, and update the training data loader".format(
                len(correct_index_epoch)))
            # update dataset loader
            train_dataset.update_data_indices(correct_index_epoch, epoch)

            # Save each lora experts
            state = {
                "args": pretrain_args,
                "state_dict": model.state_dict()
            }
            torch.save(state, f"save_stere/model.{epoch}.pt")

            # Train Lora parameters only
            for name, param in model.named_parameters():
                if "lora" not in name:  # Only allow LoRA parameters to be updated
                    param.requires_grad = False
                else:
                    print(f"lora param is {name}")

        # New lora expert training
        if not apply_lora:
            apply_lora_to_decoder(model, d_model=pretrain_args.decoder_hidden_size, rank=128, alpha=256, device=device)
            apply_lora = True
            logging.info(model)

            # TODO freeze model weights
            # Train Lora parameters only
            for name, param in model.named_parameters():
                if "lora" not in name:  # Only allow LoRA parameters to be updated
                    param.requires_grad = False
                else:
                    print(f"lora param is {name}")

        # Count the total number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")

        # Count trainable parameters only
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")

        boosting_save += 1
        model.zero_grad()

        # train_dataset.sort()
        train_dataset.shuffle_in_bucket(bucket_size=1000)
        train_dataset.batch(
            batch_type=args.batch_type,
            batch_size=args.train_batch_size
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda _batch: _batch[0],
            pin_memory=True
        )

        for batch_idx, batch in enumerate(train_loader):
            # # TODO for test
            # if batch_idx >= 20:
            #     break

            if total_step > args.max_steps:
                logging.info("Max steps reached, finish training")
                exit(0)

            batch.to(device)
            with torch.autograd.profiler.profile(enabled=args.do_profile,
                                                 record_shapes=args.record_shapes,
                                                 use_cuda=torch.cuda.is_available()) as prof:

                # Enables autocasting for the forward pass (model + loss)
                with torch.cuda.amp.autocast(enabled=args.enable_amp):
                    loss, acc, acc_each_example = model(batch, foreach_example=True)

                # Exits the context manager before backward()
                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()

                losses.append(loss.item())
                accs.append(acc.item() * 100)

                # TODO Record learned data indexes for each batch
                # Find indices where the ratio is not equal to 1
                try:
                    indices_equal_1 = torch.nonzero(acc_each_example == 1).squeeze()
                    # Optionally convert to a Python list
                    indices_equal_1_list = indices_equal_1.tolist()
                    if type(indices_equal_1_list) == int:
                        indices_equal_1_list = [indices_equal_1_list]

                    data_indices = batch.data_indices.tolist()
                    correct_indexs_batch = [data_indices[i] for i in indices_equal_1_list]
                    correct_data_index_this_epoch += correct_indexs_batch
                except Exception as e:
                    print(e)
                    print(indices_equal_1_list)
                    print(data_indices)
                    pdb.set_trace()

                # pdb.set_trace()


                accum += 1

                if accum == args.accumulation_count:
                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)

                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

                    # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                    scaler.step(optimizer)

                    # Updates the scale for next iteration.
                    scaler.update()

                    scheduler.step()

                    g_norm = grad_norm(model)
                    model.zero_grad()
                    accum = 0
                    total_step += 1

            if args.do_profile:
                logging.info(prof
                             .key_averages(group_by_input_shape=args.record_shapes)
                             .table(sort_by="cuda_time_total"))
                sys.stdout.flush()

            if (accum == 0) and (total_step > 0) and (total_step % args.log_iter == 0):
                logging.info(f"Step {total_step}, loss: {np.mean(losses)}, acc: {np.mean(accs)}, "
                             f"p_norm: {param_norm(model)}, g_norm: {g_norm}, "
                             f"lr: {get_lr(optimizer): .6f}, elapsed time: {time.time() - o_start: .0f}")
                sys.stdout.flush()
                losses, accs = [], []

            if (accum == 0) and (total_step > 0) and (total_step % args.eval_iter == 0):
                model.eval()
                eval_count = 10
                eval_meters = [0.0, 0.0]

                valid_dataset.sort()
                valid_dataset.shuffle_in_bucket(bucket_size=1000)
                valid_dataset.batch(
                    batch_type=args.batch_type,
                    batch_size=args.valid_batch_size
                )
                valid_loader = DataLoader(
                    dataset=valid_dataset,
                    batch_size=1,
                    shuffle=True,
                    collate_fn=lambda _batch: _batch[0],
                    pin_memory=True
                )

                with torch.no_grad():
                    for eval_idx, eval_batch in enumerate(valid_loader):
                        if eval_idx >= eval_count:
                            break
                        eval_batch.to(device)

                        eval_loss, eval_acc = model(eval_batch)
                        eval_meters[0] += eval_loss.item() / eval_count
                        eval_meters[1] += eval_acc * 100 / eval_count

                logging.info(f"Evaluation (with teacher) at step {total_step}, eval loss: {eval_meters[0]}, "
                             f"eval acc: {eval_meters[1]}")
                sys.stdout.flush()

                model.train()

            if (accum == 0) and (total_step > 0) and (total_step % args.save_iter == 0):
                n_iter = total_step // args.save_iter - 1

                model.eval()
                eval_count = 10

                valid_dataset.sort()
                valid_dataset.shuffle_in_bucket(bucket_size=1000)
                valid_dataset.batch(
                    batch_type=args.batch_type,
                    batch_size=args.valid_batch_size
                )
                valid_loader = DataLoader(
                    dataset=valid_dataset,
                    batch_size=1,
                    shuffle=True,
                    collate_fn=lambda _batch: _batch[0],
                    pin_memory=True
                )

                accs_token = []
                accs_seq = []

                with torch.no_grad():
                    for eval_idx, eval_batch in enumerate(valid_loader):
                        if eval_idx >= eval_count:
                            break

                        eval_batch.to(device)
                        results = model.predict_step(
                            reaction_batch=eval_batch,
                            batch_size=eval_batch.size,
                            beam_size=args.beam_size,
                            n_best=1,
                            temperature=1.0,
                            min_length=args.predict_min_len,
                            max_length=args.predict_max_len
                        )
                        predictions = [t[0].cpu().numpy() for t in results["predictions"]]

                        for i, prediction in enumerate(predictions):
                            tgt_length = eval_batch.tgt_lengths[i].item()
                            tgt_token_ids = eval_batch.tgt_token_ids[i].cpu().numpy()[:tgt_length]

                            acc_seq = np.array_equal(tgt_token_ids, prediction)
                            while len(prediction) < tgt_length:
                                prediction = np.append(prediction, vocab["_PAD"])

                            acc_token = np.mean(tgt_token_ids == prediction[:tgt_length])

                            accs_token.append(acc_token)
                            accs_seq.append(acc_seq)

                            if eval_idx % 20 == 0 and i == 0:
                                logging.info(f"Target text: {' '.join([vocab_tokens[idx] for idx in tgt_token_ids])}")
                                logging.info(f"Predicted text: {' '.join([vocab_tokens[idx] for idx in prediction])}")
                                logging.info(f"acc_token: {acc_token}, acc_seq: {acc_seq}\n")

                logging.info(f"Evaluation (without teacher) at step {total_step}, "
                             f"eval acc (token): {np.mean(accs_token)}, "
                             f"eval acc (sequence): {np.mean(accs_seq)}")
                sys.stdout.flush()

                model.train()

                logging.info(f"Saving at step {total_step}")
                sys.stdout.flush()

                state = {
                    "args": args,
                    "state_dict": model.state_dict()
                }

                # TODO Save each lora experts
                # torch.save(state, os.path.join(args.save_dir, f"model.{total_step}_{n_iter}.pt"))

        print('correct_data_index_this_epoch {} length is: {}'.format(epoch, len(correct_data_index_this_epoch)))
        with open('./correct_index_epoch_{}.txt'.format(epoch), 'w') as f:
            for i in correct_data_index_this_epoch:
                f.write(str(int(i)) + '\n')

        # lastly
        if (args.accumulation_count > 1) and (accum > 0):
            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

            scaler.step(optimizer)
            scaler.update()

            # optimizer.step()
            scheduler.step()

            model.zero_grad()
            accum = 0


if __name__ == "__main__":
    train_parser = get_train_parser()
    args = train_parser.parse_args()

    # set random seed
    set_seed(args.seed)

    # logger setup
    logger = setup_logger(args)

    torch.set_printoptions(profile="full")
    main(args)
