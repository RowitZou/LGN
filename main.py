# -*- coding: utf-8 -*-
# @Author: Yicheng Zou
# @Last Modified by:   Yicheng Zou,    Contact: yczou18@fudan.edu.cn

import time
import sys
import argparse
import random
import copy
import torch
import gc
import pickle
import os
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.LGN import Graph
from utils.data import Data


def data_initialization(data, word_file, train_file, dev_file, test_file):

    data.build_word_file(word_file)

    if train_file:
        data.build_alphabet(train_file)
        data.build_word_alphabet(train_file)
    if dev_file:
        data.build_alphabet(dev_file)
        data.build_word_alphabet(dev_file)
    if test_file:
        data.build_alphabet(test_file)
        data.build_word_alphabet(test_file)
    return data


def predict_check(pred_variable, gold_variable, mask_variable):

    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet):

    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    pred_label = []
    gold_label = []

    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert(len(pred)==len(gold))
        pred_label.append(pred)
        gold_label.append(gold)

    return pred_label, gold_label

def evaluate(data, args, model, name):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print( "Error: wrong evaluate name,", name)
        exit(0)

    pred_results = []
    gold_results = []

    ## set model in eval model
    model.eval()
    batch_size = args.batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1

    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end >train_num:
            end =  train_num
        instance = instances[start:end]
        if not instance:
            continue

        word_list, batch_char, batch_label, mask  = batchify_with_label(instance, args.use_gpu)
        _, tag_seq = model(word_list, batch_char, mask)

        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet)

        pred_results += pred_label
        gold_results += gold_label

    decode_time = time.time() - start_time
    speed = len(instances) / decode_time

    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results)
    return speed, acc, p, r, f, pred_results


def batchify_with_label(input_batch_list, gpu):

    batch_size = len(input_batch_list)
    chars = [sent[0] for sent in input_batch_list]
    words = [sent[1] for sent in input_batch_list]
    labels = [sent[2] for sent in input_batch_list]

    sent_lengths = torch.LongTensor(list(map(len, chars)))
    max_sent_len = sent_lengths.max()
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_sent_len))).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_sent_len))).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_sent_len))).byte()

    for idx, (seq, label, seq_len) in enumerate(zip(chars, labels, sent_lengths)):
        char_seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seq_len] = torch.LongTensor(label)
        mask[idx, :seq_len] = torch.Tensor([1] * int(seq_len))

    if gpu:
        char_seq_tensor = char_seq_tensor.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        mask = mask.cuda()

    return words, char_seq_tensor, label_seq_tensor, mask

def train(data, args, saved_model_path):

    print( "Training model...")
    model = Graph(data, args)
    print('# generated parameters:', sum(param.numel() for param in model.parameters()))
    print( "Finished built model.")

    best_dev_epoch = 0
    best_dev_f = -1
    best_dev_p = -1
    best_dev_r = -1

    best_test_f = -1
    best_test_p = -1
    best_test_r = -1

    # Initialize the optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    for idx in range(args.num_epoch):
        epoch_start = time.time()
        temp_start = epoch_start
        print(("Epoch: %s/%s" %(idx, args.num_epoch)))
        sample_loss = 0
        batch_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
        # set model in train model
        model.train()
        model.zero_grad()
        batch_size = args.batch_size
        train_num = len(data.train_Ids)
        total_batch = train_num // batch_size + 1

        for batch_id in range(total_batch):
            # Get one batch-sized instance
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue

            word_list, batch_char, batch_label, mask = batchify_with_label(instance, args.use_gpu)
            loss, tag_seq = model(word_list, batch_char, mask, batch_label)
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            sample_loss += loss.data
            total_loss += loss.data
            batch_loss += loss

            if end % 500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print(("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" %
                       (end, temp_cost, sample_loss, right_token, whole_token, (right_token+0.)/whole_token)))
                sys.stdout.flush()
                sample_loss = 0
            if end % args.batch_size == 0:
                batch_loss.backward()
                optimizer.step()
                model.zero_grad()
                batch_loss = 0

        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print(("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" %
               (end, temp_cost, sample_loss, right_token, whole_token, (right_token+0.)/whole_token)))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print(("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" %
               (idx, epoch_cost, train_num/epoch_cost, total_loss)))

        ## dev
        speed, acc, dev_p, dev_r, dev_f, _ = evaluate(data, args, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        print(("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" %
               (dev_cost, speed, acc, dev_p, dev_r, dev_f)))

        ## test
        speed, acc, test_p, test_r, test_f, _ = evaluate(data, args, model, "test")
        test_finish = time.time()
        test_cost = test_finish - dev_finish

        print(("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" %
               (test_cost, speed, acc, test_p, test_r, test_f)))

        if dev_f > best_dev_f:
            print("Exceed previous best f score: %.4f" % best_dev_f)

            torch.save(model.state_dict(), saved_model_path)
            best_dev_p = dev_p
            best_dev_r = dev_r
            best_dev_f = dev_f
            best_dev_epoch = idx + 1
            best_test_p = test_p
            best_test_r = test_r
            best_test_f = test_f

        print("Best dev epoch: %d" % best_dev_epoch)
        print("Best dev score: p: %.4f, r: %.4f, f: %.4f" % (best_dev_p, best_dev_r, best_dev_f))
        print("Best test score: p: %.4f, r: %.4f, f: %.4f" % (best_test_p, best_test_r, best_test_f))

        gc.collect()

    with open(saved_model_path + "_result.txt" , "a") as f:
        f.write(saved_model_path + '\n')
        f.write("Best dev score: %.4f, r: %.4f, f: %.4f\n" % (best_dev_p, best_dev_r, best_dev_f))
        f.write("Test score: %.4f, r: %.4f, f: %.4f\n" % (best_test_p, best_test_r, best_test_f))
        f.close()

def load_model_decode(model_dir, data, args, name):
    print( "Load Model from file: ", model_dir)
    model = Graph(data, args)
    ## load model need consider if the model trained in GPU and load in CPU, or vice versa
    if not args.use_gpu:
        model.load_state_dict(torch.load(model_dir, map_location=lambda storage, loc: storage))
    else:
        model.load_state_dict(torch.load(model_dir))

    print(("Decode %s data ..."%(name)))
    start_time = time.time()
    speed, acc, p, r, f, pred_results = evaluate(data, args, model, name)
    end_time = time.time()
    time_cost = end_time - start_time
    print(("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" %
           (name, time_cost, speed, acc, p, r, f)))

    return pred_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', choices=['train', 'test', 'decode'], help='Function status.', default='train')
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--train', help='Training set.')
    parser.add_argument('--dev', help='Developing set.')
    parser.add_argument('--test', help='Testing set.')
    parser.add_argument('--raw', help='Raw file for decoding.')
    parser.add_argument('--output', help='Output results for decoding.')
    parser.add_argument('--saved_set', help='Path of saved data set.')
    parser.add_argument('--saved_model', help='Path of saved model.', default="saved_model/model_best")
    parser.add_argument('--char_emb', help='Path of character embedding file.', default="data/gigaword_chn.all.a2b.uni.ite50.vec")
    parser.add_argument('--word_emb', help='Path of word embedding file.', default="data/ctb.50d.vec")

    parser.add_argument('--seed', help='Random seed', default=47, type=int)
    parser.add_argument('--batch_size', help='Batch size. For now it only works when batch size is 1.', default=1, type=int)
    parser.add_argument('--num_epoch',default=2, type=int, help="Epoch number.")
    parser.add_argument('--iters', default=3, type=int, help='The number of Graph iterations.')
    parser.add_argument('--hidden_dim', default=50, type=int, help='Hidden state size.')
    parser.add_argument('--num_head', default=10, type=int, help='Number of transformer head.')
    parser.add_argument('--head_dim', default=10, type=int, help='Head dimension of transformer.')
    parser.add_argument('--tf_drop_rate', default=0.1, type=float, help='Transformer dropout rate.')
    parser.add_argument('--emb_drop_rate', default=0.5, type=float, help='Embedding dropout rate.')
    parser.add_argument('--cell_drop_rate', default=0.2, type=float, help='Aggregation module dropout rate.')
    parser.add_argument('--lr', type=float, default=5e-04)
    parser.add_argument('--weight_decay', type=float, default=1e-08)

    args = parser.parse_args()

    status = args.status.lower()
    seed_num = args.seed
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)

    train_file = args.train
    dev_file = args.dev
    test_file = args.test
    raw_file = args.raw
    output_file = args.output
    saved_set_path = args.saved_set
    saved_model_path = args.saved_model
    char_file = args.char_emb
    word_file = args.word_emb

    assert saved_set_path is not None

    if status == 'train':
        assert not (train_file is None or dev_file is None or test_file is None)
        if os.path.exists(saved_set_path):
            print('Loading saved data set...')
            with open(saved_set_path, 'rb') as f:
                data = pickle.load(f)
        else:
            data = Data()
            data_initialization(data, word_file, train_file, dev_file, test_file)
            data.generate_instance_with_words(train_file, 'train')
            data.generate_instance_with_words(dev_file, 'dev')
            data.generate_instance_with_words(test_file, 'test')
            data.build_char_pretrain_emb(char_file)
            data.build_word_pretrain_emb(word_file)
            print('Dumping data...')
            with open(saved_set_path, 'wb') as f:
                pickle.dump(data, f)
        data.show_data_summary()
        train(data, args, saved_model_path)

    elif status == 'test':
        if os.path.exists(saved_set_path):
            print('Loading saved data set...')
            with open(saved_set_path, 'rb') as f:
                data = pickle.load(f)
        else:
            print("Cannot find saved data set: ", saved_set_path)
            exit(0)
        data.generate_instance_with_words(test_file, 'test')
        load_model_decode(saved_model_path, data, args, "test")

    elif status == 'decode':
        assert not (raw_file is None or output_file is None)
        if os.path.exists(saved_set_path):
            print('Loading saved data set...')
            with open(saved_set_path, 'rb') as f:
                data = pickle.load(f)
        else:
            print("Cannot find saved data set: ", saved_set_path)
            exit(0)
        data.generate_instance_with_words(raw_file, 'raw')
        decode_results = load_model_decode(saved_model_path, data, args, "raw")
        data.write_decoded_results(output_file, decode_results, 'raw')
    else:
        print( "Invalid argument! Please use valid arguments! (train/test/decode)")
