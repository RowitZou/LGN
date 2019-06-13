# -*- coding: utf-8 -*-
# @Author: Yicheng Zou
# @Date:   2019-01-04 10:23:47
# @Last Modified by:   Yicheng Zou,    Contact: yczou18@fudan.edu.cn
# @Last Modified time: 2019-06-03 20:22:38

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
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.LGN import Graph as SeqModel
from utils.data import Data


def data_initialization(data, gaz_file, train_file, dev_file, test_file):
    data.build_alphabet(train_file)
    data.build_alphabet(dev_file)
    data.build_alphabet(test_file)
    data.build_gaz_file(gaz_file)
    data.build_gaz_alphabet(train_file)
    data.build_gaz_alphabet(dev_file)
    data.build_gaz_alphabet(test_file)
    data.fix_alphabet()
    return data


def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    # print(("right: %s, total: %s"%(right_token, total_token)))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, print=False):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        # print( "p:",pred, pred_tag.tolist())
        # print( "g:", gold, gold_tag.tolist())
        assert(len(pred)==len(gold))
        pred_label.append(pred)
        gold_label.append(gold)

    return pred_label, gold_label

def print_labels(pred_labels, comment=""):
    with open("labels/pred_label"+comment+".txt","w") as fp:
        for pred in pred_labels:
            fp.writelines("\n".join(pred))
            fp.write("\n\n")
        fp.close()

def print_gazs(gazs, comment=""):
    with open("labels/pred_gazes"+comment+".txt","w") as fp:
        for gazlist in gazs:
            fp.writelines("\n".join(gazlist))
            fp.write("\n\n")
        fp.close()


def save_data_setting(data, save_file):
    new_data = copy.deepcopy(data)
    ## remove input instances
    new_data.train_texts = []
    new_data.dev_texts = []
    new_data.test_texts = []
    new_data.raw_texts = []

    new_data.train_Ids = []
    new_data.dev_Ids = []
    new_data.test_Ids = []
    new_data.raw_Ids = []
    ## save data settings
    with open(save_file, 'wb') as fp:
        pickle.dump(new_data, fp)
    print( "Data setting saved to file: ", save_file)


def load_data_setting(save_file):
    with open(save_file, 'rb') as fp:
        data = pickle.load(fp)
    print( "Data setting loaded from file: ", save_file)
    data.show_data_summary()
    return data

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1-decay_rate)**epoch)
    print( " Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer



def evaluate(data, model, name, printable=False):
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
    right_token = 0
    whole_token = 0
    pred_results = []
    gold_results = []
    """
    le20_pred_results = []
    le20_gold_results = []
    be2040_pred_results = []
    be2040_gold_results = []
    be4060_pred_results = []
    be4060_gold_results = []
    be6080_pred_results = []
    be6080_gold_results = []
    be80100_pred_results = []
    be80100_gold_results = []
    lg100_pred_results = []
    lg100_gold_results = []
    """
    ## set model in eval model
    model.eval()
    batch_size = 1
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    gazes = []
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end >train_num:
            end =  train_num
        instance = instances[start:end]
        if not instance:
            continue
        gaz_list,batch_word, batch_wordlen, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, True)
        tag_seq, gaz_match = model(gaz_list,batch_word, batch_wordlen,mask)
        """
        gazes = []
        for list in gaz_list[0]:
            temp = []
            if len(list)>0:
                for id in list[0]:
                    temp.append(data.gaz_alphabet.instances[id-1])
            gazes.append(temp)
        """
        # print( "tag:",tag_seq)
        if name == "dev":
            pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, True)
        else:
            pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet)
        """
        debug_word = [data.word_alphabet.instances[i-1] for i in batch_word[0]]
        debug_result = []
        for i in range(len(debug_word)):
            temp = []
            temp.append(debug_word[i])
            temp.append(gold_label[0][i])
            temp.append(pred_label[0][i])
            temp.append(gazes[i])
            debug_result.append(temp)
        """
        pred_results += pred_label
        gold_results += gold_label
        """
        if len(instance[0][0]) < 20:
            le20_pred_results += pred_label
            le20_gold_results += gold_label
        elif len(instance[0][0]) < 40:
            be2040_pred_results += pred_label
            be2040_gold_results += gold_label
        elif len(instance[0][0]) < 60:
            be4060_pred_results += pred_label
            be4060_gold_results += gold_label
        elif len(instance[0][0]) < 80:
            be6080_pred_results += pred_label
            be6080_gold_results += gold_label
        elif len(instance[0][0]) < 100:
            be80100_pred_results += pred_label
            be80100_gold_results += gold_label
        else:
            lg100_pred_results += pred_label
            lg100_gold_results += gold_label
            """
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    """
    acc, p, r, f = get_ner_fmeasure(le20_gold_results, le20_pred_results, data.tagScheme)
    print(("le20, acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(acc, p, r, f)))
    acc, p, r, f = get_ner_fmeasure(be2040_gold_results, be2040_pred_results, data.tagScheme)
    print(("2040, acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(acc, p, r, f)))
    acc, p, r, f = get_ner_fmeasure(be4060_gold_results, be4060_pred_results, data.tagScheme)
    print(("4060, acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(acc, p, r, f)))
    acc, p, r, f = get_ner_fmeasure(be6080_gold_results, be6080_pred_results, data.tagScheme)
    print(("6080, acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(acc, p, r, f)))
    acc, p, r, f = get_ner_fmeasure(be80100_gold_results, be80100_pred_results, data.tagScheme)
    print(("80100, acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(acc, p, r, f)))
    acc, p, r, f = get_ner_fmeasure(lg100_gold_results, lg100_pred_results, data.tagScheme)
    print(("lg100, acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(acc, p, r, f)))
    """
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    #print_labels(gold_results, "gold")
    return speed, acc, p, r, f, pred_results, gazes


def batchify_with_label(input_batch_list, gpu, volatile_flag=False):
    """
        input: list of words, chars and labels, various length. [[words,biwords,chars,gaz, labels],[words,biwords,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    gazs = [sent[3] for sent in input_batch_list]

    labels = [sent[4] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len))).byte()
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1]*int(seqlen))

    gazs.append(volatile_flag)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        mask = mask.cuda()
    # print('gazs', gazs)
    # print('word_seq_tensor', word_seq_tensor)
    # print('word_seq_lengths', word_seq_lengths)
    # print('label_seq_tensor', label_seq_tensor)
    return gazs, word_seq_tensor, word_seq_lengths, label_seq_tensor, mask


def train(data, save_model_dir, seg=True, print_label=False, print_gaz=False ):
    print( "Training model...")
    #data.show_data_summary()
    #save_data_name = save_model_dir +".dset"
    #save_data_setting(data, save_data_name)
    model = SeqModel(data)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    print( "finished built model.")
    loss_function = nn.NLLLoss()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=data.HP_lr, weight_decay=data.HP_weight_decay)
    best_dev_epoch = 0
    best_dev = -1
    best_dev_p = -1
    best_dev_r = -1

    best_test = -1
    best_test_p = -1
    best_test_r = -1

    #data.HP_iteration = 100
    ## start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print(("Epoch: %s/%s" %(idx,data.HP_iteration)))
        optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_id = 0
        sample_loss = 0
        batch_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
        ## set model in train model
        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size ## current only support batch size = 1 to compulate and accumulate to data.HP_batch_size update weights
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num//batch_size+1
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end >train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            gaz_list,  batch_word, batch_wordlen, batch_label, mask  = batchify_with_label(instance, data.HP_gpu)
            # print( "gaz_list:",gaz_list)
            # print('gaz_alphabet', gaz_list[0][0][0][0],gaz_list[0][0][1][0],gaz_list[0][5][0][0],gaz_list[0][5][1][0])
            # print('gaz_alphabet', data.gaz_alphabet.get_instance(gaz_list[0][0][0][0]),data.gaz_alphabet.get_instance(gaz_list[0][0][1][0]),data.gaz_alphabet.get_instance(gaz_list[0][6][0][0]),data.gaz_alphabet.get_instance(gaz_list[0][6][1][0]))
            # print('batch_word', [data.word_alphabet.get_instance(i) for i in batch_word[0]])
            # exit(0)
            instance_count += 1
            loss, tag_seq = model.neg_log_likelihood_loss(gaz_list, batch_word, batch_wordlen, mask, batch_label)
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            sample_loss += loss.data
            total_loss += loss.data
            batch_loss += loss

            if end%500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print(("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token)))
                sys.stdout.flush()
                sample_loss = 0
            if end%data.HP_batch_size == 0:
                batch_loss.backward()
                optimizer.step()
                model.zero_grad()
                batch_loss = 0
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print(("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))       )
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print(("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx, epoch_cost, train_num/epoch_cost, total_loss)))
        # exit(0)
        # continue
        speed, acc, p, r, f, pred_labels, gazs = evaluate(data, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        if seg:
            current_score = f
            print(("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(dev_cost, speed, acc, p, r, f)))
        else:
            current_score = acc
            print(("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f"%(dev_cost, speed, acc)))

        if current_score > best_dev:
            if seg:
                print( "Exceed previous best f score:", best_dev)

            else:
                print( "Exceed previous best acc score:", best_dev)
            if print_label:
                print_labels(pred_labels,"dev"+data.label_comment)
            if print_gaz:
                print_gazs(gazs)
            model_name = save_model_dir + '_devbest'
            torch.save(model.state_dict(), model_name)
            #best_dev = current_score
            best_dev_p = p
            best_dev_r = r

        # ## decode test
        #speed, acc, p, r, f, pred_labels, gazs = evaluate(data, model, "test")
        #test_finish = time.time()
        #test_cost = test_finish - dev_finish
        #if seg:
        #    current_test_score = f
        #    print(("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(test_cost, speed, acc, p, r, f)))
        #else:
        #    current_test_score = acc
        #    print(("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f"%(test_cost, speed, acc)))

        if current_score > best_dev:
            if print_label:
                print_labels(pred_labels,"test"+data.label_comment)
            if print_gaz:
                print_gazs(gazs)
            best_dev_epoch = idx+1
            best_dev = current_score
            #best_test = current_test_score
            #best_test_p = p
            #best_test_r = r

        print("Best dev epoch: {}".format(best_dev_epoch))
        print("Best dev score: p:{}, r:{}, f:{}".format(best_dev_p,best_dev_r,best_dev))
        #print("Test score: p:{}, r:{}, f:{}".format(best_test_p,best_test_r,best_test))
        gc.collect()

    with open(data.result_file,"a") as f:
        f.write(save_model_dir+'\n')
        f.write("Best dev score: p:{}, r:{}, f:{}\n".format(best_dev_p,best_dev_r,best_dev))
        f.write("Test score: p:{}, r:{}, f:{}\n\n".format(best_test_p,best_test_r,best_test))
        f.close()

def load_model_decode(model_dir, data, name, gpu, seg=True, print_label=False, print_gaz=False):
    data.HP_gpu = gpu
    print( "Load Model from file: ", model_dir)
    model = SeqModel(data)
    ## load model need consider if the model trained in GPU and load in CPU, or vice versa
    # if not gpu:
    #     model.load_state_dict(torch.load(model_dir, map_location=lambda storage, loc: storage))
    #     # model = torch.load(model_dir, map_location=lambda storage, loc: storage)
    # else:
    model.load_state_dict(torch.load(model_dir))
        # model = torch.load(model_dir)

    print(("Decode %s data ..."%(name)))
    start_time = time.time()
    speed, acc, p, r, f, pred_results, gazs = evaluate(data, model, name)
    end_time = time.time()
    time_cost = end_time - start_time
    if seg:
        print(("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(name, time_cost, speed, acc, p, r, f)))
    else:
        print(("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f"%(name, time_cost, speed, acc)))

    if print_label:
        print_labels(pred_results,name+data.label_comment)
    if print_gaz:
        print_gazs(gazs,name+data.label_comment)
    return pred_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='Batch size. For now it only works when batch size is 1.', default=1, type=int)
    parser.add_argument('--status', choices=['train', 'test', 'decode'], help='Function status.', default='train')
    parser.add_argument('--train', help='Training set.')
    parser.add_argument('--dev', help='Developing set.')
    parser.add_argument('--test', help='Testing set.')
    parser.add_argument('--raw', help='Raw file for decoding.')
    parser.add_argument('--output', help='Output results for decoding.')
    parser.add_argument('--saved_set', help='Path of saved data set.')
    parser.add_argument('--saved_model', help='Path of saved model.', default="save_model/model_")
    parser.add_argument('--char_emb', help='Path of character embedding file.', default="data/gigaword_chn.all.a2b.uni.ite50.vec")
    parser.add_argument('--word_emb', help='Path of word embedding file.', default="data/ctb.50d.vec")

    parser.add_argument('--seg', default="True")
    parser.add_argument('--extendalphabet', default="True")

    parser.add_argument('--loadmodel',default="save_model/model__devbest")
    parser.add_argument('--seed', help='Random seed', default=47, type=int)
    parser.add_argument('--labelcomment', default="")
    parser.add_argument('--resultfile',default="result/result.txt")
    parser.add_argument('--num_iter',default=50,type=int)
    parser.add_argument('--num_layer', default=2, type=int, help='The number of Graph layers')
    parser.add_argument('--use_gaz', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--lr_decay', type=float, default=0.00)
    parser.add_argument('--weight_decay', type=float, default=1e-08)

    args = parser.parse_args()

    status = args.status.lower()
    use_gpu = torch.cuda.is_available()
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

    if args.seg.lower() == "true":
        seg = True
    else:
        seg = False

    if status == 'train':
        if os.path.exists(save_data_name):
            print('Loading saved data set...')
            with open(save_data_name, 'rb') as fp:
                data = pickle.load(fp)
            data.HP_num_layer = args.num_layer
            data.HP_iteration = args.num_iter
            data.label_comment = args.labelcomment
            data.result_file = args.resultfile
            data.HP_use_gaz = args.use_gaz
            data.HP_lr = args.lr
            data.HP_lr_decay = args.lr_decay
            data.HP_weight_decay = args.weight_decay
        else:
            data = Data()
            data.HP_gpu = gpu
            data.HP_use_char = False
            data.HP_batch_size = args.batch_size
            data.HP_num_layer = args.num_layer
            data.HP_iteration = args.num_iter
            data.use_bigram = False
            data.gaz_dropout = 0.5
            data.norm_gaz_emb = False
            data.HP_fix_gaz_emb = False
            data.label_comment = args.labelcomment
            data.result_file = args.resultfile
            data.HP_use_gaz = args.use_gaz
            data.HP_lr = args.lr
            data.HP_lr_decay = args.lr_decay
            data.HP_weight_decay = args.weight_decay
            data_initialization(data, gaz_file, train_file, dev_file, test_file)
            data.generate_instance_with_gaz(train_file,'train')
            data.generate_instance_with_gaz(dev_file,'dev')
            data.generate_instance_with_gaz(test_file,'test')
            data.build_word_pretrain_emb(char_emb)
            data.build_biword_pretrain_emb(bichar_emb)
            data.build_gaz_pretrain_emb(gaz_file)
            data.build_position_pretrain_emb(None)  ###
            print('Dumping data')
            with open(save_data_name, 'wb') as f:
                pickle.dump(data, f)

        train(data, save_model_dir, seg, False, False)
    elif status == 'test':
        data = load_data_setting(dset_dir)
        #data.generate_instance_with_gaz(dev_file,'dev')
        #load_model_decode(model_dir, data , 'dev', gpu, seg)
        data.generate_instance_with_gaz(test_file,'test')
        load_model_decode(model_dir, data, 'test', gpu, seg, print_label=True, print_gaz=True)
    elif status == 'decode':
        data = load_data_setting(dset_dir)
        data.generate_instance_with_gaz(raw_file,'raw')
        decode_results = load_model_decode(model_dir, data, 'raw', gpu, seg)
        data.write_decoded_results(output_file, decode_results, 'raw')
    else:
        print( "Invalid argument! Please use valid arguments! (train/test/decode)")
