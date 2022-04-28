# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import shutil ### (추가됨) 파일을 다루는 라이브러리 - 복사/삭제/이동/권한설정
import logging
import argparse
import random
from tqdm import tqdm, trange
import json

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from transformers import BertTokenizer, BertModel, BertForMaskedLM, AdamW

import cbert_utils
import train_text_classifier ## (추가됨) finetuning.py 와 달리 사용함 => Trainer 역할

#PYTORCH_PRETRAINED_BERT_CACHE = ".pytorch_pretrained_bert"

"""initialize logger"""
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

"""cuda or cpu"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def convert_ids_to_str(ids, tokenizer):
    """converts token_ids into str."""
    tokens = []
    for token_id in ids:
        token = tokenizer._convert_id_to_token(token_id)
        tokens.append(token)
    outputs = cbert_utils.rev_wordpiece(tokens)
    return outputs

def main():
    parser = argparse.ArgumentParser()

    ## required parameters
    parser.add_argument("--data_dir", default="datasets", type=str, ### 내가 데이터셋을 놓은 파일 경로
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default="aug_data", type=str,
                        help="The output dir for augmented dataset")
    parser.add_argument("--save_model_dir", default="cbert_model", type=str, ### fine-tuning 완료한 모델 저장한 경로 (모델 가중치)
                        help="The cache dir for saved model.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="The path of pretrained bert model.")
    parser.add_argument("--task_name", default="subj",type=str, ### StyleTransfer ?
                        help="The name of the task to train.")
    parser.add_argument("--max_seq_length", default=64, type=int, ### 64 또는 128
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case", default=False, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=32, type=int, ### 128 또는 64
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=9.0, type=float, ### 100
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', default=42, type=int, 
                        help="random seed for initialization")
    parser.add_argument('--sample_num', default=1, type=int,
                        help="sample number")
    parser.add_argument('--sample_ratio', default=7, type=int,
                        help="sample ratio")
    parser.add_argument('--gpu', default=0, type=int,
                        help="gpu id")
    parser.add_argument('--temp', default=1.0, type=float,
                        help="temperature")
    

    args = parser.parse_args()
    with open("global.config", 'r') as f:
        configs_dict = json.load(f)

    args.task_name = configs_dict.get("dataset")
    args.output_dir = args.output_dir + '_{}_{}_{}_{}'.format(args.sample_num, args.sample_ratio, args.gpu, args.temp)
    print(args)
    
    """prepare processors"""
    AugProcessor = cbert_utils.AugProcessor()
    processors = {
        ## you can add your processor here
        "TREC": AugProcessor,
        "stsa.fine": AugProcessor,
        "stsa.binary": AugProcessor,
        "mpqa": AugProcessor,
        "rt-polarity": AugProcessor,
        "subj": AugProcessor,

        "StyleTransfer" : AugProcessor, ### 추가
    }

    task_name = args.task_name
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name] ### processor 변수 : cbert_utils.AugProcessor()
    label_list = processor.get_labels(task_name) ### cbert_utils.AugProcessor() 클래스에서 '태스크 이름'에 해당하는 labels 정보

    ## prepare for model
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case) ### arg에서 설정한 BERT 모델의 토크나이저 (교체 예정)

    def load_model(model_name):
        weights_path = os.path.join(args.save_model_dir, model_name)
        model = torch.load(weights_path) ### fine-tuning한 모델 가중치 로드
        return model
    
    args.data_dir = os.path.join(args.data_dir, task_name) ### Origin Sentence 데이터셋 (인풋)
    args.output_dir = os.path.join(args.output_dir, task_name) ### Augmented Sentence 데이터셋 (아웃풋)
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    shutil.copytree("aug_data/{}".format(task_name), args.output_dir)

    ## prepare for training
    train_examples = processor.get_train_examples(args.data_dir) ### 증강시킬 대상 -> train.tsv 로 여기고 폴더 내 준비
    train_features, num_train_steps, train_dataloader = \
        cbert_utils.construct_train_dataloader(train_examples, label_list, args.max_seq_length, 
        args.train_batch_size, args.num_train_epochs, tokenizer, device)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    save_model_dir = os.path.join(args.save_model_dir, task_name)
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)
    MASK_id = cbert_utils.convert_tokens_to_ids(['[MASK]'], tokenizer)[0]

    origin_train_path = os.path.join(args.output_dir, "train_origin.tsv") ###
    save_train_path = os.path.join(args.output_dir, "train_output.tsv") ### (헷갈리니까 이름 변경함)
    #save_train_path = os.path.join(args.output_dir, "train.tsv")
    shutil.copy(origin_train_path, save_train_path) ### 파일을 복사하는 함수 - shutil.copy(원본 경로, 대상 경로)
    best_test_acc = train_text_classifier.train("aug_data_{}_{}_{}_{}".format(args.sample_num, args.sample_ratio, args.gpu, args.temp)) ### def train(dir="datasets", print_log=False):
    print("before augment best acc:{}".format(best_test_acc))

    for e in trange(int(args.num_train_epochs), desc="Epoch"):
        torch.cuda.empty_cache()
        cbert_name = "{}/BertForMaskedLM_{}_epoch_{}".format(task_name.lower(), task_name.lower(), e+1)
        model = load_model(cbert_name)
        model.cuda()
        shutil.copy(origin_train_path, save_train_path)
        save_train_file = open(save_train_path, 'a')
        tsv_writer = csv.writer(save_train_file, delimiter='\t') ### 원 문장을 copy해 만든 출력(증강/변환) 문장에 -> 모델이 predict한 결과 쓰기
        for _, batch in enumerate(train_dataloader):
            model.eval()
            batch = tuple(t.cuda() for t in batch)
            init_ids, _, input_mask, segment_ids, _ = batch
            input_lens = [sum(mask).item() for mask in input_mask]
            masked_idx = np.squeeze([np.random.randint(0, l, max(l//args.sample_ratio, 1)) for l in input_lens])
            for ids, idx in zip(init_ids, masked_idx):
                ids[idx] = MASK_id
            predictions = model(init_ids, input_mask, segment_ids)
            predictions = torch.nn.functional.softmax(predictions[0]/args.temp, dim=2)
            for ids, idx, preds, seg in zip(init_ids, masked_idx, predictions, segment_ids):
                preds = torch.multinomial(preds, args.sample_num, replacement=True)[idx]
                if len(preds.size()) == 2:
                    preds = torch.transpose(preds, 0, 1)
                for pred in preds:
                    ids[idx] = pred
                    new_str = convert_ids_to_str(ids.cpu().numpy(), tokenizer)
                    tsv_writer.writerow([new_str, seg[0].item()]) ### 원 문장을 copy해 만든 출력(증강/변환) 문장에 -> 모델이 predict한 결과 쓰기
            torch.cuda.empty_cache()
        predictions = predictions.detach().cpu()
        model.cpu()
        torch.cuda.empty_cache()
        bak_train_path = os.path.join(args.output_dir, "train_epoch_{}.tsv".format(e)) ### 몇 번째 epoch가 predict이 가장 잘 되는지, 수치적으로 계산해서 알려줌 (best_test_acc (??))
        shutil.copy(save_train_path, bak_train_path)
        best_test_acc = train_text_classifier.train("aug_data_{}_{}_{}_{}".format(args.sample_num, args.sample_ratio, args.gpu, args.temp))
        print("epoch {} augment best acc:{}".format(e, best_test_acc))

if __name__ == "__main__":
    main()
