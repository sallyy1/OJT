# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import logging
import argparse
import random
import json
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

'''
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AdamW, WarmupLinearSchedule
#import train_text_classifier_new ## (finetuning.py 에서는 사용 안함)
'''
### 모델 및 토크나이저 변경
'''
### [1]
from kobert_tokenizer import KoBERTTokenizer
### 1) 일반적인 토크나이저 사용시 (예: inference)
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
### 2) BPE-dropout 이용시 - 서비스에 적합한 띄어쓰기에 강건한 모델로 튜닝 가능
#tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})

from transformers import BertModel, BertForMaskedLM
#model = BertModel.from_pretrained('skt/kobert-base-v1')
model = BertForMaskedLM.from_pretrained('skt/kobert-base-v1')
'''

### [2]
from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert-lm")
model = AutoModelForMaskedLM.from_pretrained("monologg/kobert-lm")

from transformers import AdamW, WarmupLinearSchedule
###


import cbert_utils

"""initialize logger"""
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

"""cuda or cpu"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default="datasets", type=str, ### 내가 데이터셋을 놓은 파일 경로
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default="aug_data", type=str,
                        help="The output dir for augmented dataset.")
    parser.add_argument("--save_model_dir", default="cbert_model", type=str, ### fine-tuning 완료한 모델 저장한 경로 (모델 가중치)
                        help="The cache dir for saved model.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="The path of pretrained bert model.")
    parser.add_argument("--task_name", default="subj", type=str, ### StyleTransfer ?
                        help="The name of the task to train.")
    parser.add_argument("--max_seq_length", default=64, type=int, ### 64 또는 128
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequence longer than this will be truncated, and sequences shorter \n"
                             "than this wille be padded.")
    parser.add_argument("--do_lower_case", default=False, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=32, type=int, ### 128 또는 64
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float, ### 100
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for."
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--save_every_epoch", default=True, action='store_true') ### (cbert_augdata.py 에는 없고 cbert_finetune.py 에만 추가됨)

    args = parser.parse_args()
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

        "StyleTransfer": AugProcessor,  ### 추가
    }

    task_name = args.task_name
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]  ### processor 변수 : cbert_utils.AugProcessor()
    label_list = processor.get_labels(task_name)  ### cbert_utils.AugProcessor() 클래스에서 '태스크 이름'에 해당하는 labels 정보

    """prepare model"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ## leveraging lastest bert module in Transformers to load pre-trained model tokenizer
    #tokenizer = BertTokenizer.from_pretrained(args.bert_model) ### arg에서 설정한 BERT 모델의 토크나이저
    
    ## leveraging lastest bert module in Transformers to load pre-trained model (weights)
    #model = BertForMaskedLM.from_pretrained(args.bert_model) ### arg에서 설정한 BERT 모델 (마스킹용 Masked LM으로 사용할 모델 (?))

    if task_name == 'stsa.fine':
        model.bert.embeddings.token_type_embeddings = torch.nn.Embedding(5, 768)
        model.bert.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=0.02)
    elif task_name == 'TREC':
        model.bert.embeddings.token_type_embeddings = torch.nn.Embedding(6, 768)
        model.bert.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=0.02)

    args.data_dir = os.path.join(args.data_dir, task_name)
    args.output_dir = os.path.join(args.output_dir, task_name)
    os.makedirs(args.output_dir, exist_ok=True)

    ### 이 부분 바꾸어주어야 함 - 데이터 읽어오기
    ### 저자들은 train.tsv, dev.tsv를 사전에 따로 준비해, fine-tuning 훈련 시 read 함
    train_examples = processor.get_train_examples(args.data_dir)
    train_features, num_train_steps, train_dataloader = \
        cbert_utils.construct_train_dataloader(train_examples, label_list, args.max_seq_length, 
        args.train_batch_size, args.num_train_epochs, tokenizer, device)

    ## if you have a GPU, put everything on cuda
    model.cuda()

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    ## in Transformers, optimizer and schedules are splitted and instantiated like this:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grounded_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grounded_parameters, lr=args.learning_rate, correct_bias=False)
    model.train()

    os.makedirs(args.save_model_dir, exist_ok=True)
    save_model_dir = os.path.join(args.save_model_dir, task_name)
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)

    for e in trange(int(args.num_train_epochs), desc="Epoch"):
        avg_loss = 0.

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.cuda() for t in batch)
            _, input_ids, input_mask, segment_ids, masked_ids = batch
            """train generator at each batch"""
            optimizer.zero_grad()
            outputs = model(input_ids, input_mask, segment_ids,
                    masked_lm_labels=masked_ids)
            loss = outputs[0]
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()
            if (step + 1) % 50 == 0:
                print("avg_loss: {}".format(avg_loss / 50))
                avg_loss = 0
        if args.save_every_epoch: ### arg 설정에 따라 매 에폭마다 저장
            save_model_name = "BertForMaskedLM_" + task_name + "_epoch_" + str(e + 1)
            save_model_path = os.path.join(save_model_dir, save_model_name) ## 경로 : "../style_cbert_model/BertForMaskedLM_StyleTransfer_epoch_해당 에폭"
            torch.save(model, save_model_path)
        else:
            if (e + 1) % 10 == 0: ### arg 설정에 따라 10 에폭마다 저장
                save_model_name = "BertForMaskedLM_" + task_name + "_epoch_" + str(e + 1)
                save_model_path = os.path.join(save_model_dir, save_model_name)
                torch.save(model, save_model_path)

if __name__ == "__main__":
    main()