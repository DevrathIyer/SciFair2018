# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from a PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import json
import re
import math

import numpy as np

import gensim
from gensim.test.utils import common_texts

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition

import scipy.cluster.hierarchy as hcluster
from scipy.cluster.hierarchy import fclusterdata
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

import pickle
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def dummy_fun(doc):
    return doc

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None)

def softmax(inputs):
    return np.exp(inputs) / float(sum(np.exp(inputs)))

def sigmoid(x):
    return (5 / (1 + math.exp(-15*x+7.5)))

m = interp1d([0.1,.8],[0,2])
word2vec_global = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=50000)
word2vec_local=None
def wmd(sentence1,sentence2):
    sum = 0
    num = 0
    #print(sentence1["word_embeds"])
    #print(sentence2["word_embeds"])
    #diff_matrix = [[1 for token1 in sentence2["word_embeds"]] for token2 in sentence1["word_embeds"]]
    diff_matrix = [[sigmoid(abs(token1 - token2)) for token1 in sentence2["word_embeds"]] for token2 in sentence1["word_embeds"]]
    for i, token1 in enumerate(sentence1['tokens']):
        for j, token2 in enumerate(sentence2['tokens']):
            if token1 in word2vec_global.vocab and token2 in word2vec_global.vocab:
                diff_matrix[i][j] /= word2vec_global.similarity(token1, token2)
    #diff_matrix = [[word2vec_local.similarity(token1,token2) for token1 in sentence1['tokens']] for token2 in sentence2['tokens']]
    diff_matrix_args = np.argsort(diff_matrix)
    """
    for i,token1 in enumerate(sentence1['tokens']):
        for j,token2 in enumerate(sentence2['tokens']):
            if token1 in word2vec_global.vocab and token2 in word2vec_global.vocab:
                diff_matrix[i][j] = word2vec_global.similarity(token1,token2)
    diff_matrix_args = np.argsort(diff_matrix)
    for i in range(0, len(sentence1["tokens"])):
       print("{}: {}, {}".format(sentence1["tzokens"][i], sentence2["tokens"][diff_matrix_args[i][0]],word2vec_global.similarity(sentence1["tokens"][i], sentence2["tokens"][diff_matrix_args[i][0]])))
    print()
    #print()
    """
    idfs = [tfidf.idf_[tfidf.vocabulary_[token]] for token in sentence1["tokens"]]
    idf_softmax = softmax(idfs)
    matched = [[-1 for token in sentence1["tokens"]],[-1 for token in sentence2["tokens"]]]
    for (i,token) in enumerate(sentence1['tokens']):
        flag = False
        for j in range(0,len(sentence2['tokens'])):
            if matched[1][int(diff_matrix_args[i][j])] == -1 or diff_matrix[i][int(diff_matrix_args[i][j])] > diff_matrix[matched[1][int(diff_matrix_args[i][j])]][int(diff_matrix_args[i][j])]:
                matched[1][int(diff_matrix_args[i][j])] = i
                matched[0][i] = int(diff_matrix_args[i][j])
                break

    for i in range(0,len(sentence1["tokens"])):
        min = 100000
        if sentence1["tokens"][i] in word2vec_global.vocab:
            if sentence2["tokens"][matched[0][i]] in word2vec_global.vocab:
                #print(idf_softmax[i] * abs(word2vec.similarity(sentence1["tokens"][i], sentence2["tokens"][matched[0][i]])))
                sum += idf_softmax[i]*abs(word2vec_global.similarity(sentence1["tokens"][i],sentence2["tokens"][matched[0][i]]))
                num+=1
    if num == 0:
        return 100000
    else:
        return sum

def remove_first_principal_component(X):
    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
    svd.fit(X)
    pc = svd.components_
    XX = X - X.dot(pc.transpose()) * pc
    return XX

class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        print(input_type_ids)
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))

    #word2vec_model = gensim.models.Word2Vec([example.tokens for example in features], size=100, window=5, min_count=1, workers=4)
    #word2vec_model.train([example.tokens for example in features], total_examples=len([example.tokens for example in features]), epochs=10)
    #global word2vec_local
    #word2vec_local = word2vec_model.wv
    tfidf.fit([example.tokens for example in features])
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()
            text_a = None
            text_b = None
            m = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", line)
            print(m)
            if len(m) == 1:
                text_a = line
            else:
                text_a = m[0]
                text_b = m[1]
            print(text_a)
            print(text_b)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    print("NUMBER OF EXAMPLES: {}".format(len(examples)))
    return examples


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    ## Other parameters
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--layers", default="-1,-2,-3,-4", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=1000, type=int, help="Batch size for predictions.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help = "local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

    layer_indexes = [int(x) for x in args.layers.split(",")]

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    examples = read_examples(args.input_file)

    features = convert_examples_to_features(
        examples=examples, seq_length=args.max_seq_length, tokenizer=tokenizer)

    with open('text.out','wb') as f:
        for feature in features:
            pickle.dump(feature, f)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model = BertModel.from_pretrained(args.bert_model)
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    model.eval()

    a = 0.001

    for input_ids, input_mask, example_indices in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)

        all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
        all_encoder_layers = all_encoder_layers

        sentences = [{} for i in range(0,len(example_indices))]

        for b, example_index in enumerate(example_indices):
            feature = features[example_index.item()]
            unique_id = int(feature.unique_id)
            feature = unique_id_to_feature[unique_id]
            sentences[b]["tokens"] = [token for token in feature.tokens]# if token in word2vec_global.vocab and len(token) >= 3]
            sentences[b]['word_embeds'] = []
            for (i, token) in enumerate([token for token in feature.tokens]):# if token in word2vec_global.vocab and len(token) >= 3]):
                embed = 0
                for (j, layer_index) in enumerate(layer_indexes):
                    layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
                    layer_output = layer_output[b]
                    embed +=round(sum(x.item() for x in layer_output[i])/len(layer_output[i]),6)
                embed /= len(layer_indexes)
                embed *= tfidf.idf_[tfidf.vocabulary_[token]]
                sentences[b]["word_embeds"].append(embed)
                #print("{}: {}".format(token,embed))

    print("Finished embeds...")


    #wmd_matrix = np.array([[round(wmd(sentence1, sentence2) * 100/wmd(sentence1, sentence1), 0) for sentence2 in sentences] for sentence1 in sentences], np.int32)
    wmd_matrix = np.array([[wmd(sentence1, sentence2) for sentence2 in sentences] for sentence1 in sentences])
    print(wmd_matrix)
    #print("Finished wmd_matrix...")/wmd(sentence1, sentence1)
    pca = decomposition.PCA(n_components=1)
    pca.fit(wmd_matrix)
    X = pca.transform(wmd_matrix)
    fclust2 = fclusterdata(wmd_matrix, t=1.0,method='complete')
    centroids, _ = kmeans(wmd_matrix,4)
    idx, _ = vq(wmd_matrix, centroids)


    print()
    print()

    print("PCA")
    print(X)
    print(pca.explained_variance_ratio_)
    print()

    plt.plot(X, np.zeros_like(X), 'x')
    plt.show()

    print("KMEANS CLUSTERING")
    ordered = np.argsort(idx)
    num = 1
    for i, arg in enumerate(ordered):
        if i == 0:
            print("CLUSTER 1:")
        elif idx[arg] > num:
            num += 1
            print("\nCLUSTER {}".format(num))
        print(sentences[arg]["tokens"])

    print()
    print("F CLUSTERING")
    ordered = np.argsort(fclust2)
    num = 1
    for i,arg in enumerate(ordered):
        if i == 0:
            print("CLUSTER 1:")
        elif fclust2[arg] > num:
            num+=1
            print("\nCLUSTER {}".format(num))
        print(sentences[arg]["tokens"])

    #print(wmd_matrix)

if __name__ == "__main__":
    main()
