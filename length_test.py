# from torchtext.data import Field, Example, Dataset
# from torchtext.vocab import Vectors
# from collections import Counter
# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# labeled_path = "labeled_data/ner/sentence-test.conll"
# vocab_path = "pretrained_vectors/nlpl/vocab.txt"
#
# sentences = []
# with open(labeled_path, "r") as f:
#     for line in f:
#         sentences.append(line.replace("\n",""))
#
# examples = []
# words_field = Field(batch_first=True, init_token='<s>', eos_token='</s>')
# for s in sentences:
#     cols = s.split("\t")
#     words = [word for word in cols[0].split()]
#     tags = [tag for tag in cols[1].split()]
#     examples.append(Example.fromlist([words], [("words",words_field)]))
# data = Dataset(examples, [("words",words_field)])
#
# vocabs = []
# with open(vocab_path, "r") as f:
#     for line in f:
#         vocabs.append(line.replace("\n",""))
# print(len(vocabs))
#
# cnt = Counter()
# for v in vocabs:
#     cnt[v] += 1
#
# # vec = Vectors('pretrained_vectors/nlpl/model.txt', cache='pretrained_vectors/cache/')
# # words_field.build_vocab(cnt)
# # for i in range(len(words_field.vocab)):
# #     print(words_field.vocab.itos[i])
#
# for i in data:
#     print(i.to(device))

import torch
import numpy as np
import pickle

from src.data_loader import Data
data = Data(train_path="labeled_data/ner/sentence-train.conll",
                         unlabeled_path ="unlabeled_data/informal/test.txt",
                         semi_supervised=True,
                         dev_path="labeled_data/ner/sentence-dev.conll",
                         test_path="labeled_data/ner/sentence-test.conll",
                         batch_size=5,
                         device=None)

data.initialize()


# for b in data._unlabeled_data_lazy_loader():
#     for mb in b:
#         print(mb)

# for ex in data._get_unlabeled_examples():
#     print(ex)

# for i in range(len(data.words.vocab)):
#     print(i, data.words.vocab.itos[i])

print(data.words.vocab.stoi["<pad>"])
print(data.words.vocab.stoi["<unk>"])
# for mb, mode in data.get_alternating_minibatch():
#     word_input = getattr(mb, "words")
#     char_input = getattr(mb, "char")
#     label = None
#     if mode == "labeled":
#         label = getattr(mb, "lab")
#         print(label)
    # if mode == "unlabeled":
    #     print(word_input)