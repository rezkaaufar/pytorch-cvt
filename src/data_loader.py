import logging
import typing
import random
import torch

from torchtext.data import Dataset, Example, Field, NestedField, BucketIterator
from src.iterators import GroupedBucketIterator

class Data(object):

    WORDS_NAME = "words"
    LAB_NAME = "lab"
    CHAR_NAME = "char"

    def __init__(self,
                 train_path: str,
                 unlabeled_path: str,
                 semi_supervised: bool,
                 dev_path: str = None,
                 test_path: str= None,
                 batch_size: int= 32,
                 device: object = None,
                 logger: typing.Optional[logging.Logger] = None,
                ) -> None:
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s - %(name)s - %(message)s'))
            logger.addHandler(handler)

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.unlabeled_path = unlabeled_path
        self.batch_size = batch_size
        self.semi_supervised = semi_supervised
        self.device = device
        self.logger = logger

    def initialize(self):
        ## initialize fields and create dataset ##
        self._init_fields()
        self._read_sentences()
        self.train = self._make_bucket_iterator(self._make_dataset(False),
                                                batch_size=self.batch_size, device=self.device)
        self.dev = self._make_bucket_iterator(self._make_dataset(False, which="dev"),
                                              batch_size=self.batch_size, device=self.device)
        self.test = self._make_bucket_iterator(self._make_dataset(False, which="test"),
                                               batch_size=self.batch_size, device=self.device)
        # self.unlabeled_train = self._make_bucket_iterator(self._make_dataset(True),
        #                                                   batch_size=self.batch_size, device=self.device)
        self.unlabeled_data = self._make_dataset(True)
        self._build_vocabularies()

    def _read_sentences(self):
        self.train_sentences = []
        with open(self.train_path) as f:
            for line in f:
                self.train_sentences.append(line.replace("\n", ""))
        self.logger.info('{} train sentences successfully read'.format(len(self.train_sentences)))

        self.dev_sentences= []
        with open(self.dev_path) as f:
            for line in f:
                self.dev_sentences.append(line.replace("\n", ""))
        self.logger.info('{} dev sentences successfully read'.format(len(self.dev_sentences)))

        self.unlabeled_sentences = []
        temp = []
        with open(self.unlabeled_path) as f:
            for line in f:
                sen_len = len(line.split())
                if sen_len > 0 and sen_len <= 20:
                    temp.append(line.replace("\n", ""))
        #self.unlabeled_sentences = random.sample(temp, 101420)
        self.unlabeled_sentences = temp
        self.logger.info('{} unlabeled sentences successfully read'.format(len(self.unlabeled_sentences)))

        self.test_sentences = []
        with open(self.test_path) as f:
            for line in f:
                self.test_sentences.append(line.replace("\n", ""))
        self.logger.info('{} test sentences successfully read'.format(len(self.train_sentences)))

    def _init_fields(self):
        self.words = Field(batch_first=True, init_token='<s>', eos_token='</s>')
        self.lab = Field(batch_first=True, unk_token=None, pad_token=None)
        # self.char = NestedField(Field(batch_first=True, tokenize=list, unk_token='<cunk>')
        #                         , init_token='<s>', eos_token='</s>')
        self.char = NestedField(Field(batch_first=True, tokenize=list, unk_token='<cunk>',init_token='<w>', eos_token='</w>')
                                , init_token='<s>', eos_token='</s>')

        self.labeled_fields = [(self.WORDS_NAME, self.words), (self.CHAR_NAME, self.char),
                               (self.LAB_NAME, self.lab)]
        self.unlabeled_fields = [(self.WORDS_NAME, self.words), (self.CHAR_NAME, self.char)]
        self.logger.info('fields initialized successfully')

    def _make_dataset(self, unlabeled, which=None) -> Dataset:
        if not unlabeled:
            sentences = self.train_sentences
            if which == "dev":
                sentences = self.dev_sentences
            elif which == "test":
                sentences = self.test_sentences
            examples = [self._make_example(s) for s in sentences]
            return Dataset(examples, self.labeled_fields)
        else:
            sentences = self.unlabeled_sentences
            examples = [self._make_example_unlabeled(s) for s in sentences]
            return Dataset(examples, self.unlabeled_fields)

    def _make_example(self, sent) -> Example:
        cols = sent.split("\t")
        words = [word for word in cols[0].split()]
        tags = [tag for tag in cols[1].split()]
        return Example.fromlist([words, words, tags], self.labeled_fields)

    def _make_example_unlabeled(self, sent) -> Example:
        words = [word for word in sent.split()]
        return Example.fromlist([words, words], self.unlabeled_fields)

    def _make_bucket_iterator(self, data, batch_size=32, device=None):
        # return BucketIterator(
        #     dataset=data, batch_size=batch_size,
        #     sort=False, sort_within_batch=True,
        #     sort_key=lambda x: len(x.words),
        #     device=device, repeat=False)
        return GroupedBucketIterator(data, batch_size,
                                     lambda ex: len(ex.words), device=device)

    def _build_vocabularies(self):
        self.words.build_vocab(self.train.dataset)
        self.lab.build_vocab(self.train.dataset)
        self.char.build_vocab(self.train.dataset)

        self.num_words = len(self.words.vocab)
        self.num_tags = len(self.lab.vocab)
        self.num_char = len(self.char.vocab)

        self.logger.info('Found %d words, %d chars, and %d tags for both the labeled and unlabeled dataset',
                         self.num_words, self.num_char, self.num_tags)



    def _get_unlabeled_sentences(self):
        while True:
            for us in self.unlabeled_sentences:
                yield us

    def _get_unlabeled_examples(self):
        #while True:
            lines = []
            for words in self._get_unlabeled_sentences():
                lines.append(words)
                if len(lines) >= 10142:
                    yield [self._make_example_unlabeled(line) for line in lines]
                    lines = []

    def _endless_unlabeled(self):
        #while True:
            for ex in self._get_unlabeled_examples():
                unlabeled_iterator = self._make_bucket_iterator(Dataset(ex, self.unlabeled_fields),
                                                 batch_size=self.batch_size,
                                                 device=self.device)
                yield unlabeled_iterator
                del unlabeled_iterator
                torch.cuda.empty_cache()



    def _endless_minibatch(self, data):
        while True:
            for i, batch in enumerate(data):
                yield batch

    def get_alternating_minibatch(self):
        # self._create_dataset()
        while True:
            for iter in self._endless_unlabeled():
                for mb in iter:
                    yield next(self._endless_minibatch(self.train)), "labeled"
                    if self.semi_supervised:
                        yield mb, "unlabeled"

    def get_input_sizes(self):
        return self.num_words, self.num_char, self.num_tags

    def get_pad_token_id(self):
        return self.char.vocab.stoi[self.char.pad_token]

    def get_unk_token_id(self):
        return self.char.vocab.stoi[self.char.unk_token]

    def get_train_sentences_length(self):
        return len(self.train_sentences)