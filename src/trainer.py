import typing
import logging
import torch

from src.data_loader import Data
from src.cvt_model import CVTModel
from torch import optim
from src.evaluator import Evaluator
from src.span_evaluator import SpanEvaluator
from src.artifacts_manager import ArtifactsManager
from tensorboardX import SummaryWriter
import numpy as np
import math

class Trainer(object):
    def __init__(self,
                 train_path: str,
                 dev_path: str,
                 test_path: str,
                 unlabeled_path: str,
                 save_to: str = "models/",
                 semi_supervised: bool = True,
                 resume_training: bool = False,
                 save_every: int = 1000,
                 print_every: int =101,
                 batch_size: int = 32,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 5e-3,
                 ### parameters for encoder goes here ###
                 word_embedding_size: int = 100,
                 dropout_lab: float = 0.5,
                 dropout_unlab: float = 0.8,
                 char_compose_method: str = "cnn",
                 uni_encoder_hidden_size: int = 300,
                 bi_encoder_hidden_size: int = 200,
                 uses_char_embeddings: bool = True,
                 char_embedding_size: int = 50,
                 char_encoder_hidden_size: int = 200,
                 out_channels_size: int = 100,
                 kernel_size_list: [int] = None,
                 lm_layer_size: int = 50,
                 lm_max_vocab_size: int = 7500,
                 pretrained_embeddings_path: str = None,
                 ### parameters for prediction module goes here ###
                 pre_output_layer_size: int = 200,
                 use_crf: bool = True,
                 logger: typing.Optional[logging.Logger] = None
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
        self.save_to = save_to
        self.semi_supervised = semi_supervised
        self.resume_training = resume_training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.save_every = save_every
        self.print_every = print_every
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.logger = logger

        # encoders parameters
        self.word_embedding_size = word_embedding_size
        self.dropout_lab_prob = dropout_lab
        self.dropout_unlab_prob = dropout_unlab
        self.char_compose_method = char_compose_method
        self.uni_encoder_hidden_size = uni_encoder_hidden_size
        self.bi_encoder_hidden_size = bi_encoder_hidden_size
        self.uses_char_embeddings = uses_char_embeddings
        self.char_embedding_size = char_embedding_size
        self.char_encoder_hidden_size = char_encoder_hidden_size
        self.out_channels_size = out_channels_size
        self.kernel_size_list = kernel_size_list

        if self.kernel_size_list is None:
            self.kernel_size_list = [2, 3, 4]

        self.lm_layer_size = lm_layer_size
        self.lm_max_vocab_size = lm_max_vocab_size
        self.pretrained_embeddings_path = pretrained_embeddings_path

        # prediction module parameters
        self.pre_output_layer_size = pre_output_layer_size
        self.use_crf = use_crf


    def _initial_preparation(self):
        self.data = Data(train_path=self.train_path,
                         unlabeled_path =self.unlabeled_path,
                         semi_supervised=self.semi_supervised,
                         dev_path=self.dev_path,
                         test_path=self.test_path,
                         batch_size=self.batch_size,
                         device=self.device)
        self.data.initialize()
        self.pad_token_id = self.data.get_pad_token_id()
        self.unk_token_id = self.data.get_unk_token_id()
        self.num_words, self.num_chars, self.num_tags = self.data.get_input_sizes()
        self.len_train = self.data.get_train_sentences_length()
        self.artifacts_manager = ArtifactsManager(self.save_to,
                                                  device=self.device)
        self.writer = SummaryWriter(log_dir='runs/cvt')
        self.pretrained_embeddings = self._load_pretrained_embeddings()

    def _load_pretrained_embeddings(self):
        if self.pretrained_embeddings_path is not None:
            idx = 0
            vectors = {}
            max = 0
            with open(self.pretrained_embeddings_path, 'r') as f:
                for l in f:
                    if max == 0:
                        max += 1
                        continue
                    line = l.split()
                    word = line[0]
                    idx += 1
                    vect = np.array(line[1:]).astype(np.float)
                    vectors[word] = vect
                    # print(vect)
                    max += 1
            weights_matrix = np.zeros((self.num_words, self.word_embedding_size))
            words_found = 0

            for i in range(self.num_words):
                try:
                    weights_matrix[i] = vectors[self.data.words.vocab.itos[i]]
                    words_found += 1
                except KeyError:
                    weights_matrix[i] = np.random.normal(scale=0.6, size=(self.word_embedding_size,))

            return weights_matrix

        else:

            return None

    def _model_preparation(self):
        if not self.resume_training:
            self.cvt = CVTModel(self.num_words,
                                self.num_tags,
                                num_chars=self.num_chars,
                                word_embedding_size=self.word_embedding_size,
                                dropout_lab=self.dropout_lab_prob,
                                dropout_unlab=self.dropout_unlab_prob,
                                char_compose_method=self.char_compose_method,
                                uni_encoder_hidden_size=self.uni_encoder_hidden_size,
                                bi_encoder_hidden_size=self.bi_encoder_hidden_size,
                                uses_char_embeddings=self.uses_char_embeddings,
                                char_embedding_size=self.char_embedding_size,
                                char_encoder_hidden_size=self.char_encoder_hidden_size,
                                out_channels_size=self.out_channels_size,
                                kernel_size_list=self.kernel_size_list,
                                lm_layer_size=self.lm_layer_size,
                                lm_max_vocab_size=self.lm_max_vocab_size,
                                pre_output_layer_size=self.pre_output_layer_size,
                                pretrained_embeddings=self.pretrained_embeddings,
                                use_crf=self.use_crf,
                                device=self.device)
            self.cvt.initialize()
            self.cvt.to(self.device)
            # artifacts_manager.save_model_and_artifacts(cvt)
        else:
            self.cvt = self.artifacts_manager.load_model_and_artifacts()
            self.cvt.to(self.device)

        # self.optimizer = optim.Adam(self.cvt.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.optimizer = optim.SGD(self.cvt.parameters(), lr=0.5, momentum=0.9)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, verbose=True)
        # max_epoch = 1200
        # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoch)
        # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=800, after_scheduler=scheduler_cosine)


        #self.evaluator = Evaluator()
        self.evaluator = SpanEvaluator(use_crf=self.use_crf)

    def _endless_train(self):
        count = 0
        running_loss = 0.0
        epoch_approx = 0

        multiplier = lambda count: min(count, 5000) * (1.0 / (1 + 0.005 * math.sqrt(count)))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=multiplier)

        try:
            for mb, mode in self.data.get_alternating_minibatch():
                self.scheduler.step(count)
                self.cvt.train()
                self.optimizer.zero_grad()
                word_input = getattr(mb, "words")
                char_input = getattr(mb, "char")
                label = None
                if mode == "labeled":
                    label = getattr(mb, "lab")
                mask = char_input != self.pad_token_id
                unk_id = self.unk_token_id

                # try:
                loss = self.cvt(word_input, mode, char_input=char_input, label=label, mask=mask, unk_id=unk_id)

                if mode == "labeled":
                    running_loss += loss.item()
                    self.writer.add_scalar('data/labeled_loss', loss.item(), count)
                elif mode == "unlabeled":
                    self.writer.add_scalar('data/unlabeled_loss', loss.item(), count)

                loss.backward()
                self.optimizer.step()

                if count % self.print_every == 0 and count != 0:
                    self.cvt.eval()
                    precision, recall, f1 = self.evaluator.evaluate_on_data(self.cvt, self.data, which="dev")
                    self.writer.add_scalar('data/dev_precision', precision, count)
                    self.writer.add_scalar('data/dev_recall', recall, count)
                    self.writer.add_scalar('data/dev_f1', f1, count)
                    self.logger.info("dev precision: {}, dev recall: {}, dev f1: {}, loss: {} at training step {}".
                                      format(precision, recall, f1, loss, count))
                    # self.logger.info("{} mode loss: {}".format(mode, loss.item()))
                if count % self.save_every == 0 and count != 0:
                    self.artifacts_manager.save_model_and_artifacts(self.cvt)
                if count % (round(self.len_train * 2 / self.batch_size)) == 0 and count != 0:  # this roughly equivalent to one epoch
                    epoch_approx += 1
                    self.logger.info("approximately {} epoch has passed!".format(epoch_approx))
                    # self.scheduler.step(running_loss)
                    running_loss = 0

                del loss

                # except RuntimeError as e:
                #     if 'out of memory' in str(e):
                #         self.logger.info("ran out of memory!, skipping this {} batch".format(mode))
                #         # for p in self.cvt.parameters():
                #         #     if p.grad is not None:
                #         #         del p.grad
                #         torch.cuda.empty_cache()

                count += 1

        except KeyboardInterrupt:
            self.logger.info("Interrupting training, evaluating on test set")
            self.cvt.eval()
            precision, recall, f1 = self.evaluator.evaluate_on_data(self.cvt, self.data, which="test")
            self.writer.add_scalar('data/test_precision', precision, count)
            self.writer.add_scalar('data/test_recall', recall, count)
            self.writer.add_scalar('data/test_f1', f1, count)
            self.logger.info("test precision: {}, test recall: {}, test f1: {} at training step {}".
                             format(precision, recall, f1, count))
            self.artifacts_manager.save_model_and_artifacts(self.cvt)

        self.writer.close()

    def run(self):
        self._initial_preparation()
        self._model_preparation()
        self._endless_train()
