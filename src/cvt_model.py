import torch.nn as nn
from .prediction_module import PredictionModule
from .encoder import Encoder
import torch
from torch.nn import functional as F

class CVTModel(nn.Module):
    def __init__(self,
                 num_words: int,
                 num_tags: int,
                 num_chars: int = 0,
                 device: object = None,
                 ### parameters for encoder goes here ###
                 word_embedding_size: int = 300,
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
                 pretrained_embeddings = None,
                 ### parameters for prediction module goes here ###
                 pre_output_layer_size: int = 200,
                 use_crf: bool = True,
                 ) -> None:

        super().__init__()

        self.num_words = num_words
        self.num_tags = num_tags
        self.num_chars = num_chars
        self.device = device

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
        self.pretrained_embeddings = pretrained_embeddings

        # prediction module parameters
        self.pre_output_layer_size = pre_output_layer_size
        self.use_crf = use_crf

    def initialize(self):
        # initialize model
        self.encoder = Encoder(self.num_words,
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
                               pretrained_embeddings=self.pretrained_embeddings
                               )
        # get input sizes for primary and auxiliary module
        self.wr_size, self.uni_input_size, self.bi_input_size = self.encoder.get_input_sizes()
        # primary module
        self.primary = PredictionModule("primary",
                                        self.uni_input_size * 2,
                                        self.num_tags,
                                        input_size_2=self.bi_input_size * 2,
                                        pre_output_layer_size=self.pre_output_layer_size,
                                        use_crf=self.use_crf)
        # auxiliary modules
        # [full, forwards, backwards, future, past]
        self.full = PredictionModule("full",
                                     self.uni_input_size * 2,
                                     self.num_tags,
                                     input_size_2=self.bi_input_size * 2,
                                     pre_output_layer_size=self.pre_output_layer_size,
                                     activate=False,
                                     use_crf=self.use_crf)
        self.forwards = PredictionModule("forwards",
                                         self.uni_input_size,
                                         self.num_tags,
                                         pre_output_layer_size=self.pre_output_layer_size,
                                         use_crf=self.use_crf)
        self.backwards = PredictionModule("backwards",
                                          self.uni_input_size,
                                          self.num_tags,
                                          pre_output_layer_size=self.pre_output_layer_size,
                                          use_crf=self.use_crf)
        self.future = PredictionModule("future",
                                       self.uni_input_size,
                                       self.num_tags,
                                       pre_output_layer_size=self.pre_output_layer_size,
                                       roll_direction=1, use_crf=self.use_crf)
        self.past = PredictionModule("past",
                                     self.uni_input_size,
                                     self.num_tags,
                                     pre_output_layer_size=self.pre_output_layer_size,
                                     roll_direction=-1, use_crf=self.use_crf)

    def set_device(self, device):
        self.device = device

    def forward(self, word_input, mode, char_input=None, mask=None, label=None, unk_id=None):
        wr, un, bi = self.encoder.get_representations(word_input, mode, char_tensor=char_input, mask=mask)
        loss = 0.
        if mode == "labeled":
            # unfreeze the main encoder and the primary module
            self._unfreeze_model()

            loss += self.primary.calculate_loss(un, label, repr_2=bi)
            if self.char_compose_method == "rnn":
                loss += 0.1 * self.encoder.compute_lm_loss(bi, word_input)
                loss += self.encoder.compute_similarity_loss(word_input, wr[:, :, :self.word_embedding_size],
                                                             wr[:, :, self.word_embedding_size:], unk_id=unk_id)

            del wr, un, bi
            return loss
        elif mode == "unlabeled":
            # freeze the primary module
            self._freeze_model()

            # calculate label primary
            if self.use_crf:
                label_primary = self.primary.predict(un, repr_2=bi)
                label_primary = torch.tensor(label_primary)
                label_primary = label_primary.to(self.device).detach()  # the primary module should not be backprop'ed
            else:
                label_primary = self.primary.project(un, repr_2=bi)
                label_primary = label_primary.detach()

            # calculate the prediction and loss of the five auxiliary module
            loss_full = self.full.calculate_loss(un, label_primary, repr_2=bi)

            loss_forwards = self.forwards.calculate_loss(un[:, :, :self.uni_input_size], label_primary)

            loss_backwards = self.backwards.calculate_loss(un[:, :, self.uni_input_size:], label_primary)

            loss_future = self.future.calculate_loss(un[:, :, :self.uni_input_size], label_primary)

            loss_past = self.past.calculate_loss(un[:, :, self.uni_input_size:], label_primary)

            loss += loss_full + loss_forwards + loss_backwards + loss_future + loss_past

            if self.char_compose_method == "rnn":
                loss += 0.1 * self.encoder.compute_lm_loss(bi, word_input)
                loss += self.encoder.compute_similarity_loss(word_input, wr[:, :, :self.word_embedding_size],
                                                             wr[:, :, self.word_embedding_size:], unk_id=unk_id)

            del label_primary, wr, un, bi
            torch.cuda.empty_cache()

            return loss

    def _freeze_model(self):
        # self.encoder.eval()
        self.primary.eval()
        # for params in self.encoder.parameters():
        #     params.requires_grad = False
        for params in self.primary.parameters():
            params.requires_grad = False

    def _unfreeze_model(self):
        # self.encoder.train()
        self.primary.train()
        # for params in self.encoder.parameters():
        #     params.requires_grad = True
        for params in self.primary.parameters():
            params.requires_grad = True
