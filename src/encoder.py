import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

class Encoder(nn.Module):
    def __init__(self,
                 num_words: int,
                 num_tags: int,
                 num_chars: int = 0,
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
                 ) -> None:

        super().__init__()

        # Attributes
        self.num_words = num_words
        self.num_tags = num_tags
        self.num_chars = num_chars
        self.word_embedding_size = word_embedding_size
        self.dropout_lab_prob = dropout_lab
        self.dropout_unlab_prob = dropout_unlab
        self.char_compose_method = char_compose_method
        self.char_dropout_layer = nn.Dropout(dropout_lab)
        self.uni_dropout_layer = nn.Dropout(dropout_lab)
        self.bi_dropout_layer = nn.Dropout(dropout_lab)
        self.uni_encoder_hidden_size = uni_encoder_hidden_size
        self.bi_encoder_hidden_size = bi_encoder_hidden_size
        self.uses_char_embeddings = uses_char_embeddings
        self.char_embedding_size = char_embedding_size
        self.char_encoder_hidden_size = char_encoder_hidden_size
        self.out_channels_size = out_channels_size
        self.kernel_size_list = kernel_size_list

        if self.kernel_size_list is None:
            self.kernel_size_list = [2,3,4]

        self.lm_layer_size = lm_layer_size
        self.lm_max_vocab_size = lm_max_vocab_size

        # Embeddings
        self.word_embedding = nn.Embedding(num_words, word_embedding_size)
        if pretrained_embeddings is not None:
            wm = torch.Tensor(pretrained_embeddings)
            self.word_embedding.from_pretrained(wm)
        if self.uses_char_embeddings:
            self.char_embedding = nn.Embedding(num_chars, char_embedding_size)

        # Char Embeddings Processor
        if self.uses_char_embeddings:
            if self.char_compose_method == "cnn":
                in_channels_size = char_embedding_size
                self.conv = nn.ModuleList([nn.Conv1d(in_channels_size, out_channels_size, kernel_size,
                                                 padding=kernel_size // 2) for kernel_size in self.kernel_size_list])
            elif self.char_compose_method == "rnn":
                self.char_encoder = nn.LSTM(
                    char_embedding_size,
                    char_encoder_hidden_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.,
                    bidirectional=True,
                )
                self.char_projection = nn.Sequential(
                    nn.Linear(2 * char_encoder_hidden_size, word_embedding_size),
                    nn.Tanh(),
                )

        # Encoder
        encoder_input_size = word_embedding_size
        if self.uses_char_embeddings:
            if self.char_compose_method == "cnn":
                encoder_input_size = word_embedding_size + self.out_channels_size * len(self.kernel_size_list)
            elif self.char_compose_method == "rnn":
                encoder_input_size *= 2
        self.encoder_input_size = encoder_input_size

        # first layer of the encoder
        self.uni_encoder = nn.LSTM(
            encoder_input_size,
            uni_encoder_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.,
            bidirectional=True,
        )

        # second layer of the encoder
        self.bi_encoder = nn.LSTM(
            uni_encoder_hidden_size * 2,
            bi_encoder_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.,
            bidirectional=True,
        )

        # additional loss for rnn char compose method
        if self.uses_char_embeddings:
            if self.char_compose_method == "rnn":
                lm_output_size = min(num_words, lm_max_vocab_size) + 1
                self.lm_ff_fwd = nn.Sequential(
                    nn.Linear(bi_encoder_hidden_size, lm_layer_size),
                    nn.Tanh(),
                    nn.Linear(lm_layer_size, lm_output_size),
                )
                self.lm_ff_bwd = nn.Sequential(
                    nn.Linear(bi_encoder_hidden_size, lm_layer_size),
                    nn.Tanh(),
                    nn.Linear(lm_layer_size, lm_output_size),
                )

    def _get_word_representations(self, word_tensor, mode, char_tensor=None, mask=None):
        if mode == "unlabeled":
            self.char_dropout_layer.p = self.dropout_unlab_prob
        elif mode == "labeled":
            self.char_dropout_layer.p = self.dropout_lab_prob
        word_emb = self.word_embedding(word_tensor)

        ## compose word embeddings ##
        if self.uses_char_embeddings:
            char_embs = self._compose_char_embeddings(char_tensor, mask=mask)
            return torch.cat([word_emb] + char_embs, dim=2)
        else:
            return word_emb

    def _get_uni_and_bi_representations(self, word_repr, mode):
        if mode == "unlabeled":
            self.uni_dropout_layer.p = self.dropout_unlab_prob
            self.bi_dropout_layer.p = self.dropout_unlab_prob
        elif mode == "labeled":
            self.uni_dropout_layer.p = self.dropout_lab_prob
            self.bi_dropout_layer.p = self.dropout_lab_prob
        uni_encoded, _ = self.uni_encoder(word_repr)
        uni_encoded = self.uni_dropout_layer(uni_encoded)
        bi_encoded, _ = self.bi_encoder(uni_encoded)
        bi_encoded = self.bi_dropout_layer(bi_encoded)
        return uni_encoded, bi_encoded

    def get_representations(self, word_tensor, mode, char_tensor=None, mask=None):
        word_repr = self._get_word_representations(word_tensor, mode, char_tensor=char_tensor, mask=mask)
        uni_repr, bi_repr = self._get_uni_and_bi_representations(word_repr, mode)
        return word_repr, uni_repr, bi_repr

    def get_input_sizes(self):
        return self.encoder_input_size, self.uni_encoder_hidden_size, self.bi_encoder_hidden_size

    def _compose_char_embeddings(self, char_tensor, mask=None):
        char_embs = []

        if self.char_compose_method == "cnn":
            char_emb = self.char_embedding(char_tensor)
            char_size = char_emb.size()
            # batch_size * seq_len, char_hidden, char_len
            char_emb = torch.squeeze(char_emb.view(char_size[0] * char_size[1], 1, char_size[3], char_size[2]))
            for conv in self.conv:
                # batch_size * seq_len, out_channels, conv_res_size
                char_emb_res = conv(char_emb)
                char_emb_res = F.relu(char_emb_res)
                char_emb_res = self.char_dropout_layer(torch.max(char_emb_res, dim=2)[0])
                char_emb_res = char_emb_res.unsqueeze(2).view(char_size[0], char_size[1], -1)
                char_embs.append(char_emb_res)

        elif self.char_compose_method == "rnn":
            # (batch_size * seq_length, char_seq_length, char_emb_size)
            embedded_chars = self.char_embedding(char_tensor.view(-1, char_tensor.size(-1)))
            # (batch_size * seq_length,)
            chars_lengths = mask.long().sum(dim=-1).view(-1)
            # PyTorch cuDNN restricts that the lengths must be sorted descending
            sorted_lengths, sort_indices = chars_lengths.sort(dim=0, descending=True)
            sorted_embedded_chars = embedded_chars[sort_indices]
            packed_encoded, _ = self.char_encoder(
                pack_padded_sequence(
                    sorted_embedded_chars, sorted_lengths.data.tolist(), batch_first=True))
            # (batch_size * seq_length, char_seq_length, char_hidden_size * 2)
            encoded_chars, _ = pad_packed_sequence(packed_encoded, batch_first=True)
            # Restore original ordering
            _, restore_indices = sort_indices.sort(dim=0)
            encoded_chars = encoded_chars[restore_indices]
            # (batch_size * seq_length, char_seq_length, char_hidden_size)
            encoded_chars_fwd, encoded_chars_bwd = encoded_chars.chunk(2, dim=-1)

            # We only want the output at the last timestep, but last timestep for
            # forward LSTM is determined by the mask
            # (batch_size * seq_length, 1, char_hidden_size)
            last_timestep_indices = (
                (chars_lengths - 1)
                    .view(-1, 1, 1)
                    .expand(encoded_chars_fwd.size(0), 1, encoded_chars_fwd.size(-1)))
            # (batch_size * seq_length, char_hidden_size)
            last_fwd = encoded_chars_fwd.gather(1, last_timestep_indices).squeeze(1)

            # Last timestep of backward LSTM is easier: it's at index 0
            # (batch_size * seq_length, char_hidden_size)
            last_bwd = encoded_chars_bwd[:, 0, :]

            # (batch_size * seq_length, char_hidden_size * 2)
            encoded_chars = torch.cat([last_fwd, last_bwd], dim=-1)
            # (batch_size, seq_length, word_emb_size)
            res = self.char_projection(encoded_chars).view(char_tensor.size(0), char_tensor.size(1), -1)
            char_embs.append(res)
        return char_embs

    def compute_similarity_loss(self, words, embedded_words, encoded_chars, unk_id):
        # words: (batch_size, seq_length)
        # embedded_words: (batch_size, seq_length, word_emb_size)
        # encoded_chars: (batch_size, seq_length, word_emb_size)

        assert words.dim() == 2
        assert embedded_words.dim() == 3
        assert embedded_words.size() == encoded_chars.size()

        # Disconnect the gradients; this should be equivalent to theano's `disconnected_grad`
        detached_embedded_words = embedded_words.detach()
        # (batch_size, seq_length)
        cosine_similarity = F.cosine_similarity(detached_embedded_words, encoded_chars, dim=-1)
        # Return cosine similarity loss for non-unk words
        return torch.sum((1. - cosine_similarity)[words != unk_id])

    def compute_lm_loss(self, encoded, words):
        # encoded: (batch_size, seq_length, bi_hidden_size) bi_encoded
        # words: (batch_size, seq_length) word_tensor

        assert encoded.dim() == 3
        assert words.dim() == 2
        assert encoded.size()[:2] == words.size()

        # (batch_size, seq_length, hidden_size)
        encoded_fwd, encoded_bwd = encoded.chunk(2, dim=-1)

        # Prepare for language modeling: (batch_size, seq_length - 1, lm_output_size)
        # NOTE sliced variables are not contiguous, and we need them so to be .view()'ed
        # (batch_size, seq_length - 1, hidden_size)
        encoded_fwd = encoded_fwd[:, :-1, :].contiguous()
        encoded_bwd = encoded_bwd[:, 1:, :].contiguous()

        # (batch_size * (seq_length - 1), lm_output_size)
        lm_fwd = self.lm_ff_fwd(encoded_fwd.view(-1, encoded_fwd.size(-1)))
        lm_bwd = self.lm_ff_bwd(encoded_bwd.view(-1, encoded_bwd.size(-1)))

        # NOTE sliced variables share the underlying tensors, thus preventing from
        # in-place operations, hence we clone them here; no need to call .contiguous()
        # because cloning makes them contiguous
        # (batch_size, seq_length - 1)
        targets_fwd, targets_bwd = words[:, 1:].clone(), words[:, :-1].clone()

        # Convert all word indices greater than max vocab size to max index
        targets_fwd[targets_fwd >= self.lm_max_vocab_size] = lm_fwd.size(-1) - 1
        targets_bwd[targets_bwd >= self.lm_max_vocab_size] = lm_bwd.size(-1) - 1

        # Compute language modeling loss; the loss is summed over batch
        return (F.cross_entropy(lm_fwd, targets_fwd.view(-1), size_average=False)
                + F.cross_entropy(lm_bwd, targets_bwd.view(-1), size_average=False))
