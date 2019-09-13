import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.functional as Ff
from torchcrf import CRF

class PredictionModule(nn.Module):
    def __init__(self,
                 name: str,
                 input_size: int,
                 num_tags: int,
                 input_size_2 : int = None,
                 pre_output_layer_size: int = 200,
                 use_crf: bool = True,
                 roll_direction: int = 0,
                 activate: bool = True,
                 ) -> None:

        super().__init__()

        self.name = name
        self.input_size = input_size
        self.pre_output_layer_size = pre_output_layer_size
        self.roll_direction = roll_direction
        self.pre_output_layer = nn.Linear(input_size, pre_output_layer_size)
        self.pre_output_layer_2 = None
        if input_size_2 is not None:
            self.pre_output_layer_2 = nn.Linear(input_size_2, pre_output_layer_size)
        self.activate = activate
        self.output_layer = nn.Linear(pre_output_layer_size, num_tags)
        self.crf = CRF(num_tags)
        self.loss = nn.CrossEntropyLoss()
        # if name != "primary":
        #     self.loss = nn.KLDivLoss()
        self.use_crf = use_crf

    def project(self, repr, repr_2=None):
        projected = self.pre_output_layer(repr)
        if repr_2 is not None:
            projected += self.pre_output_layer_2(repr_2)
        if self.activate:
            F.relu(projected)
        outputs = self.output_layer(projected)
        # removing init and eos token
        outputs = outputs[:, 1:-1, :]

        # outputs = outputs.transpose(0, 1).contiguous()
        # best_tags = self.crf.decode(outputs)
        #
        # return best_tags

        return outputs

    def predict(self, repr, repr_2=None):
        projected = self.pre_output_layer(repr)
        if repr_2 is not None:
            projected += self.pre_output_layer_2(repr_2)
        if self.activate:
            F.relu(projected)
        outputs = self.output_layer(projected)
        # removing init and eos token
        outputs = outputs[:, 1:-1, :]

        outputs = outputs.transpose(0, 1).contiguous()
        best_tags = self.crf.decode(outputs)

        return best_tags

    def calculate_loss(self, repr, tags, repr_2=None):
        loss = 0.
        projected = self.pre_output_layer(repr)
        if repr_2 is not None:
            projected += self.pre_output_layer_2(repr_2)
        if self.activate:
            F.relu(projected)
        outputs = self.output_layer(projected)

        # removing init and eos token
        outputs = outputs[:, 1:-1, :]

        # Transpose batch_size and seq_length for CRF
        # NOTE transposing tensors make them not contiguous, but CRF needs them so
        # (seq_length - 2, batch_size, num_tags)
        sparse = True
        if self.use_crf:
            tags = self._roll(tags, sparse=sparse)
            outputs = outputs.transpose(0, 1).contiguous()
            tags = tags.transpose(0, 1).contiguous()
            loss += -self.crf(outputs, tags)
        else:
            if self.name != "primary":
                sparse = False
            tags = self._roll(tags, sparse=sparse)

            if self.name == "primary":
                loss += self.loss(outputs.transpose(1,2), tags)
            else:
                loss += self.loss(outputs.transpose(1,2),
                                  Ff.argmax(tags, dim=-1))
                # loss += self.loss(F.log_softmax(tags.contiguous(), dim=-1),
                #                   F.softmax(outputs.contiguous(), dim=-1))

        return loss

    def _roll(self, tensor, sparse=True):
        # assuming size (batch_size, seq_len - 2). this method is to shift the label
        if sparse:
            return torch.cat([tensor[:, self.roll_direction:], tensor[:, :self.roll_direction]], dim=1)
        return torch.cat([tensor[:, self.roll_direction:, :], tensor[:, :self.roll_direction, :]], dim=1)