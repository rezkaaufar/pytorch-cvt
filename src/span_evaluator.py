import logging
import typing
import torch
from src.data_loader import Data
import torch.nn.functional as F
import torch.functional as Ff

class SpanEvaluator(object):

    def __init__(self,
                 use_crf: bool = True,
                 logger: typing.Optional[logging.Logger] = None,
                 ) -> None:
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s - %(name)s - %(message)s'))
            logger.addHandler(handler)

        super().__init__()

        self.logger = logger
        self.use_crf = use_crf

    def _get_span_labels(self, sentence_tags, inv_label_mapping=None):
      """Go from token-level labels to list of entities (start, end, class)."""

      if inv_label_mapping:
        sentence_tags = [inv_label_mapping[i] for i in sentence_tags]
      span_labels = []
      last = 'O'
      start = -1
      for i, tag in enumerate(sentence_tags):
        pos, _ = (None, 'O') if tag == 'O' else tag.split('-')
        if (pos == 'S' or pos == 'B' or tag == 'O') and last != 'O':
          span_labels.append((start, i - 1, last.split('-')[-1]))
        if pos == 'B' or pos == 'S' or last == 'O':
          start = i
        last = tag
      if sentence_tags[-1] != 'O':
        span_labels.append((start, len(sentence_tags) - 1,
                            sentence_tags[-1].split('-')[-1]))
      return span_labels

    def _get_results(self, golds, preds):
        n_correct, n_predicted, n_gold = 0, 0, 0
        for gold, preds in zip(golds, preds):
            sent_spans = set(self._get_span_labels(
                gold))
            span_preds = set(self._get_span_labels(
                preds))
            n_correct += len(sent_spans & span_preds)
            n_gold += len(sent_spans)
            n_predicted += len(span_preds)

        if n_correct == 0:
            p, r, f1 = 0, 0, 0
        else:
            p = 100.0 * n_correct / n_predicted
            r = 100.0 * n_correct / n_gold
            f1 = 2 * p * r / (p + r)
        return p, r, f1

    def evaluate_on_data(self, model, data, which="dev"):
        """
        :param model: CVTModel
        :param data: torchtext BucketIterator (either train, dev, or test)
        :return:
        """
        references = []
        hypotheses = []

        dataset = data.dev
        if which == "train":
            dataset = data.train
        elif which == "test":
            dataset = data.test

        ## iterate and gather predictions ##
        pad_token_id = data.get_pad_token_id()
        for i, mb in enumerate(dataset):
            word_input = getattr(mb, Data.WORDS_NAME)
            char_input = getattr(mb, Data.CHAR_NAME)
            label = getattr(mb, Data.LAB_NAME)
            mask = char_input != pad_token_id
            wr, un, bi = model.encoder.get_representations(word_input, "labeled", char_tensor=char_input, mask=mask)
            if self.use_crf:
                outputs = model.primary.predict(un, repr_2=bi) # python array type
            else:
                outputs = model.primary.project(un, repr_2=bi) # tensor array type
                outputs = Ff.argmax(outputs, dim=-1)
                outputs = outputs.long().cpu().numpy().tolist() # python array type
            label = label.long().cpu().numpy().tolist() # python array type

            assert len(label) == len(outputs)
            assert len(label[0]) == len(outputs[0])

            hypotheses += outputs
            references += label


        ## convert number predictions to token form ##
        hypotheses_in_token = []
        references_in_token = []
        for hyp in hypotheses:
            hypothesis = []
            for num in hyp:
                hypothesis.append(data.lab.vocab.itos[num])
                hypotheses_in_token.append(hypothesis)
        for ref in references:
            reference = []
            for num in ref:
                reference.append(data.lab.vocab.itos[num])
                references_in_token.append(reference)

        result = self._get_results(references_in_token, hypotheses_in_token)

        return result