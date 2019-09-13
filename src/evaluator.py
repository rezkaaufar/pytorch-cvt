#### code is fetched from https://github.com/sighsmile/conlleval/blob/master/conlleval.py and modified ####

import logging
import typing
from collections import defaultdict
import torch
from src.data_loader import Data

class Evaluator(object):

    def __init__(self,
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

    def _split_tag(self, chunk_tag):
        """
        split chunk tag into IOBES prefix and chunk_type
        e.g.
        B-PER -> (B, PER)
        O -> (O, None)
        """
        if chunk_tag == 'O':
            return ('O', None)
        # elif chunk_tag == '<pad>':
        #     return ('O', None)
        return chunk_tag.split('-', maxsplit=1)

    def _is_chunk_end(self, prev_tag, tag):
        """
        check if the previous chunk ended between the previous and current word
        e.g.
        (B-PER, I-PER) -> False
        (B-LOC, O)  -> True
        Note: in case of contradicting tags, e.g. (B-PER, I-LOC)
        this is considered as (B-PER, B-LOC)
        """
        prefix1, chunk_type1 = self._split_tag(prev_tag)
        prefix2, chunk_type2 = self._split_tag(tag)

        if prefix1 == 'O':
            return False
        if prefix2 == 'O':
            return prefix1 != 'O'

        if chunk_type1 != chunk_type2:
            return True

        return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']

    def _is_chunk_start(self, prev_tag, tag):
        """
        check if a new chunk started between the previous and current word
        """
        prefix1, chunk_type1 = self._split_tag(prev_tag)
        prefix2, chunk_type2 = self._split_tag(tag)

        if prefix2 == 'O':
            return False
        if prefix1 == 'O':
            return prefix2 != 'O'

        if chunk_type1 != chunk_type2:
            return True

        return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']

    def _calc_metrics(self, tp, p, t, percent=True):
        """
        compute overall precision, recall and FB1 (default values are 0.0)
        if percent is True, return 100 * original decimal value
        """
        precision = tp / p if p else 0
        recall = tp / t if t else 0
        fb1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        if percent:
            return 100 * precision, 100 * recall, 100 * fb1
        else:
            return precision, recall, fb1

    def _count_chunks(self, true_seqs, pred_seqs):
        """
        true_seqs: a list of true tags
        pred_seqs: a list of predicted tags
        return:
        correct_chunks: a dict (counter),
                        key = chunk types,
                        value = number of correctly identified chunks per type
        true_chunks:    a dict, number of true chunks per type
        pred_chunks:    a dict, number of identified chunks per type
        correct_counts, true_counts, pred_counts: similar to above, but for tags
        """
        correct_chunks = defaultdict(int)
        true_chunks = defaultdict(int)
        pred_chunks = defaultdict(int)

        correct_counts = defaultdict(int)
        true_counts = defaultdict(int)
        pred_counts = defaultdict(int)

        prev_true_tag, prev_pred_tag = 'O', 'O'
        correct_chunk = None

        for true_tag, pred_tag in zip(true_seqs, pred_seqs):
            if true_tag == pred_tag:
                correct_counts[true_tag] += 1
            true_counts[true_tag] += 1
            pred_counts[pred_tag] += 1

            _, true_type = self._split_tag(true_tag)
            _, pred_type = self._split_tag(pred_tag)

            if correct_chunk is not None:
                true_end = self._is_chunk_end(prev_true_tag, true_tag)
                pred_end = self._is_chunk_end(prev_pred_tag, pred_tag)

                if pred_end and true_end:
                    correct_chunks[correct_chunk] += 1
                    correct_chunk = None
                elif pred_end != true_end or true_type != pred_type:
                    correct_chunk = None

            true_start = self._is_chunk_start(prev_true_tag, true_tag)
            pred_start = self._is_chunk_start(prev_pred_tag, pred_tag)

            if true_start and pred_start and true_type == pred_type:
                correct_chunk = true_type
            if true_start:
                true_chunks[true_type] += 1
            if pred_start:
                pred_chunks[pred_type] += 1

            prev_true_tag, prev_pred_tag = true_tag, pred_tag
        if correct_chunk is not None:
            correct_chunks[correct_chunk] += 1

        return (correct_chunks, true_chunks, pred_chunks,
                correct_counts, true_counts, pred_counts)

    def _get_result(self, correct_chunks, true_chunks, pred_chunks,
                   correct_counts, true_counts, pred_counts, verbose=True):
        """
        if verbose, print overall performance, as well as preformance per chunk type;
        otherwise, simply return overall prec, rec, f1 scores
        """
        # sum counts
        sum_correct_chunks = sum(correct_chunks.values())
        sum_true_chunks = sum(true_chunks.values())
        sum_pred_chunks = sum(pred_chunks.values())

        sum_correct_counts = sum(correct_counts.values())
        sum_true_counts = sum(true_counts.values())

        nonO_correct_counts = sum(v for k, v in correct_counts.items() if k != 'O')
        nonO_true_counts = sum(v for k, v in true_counts.items() if k != 'O')

        chunk_types = sorted(list(set(list(true_chunks) + list(pred_chunks))))

        # compute overall precision, recall and FB1 (default values are 0.0)
        prec, rec, f1 = self._calc_metrics(sum_correct_chunks, sum_pred_chunks, sum_true_chunks)
        res = (prec, rec, f1)
        if not verbose:
            return res

        # print overall performance, and performance per chunk type

        # self.logger.info("processed %i tokens with %i phrases; " % (sum_true_counts, sum_true_chunks))
        # self.logger.info("found: %i phrases; correct: %i.\n" % (sum_pred_chunks, sum_correct_chunks))

        self.logger.info("accuracy: %6.2f%%; (non-O)" % (100 * nonO_correct_counts / nonO_true_counts))
        self.logger.info("accuracy: %6.2f%%; " % (100 * sum_correct_counts / sum_true_counts))
        # self.logger.info("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" % (prec, rec, f1))

        # # for each chunk type, compute precision, recall and FB1 (default values are 0.0)
        # for t in chunk_types:
        #     prec, rec, f1 = self._calc_metrics(correct_chunks[t], pred_chunks[t], true_chunks[t])
        #     self.logger.info("%17s: " % t)
        #     self.logger.info("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
        #           (prec, rec, f1))
        #     self.logger.info("  %d" % pred_chunks[t])

        return res
        # you can generate LaTeX output for tables like in
        # http://cnts.uia.ac.be/conll2003/ner/example.tex
        # but I'm not implementing this

    def _evaluate(self, true_seqs, pred_seqs, verbose=True):
        (correct_chunks, true_chunks, pred_chunks,
         correct_counts, true_counts, pred_counts) = self._count_chunks(true_seqs, pred_seqs)
        result = self._get_result(correct_chunks, true_chunks, pred_chunks,
                            correct_counts, true_counts, pred_counts, verbose=verbose)
        return result

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
            outputs = model.primary.predict(torch.cat([un, bi], dim=2)) # python array type
            label = label.long().cpu().numpy().tolist() # python array type
            hypotheses += outputs
            references += label

        ## convert number predictions to token form ##
        hypotheses_in_token = []
        references_in_token = []
        for hyp in hypotheses:
            for num in hyp:
                hypotheses_in_token.append(data.lab.vocab.itos[num])
            hypotheses_in_token.append("O") # this is done to separate sentences
        for ref in references:
            for num in ref:
                references_in_token.append(data.lab.vocab.itos[num])
            references_in_token.append("O")  # this is done to separate sentences

        result = self._evaluate(references_in_token, hypotheses_in_token)

        return result