from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from torchtext.data import Dataset, Example, Iterator, batch


class GroupedBucketIterator(Iterator):
    """A bucket iterator that can group examples by a custom key.

    When creating batches, this iterator groups examples by the specified ``group_by``
    function and batch examples belonging to the same group. In other words, two
    examples are in one batch if and only if they belong to the same group. Shuffling
    is done a wee bit differently by this iterator: only the batches are shuffled, examples
    in one batch are not. This behavior is consistent with that of Rei et al. (2016).

    Arguments
    ---------
    dataset : :class:`~torchtext.data.Dataset`
        Dataset on which the iterator operates.
    batch_size : int
        Maximum number of examples in one batch.
    group_by : callable
        A callable whose return value is treated as the grouping key. This callable should
        accept an example and return a hashable object.
    device : int, optional
        GPU device in which to store the variable tensors returned by this iterator. If
        ``None`` then the current GPU device is used. Set to -1 to store in CPU instead.
    train : bool, optional
        Whether the iterator is for training. If ``False`` then the variable tensors
        returned will have ``volatile=True``. Defaults to ``True``.
    shuffle_batches : bool, optional
        Whether to shuffle the batches. If ``None`` then this will be set to whatever
        ``train`` is.
    """
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int,
                 group_by: Callable,
                 device: Optional[int] = None,
                 train: bool = True,
                 shuffle_batches: Optional[bool] = None,
                 ) -> None:
        super(GroupedBucketIterator, self).__init__(
            dataset,
            batch_size,
            device=device,
            train=train,
            repeat=False,
            shuffle=False,
            sort=False,
            sort_within_batch=False,
        )
        self.group_by = group_by
        self.shuffle_batches = self.train if shuffle_batches is None else shuffle_batches
        self._batches = self._do_batch()

    def _do_batch(self) -> List[List[Example]]:
        groups: Dict[Any, List[Example]] = defaultdict(list)
        for example in self.data():
            groups[self.group_by(example)].append(example)

        minibatches = []
        for examples in groups.values():
            for minibatch in batch(examples, self.batch_size):
                minibatches.append(minibatch)
        return minibatches

    def __len__(self) -> int:
        return len(self._batches)

    def create_batches(self) -> None:
        if self.shuffle_batches:
            self.batches = self.random_shuffler(self._batches)
        else:
            self.batches = self._batches
        # Turn into generator for compatibility
        self.batches = (minibatch for minibatch in self.batches)
