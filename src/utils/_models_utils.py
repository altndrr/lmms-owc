from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, ValuesView

__all__ = ["Collator"]


# from more_itertools
class Collator:
    """Reorder and batch elements of an array.

    It allows for sorting an array based on a provided sorting function, grouping elements based on
    a grouping function, and generating batches from the sorted and grouped data.

    Args:
    ----
        data_source (list): The array to be reordered and batched.
        sort_fn (Callable): The function to sort the array.
        group_fn (Callable): The function to group the array. Defaults to ``lambda x: x[1]``.
        grouping (bool): If True, groups the array based on the group function. Defaults to False.

    """

    def __init__(
        self,
        data_source: list,
        sort_fn: Callable,
        group_fn: Callable = lambda x: x[1],
        grouping: bool = False,
    ) -> None:
        self._data_source = data_source
        self._data_source_with_indices = tuple(enumerate(data_source))
        self._sort_fn = sort_fn
        self._group_fn = lambda x: group_fn(x[1])  # first index are enumerated indices
        self._grouping = grouping
        self._reorder_indices = []
        self.size = len(data_source)

        if self._grouping is True:
            self._data_source_with_indices = self.group(
                self._data_source_with_indices, self._group_fn
            )

    @staticmethod
    def split(_iter: Iterable, n: int = 0, fn: Callable | None = None) -> Iterator:
        """Split an iterable into chunks.

        Args:
        ----
            _iter (Iterable): The input iterable to be divided into chunks.
            n (int): The size of each chunk. Defaults to 0.
            fn (Callable): A function to determine the size of each chunk. Defaults to None.

        """
        arr = []
        _iter = tuple(_iter)
        for i, x in enumerate(_iter):
            arr.append(x)
            if len(arr) == (fn(i, _iter) if fn else n):
                yield arr
                arr = []

        if arr:
            yield arr

    @staticmethod
    def group(arr: Iterable, fn: Callable, values: bool = False) -> dict | ValuesView:
        """Group elements of an iterable based on a provided function.

        Args:
        ----
            arr (Iterable): The iterable to be grouped.
            fn (Callable): The function to determine the grouping.
            values (bool): If True, returns the values of the group. Defaults to False.

        """
        res = defaultdict(list)
        for ob in arr:
            try:
                hashable_dict = tuple(
                    (key, tuple(value) if isinstance(value, Iterable) else value)
                    for key, value in sorted(fn(ob).items())
                )
                res[hashable_dict].append(ob)
            except TypeError:
                res[fn(ob)].append(ob)

        if not values:
            return res

        return res.values()

    def get_batched(self, n: int = 1, batch_fn: Callable | None = None) -> Iterable:
        """Generate and yields batches from the reordered array.

        Args:
        ----
            n (int): The size of each batch. Defaults to 1.
            batch_fn (Callable): A function to determine the size of each batch. Defaults to None

        """
        if self._grouping:
            if not isinstance(self._data_source_with_indices, dict):
                raise TypeError("Data source must be a dictionary when grouping is enabled")

            for _, values in self._data_source_with_indices.items():
                values = self._reorder(values)
                batch = self.split(values, n=n, fn=batch_fn)
                yield from batch
        else:
            values = self._reorder(self._data_source_with_indices)
            batch = self.split(values, n=n, fn=batch_fn)
            yield from batch

    def get_original(self, new_arr: list) -> list:
        """Restore the original order of elements from the reordered list.

        Args:
        ----
            new_arr (list): The reordered array.

        """
        res = [None] * self.size
        cov = [False] * self.size

        for ind, v in zip(self._reorder_indices, new_arr, strict=True):
            res[ind] = v
            cov[ind] = True

        if not all(cov):
            raise ValueError("Not all elements were covered in the reordering.")

        return res

    def _reorder(self, arr: Iterable | Iterable[tuple]) -> Iterator:
        """Reorder the elements in the array based on the sorting function.

        Args:
        ----
            arr (Iterable | Iterable[tuple]): The array to be reordered.

        """
        arr = sorted(arr, key=lambda x: self._sort_fn(x[1]))
        self._reorder_indices.extend([x[0] for x in arr])
        yield from [x[1] for x in arr]

    def __len__(self) -> int:
        """Return the size of the array."""
        return self.size
