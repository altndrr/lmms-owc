import math
import random
from collections.abc import Callable

from src import utils
from src.schema import AggregationInfo, MetricInfo

__all__ = [
    "AGGREGATIONS",
    "DEFAULT_METRICS_PER_OUTPUT_TYPE",
    "METRICS",
    "get_aggregation",
    "get_aggregation_builder",
    "get_aggregation_info",
    "get_aggregations_info",
    "get_metric",
    "get_metric_builder",
    "get_metric_info",
    "get_metrics_info",
    "get_metric_stderr_builder",
    "register_aggregation",
    "register_metric",
]

log = utils.get_logger(__name__, rank_zero_only=True)


AGGREGATIONS: dict[str, AggregationInfo] = {}
METRICS: dict[str, MetricInfo] = {}

DEFAULT_METRICS_PER_OUTPUT_TYPE = {
    "loglikelihood": ["perplexity", "acc"],
    "multiple_choice": ["acc", "acc_norm"],
    "generate_until": ["exact_match"],
    "generate_until_multi_round": ["exact_match"],
}


def get_aggregation(aggregation_id: str, **aggregation_kwargs) -> float:
    """Get an aggregation.

    Args:
    ----
        aggregation_id (str): The name of the aggregation function.
        aggregation_kwargs (dict): Keyword arguments to pass to the aggregation builder.

    """
    return AGGREGATIONS[aggregation_id].builder_fn(**aggregation_kwargs)


def get_aggregation_builder(aggregation_id: str) -> Callable:
    """Get an aggregation builder.

    Args:
    ----
        aggregation_id (str): The name of the aggregation.

    """
    return AGGREGATIONS[aggregation_id].builder_fn


def get_aggregation_info(aggregation_id: str) -> AggregationInfo:
    """Get the aggregation info.

    Args:
    ----
        aggregation_id (str): The name of the aggregation.

    """
    return AGGREGATIONS[aggregation_id]


def get_aggregations_info() -> list[AggregationInfo]:
    """Get all aggregation info."""
    return list(AGGREGATIONS.values())


def get_metric(metric_id: str, **metric_kwargs) -> float:
    """Get a metric.

    Args:
    ----
        metric_id (str): The name of the metric.
        metric_kwargs (dict): Keyword arguments to pass to the metric builder.

    """
    return METRICS[metric_id].builder_fn(**metric_kwargs)


def get_metric_builder(metric_id: str) -> Callable:
    """Get a metric builder.

    Args:
    ----
        metric_id (str): The name of the metric.

    """
    return METRICS[metric_id].builder_fn


def get_metric_info(metric_id: str) -> MetricInfo:
    """Get the metric info.

    Args:
    ----
        metric_id (str): The name of the metric.

    """
    return METRICS[metric_id]


def get_metrics_info() -> list[MetricInfo]:
    """Get all metrics info."""
    return list(METRICS.values())


def _sample_stddev(arr: list) -> float:
    """Calculate the sample standard deviation.

    Args:
    ----
        arr (list): List of values.

    """
    mu = sum(arr) / len(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))


def _mean_stderr(arr: list) -> float:
    """Calculate the standard error of the mean.

    Args:
    ----
        arr (list): List of values.

    """
    return _sample_stddev(arr) / math.sqrt(len(arr))


def _acc_all_stderr(items: list) -> float:
    """Calculate the standard error of the accuracy on a list of documents.

    Args:
    ----
        items (list): List of documents.

    """
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items, strict=True))[0]
    docs = list(zip(*items, strict=True))[1]

    for doc, pred in zip(docs, preds, strict=True):
        question_id = doc["idx"]["question"]
        if question_id not in question_scoring_dict:
            question_scoring_dict[question_id] = []

        gold_label = doc["label"] == 1
        question_scoring_dict[question_id].append(gold_label == pred)

    acc = _mean_stderr([int(all(x)) for x in question_scoring_dict.values()])
    return acc


class BootstrapInternal:
    """Internal class for bootstrapping.

    Args:
    ----
        f (Callable): The function.
        n (int): The number of samples.

    """

    def __init__(self, f: Callable, n: int) -> None:
        self.f = f
        self.n = n

    def __call__(self, v: tuple) -> list:
        """Call the function.

        Args:
        ----
            v (tuple): The tuple.

        """
        i, xs = v
        rnd = random.Random()  # noqa: S311
        rnd.seed(i)
        res = []
        for _ in range(self.n):
            res.append(self.f(rnd.choices(xs, k=len(xs))))

        return res


def _bootstrap_stderr(f: Callable, xs: list, iters: int) -> float:
    """Compute the bootstrapped standard error.

    Args:
    ----
        f (Callable): The function.
        xs (list): The list of values.
        iters (int): The number of iterations.

    """
    import multiprocessing as mp

    pool = mp.Pool(mp.cpu_count())
    # This gives a biased estimate of the stderr (i.e, w/ the mean, it gives something
    # equivalent to stderr calculated without Bessel's correction in the stddev.
    # Unfortunately, I haven't been able to figure out what the right correction is
    # to make the bootstrap unbiased - I considered multiplying by sqrt(n/(n-1)) but
    # that would be ad-hoc and I can't prove that that would actually be an unbiased estimator).
    # Thankfully, shouldn't matter because our samples are pretty big usually anyways.
    res = []
    chunk_size = min(1000, iters)
    from tqdm import tqdm

    log.info("bootstrapping for stddev:", f.__name__)
    for bootstrap in tqdm(
        pool.imap(
            BootstrapInternal(f, chunk_size),
            [(i, xs) for i in range(iters // chunk_size)],
        ),
        total=iters // chunk_size,
    ):
        # sample w replacement
        res.extend(bootstrap)

    pool.close()
    return _sample_stddev(res)


def get_metric_stderr_builder(metric: Callable, bootstrap_iters: int) -> Callable | None:
    """Get the metric standard error calculation builder function.

    Args:
    ----
        metric (Callable): The metric.
        bootstrap_iters (int): The number of bootstrap iterations

    """
    from src.data.metrics._group import mean
    from src.data.metrics._instance import acc_all

    if bootstrap_iters <= 0:
        return None

    can_bootstrap = [name for name, agg in AGGREGATIONS.items() if agg.can_bootstrap]
    can_bootstrap += [name for name, metric in METRICS.items() if metric.can_bootstrap]

    if metric in can_bootstrap:
        return lambda x: _bootstrap_stderr(metric, x, iters=bootstrap_iters)

    stderr = {mean: _mean_stderr, acc_all: _acc_all_stderr}
    return stderr.get(metric)


def register_aggregation(name: str | None = None, can_bootstrap: bool = False) -> Callable:
    """Register an aggregation function.

    Args:
    ----
        name (str, optional): Name of the aggregation function.
        can_bootstrap (bool): Whether the aggregation can be bootstrapped to evaluate its standard
            error. Default to False.

    """

    def decorator(aggregation: Callable) -> Callable:
        AGGREGATIONS[name or aggregation.__name__.lower()] = AggregationInfo(
            name=name or aggregation.__name__.lower(),
            builder_fn=aggregation,
            can_bootstrap=can_bootstrap,
        )
        return aggregation

    return decorator


def register_metric(
    name: str | None = None,
    group_fn_name: str | None = None,
    higher_is_better: bool | None = None,
    output_types: list | None = None,
    can_bootstrap: bool = False,
) -> Callable:
    """Register a metric.

    Args:
    ----
        name (str, optional): Name of the metric. Defaults to None.
        group_fn_name (str, optional): Name of the aggregation function. Defaults to None.
        higher_is_better (bool, optional): Whether higher values are better. Defaults to None.
        output_types (list, optional): The output types of the metric. Defaults to None.
        can_bootstrap (bool): Whether the metric can be bootstrapped to evaluate its standard
            error. Default to False.

    """

    def decorator(metric: Callable) -> Callable:
        METRICS[name or metric.__name__.lower()] = MetricInfo(
            name=name or metric.__name__.lower(),
            higher_is_better=higher_is_better,
            builder_fn=metric,
            group_fn=AGGREGATIONS[group_fn_name].builder_fn,
            output_types=output_types or [],
            can_bootstrap=can_bootstrap,
        )

        return metric

    return decorator
