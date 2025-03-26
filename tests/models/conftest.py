import argparse
import gc
import random

import numpy as np
import pytest
import torch


@pytest.fixture
def default_args() -> argparse.Namespace:
    """Create and returns a default Namespace object with predefined arguments."""
    return argparse.Namespace(
        acc_norm=False,
        perplexity=None,
        num_fewshot=0,
        limit=8,
        batch_size=1,
    )


@pytest.fixture(scope="session", autouse=True)
def deterministic() -> None:
    """Set the random seed for deterministic results.

    Args:
    ----
        monkeypatch (fixture): Pytest fixture for monkey-patching.

    """
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@pytest.fixture(autouse=True)
def gc_cuda(request: pytest.FixtureRequest) -> None:
    """Perform garbage collection on the GPU after each test.

    Args:
    ----
        request (pytest.FixtureRequest): A pytest fixture request object that handles test case
            setup and cleanup.

    """

    def gc_cuda_finalizer() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    request.addfinalizer(gc_cuda_finalizer)
