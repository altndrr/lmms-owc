__all__ = ["Sampler"]


class Sampler:
    """Class describing a sampler.

    Args:
    ----
        args (tuple | list, optional): Positional arguments.
        kwargs (dict, optional) Keyboard arguments.

    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def sample(self, n_samples: int) -> list:
        """Draw `n` samples from the fewshot docs.

        Args:
        ----
            n_samples (int): The number of samples to draw.

        """
        raise NotImplementedError
