import torch
from PIL import Image

__all__ = ["encode_clip"]


clip_model = None
clip_processor = None


def encode_clip(batch: dict, rank: int | None = None, **kwargs) -> dict:
    """Encode image and text pairs with the CLIP model.

    Args:
    ----
        batch (dict): The batch of data.
        rank (int, optional): The rank of the process. Defaults to None.
        **kwargs: The keyword arguments.

    """
    input_image_column = kwargs.pop("input_image_column", "image")
    input_text_column = kwargs.pop("input_text_column", "text")

    global clip_model
    global clip_processor
    if clip_model is None:
        from transformers import CLIPModel, CLIPProcessor

        model_name = "openai/clip-vit-large-patch14"
        clip_model = CLIPModel.from_pretrained(model_name)
        clip_processor = CLIPProcessor.from_pretrained(model_name)

    if rank is not None or torch.cuda.is_available():
        device = f"cuda:{(rank or 0)% torch.cuda.device_count()}"
        dtype = torch.float16 if device != "cpu" else torch.float32
        clip_model.to(device=device, dtype=dtype)
    else:
        device = "cpu"

    if input_image_column not in batch:
        raise ValueError(f"{input_image_column} missing in dataset")
    if input_text_column not in batch:
        raise ValueError(f"{input_text_column} missing in dataset")

    if isinstance(batch[input_image_column], list):
        images = batch[input_image_column]
        texts = batch[input_text_column]

        images = [Image.open(image) for image in images]

        with torch.no_grad():
            inputs = clip_processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {key: value.to(clip_model.device) for key, value in inputs.items()}
            outputs = clip_model(**inputs)

        image_logits_column_name = f"{input_image_column}_{input_text_column}_logits"
        batch[image_logits_column_name] = outputs.logits_per_image.diag().cpu().numpy().tolist()
    else:
        raise NotImplementedError

    return batch
