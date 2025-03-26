import argparse
import json
import os
import shutil
import subprocess
import time
from glob import glob

import pytest
import torch

from src.models import get_model, get_model_info

RESULTS = {
    "llava-1.5-7b": {
        "ai2d": 0.625,
        "mmmu_val": 0.250,
        "muirbench": 0.625,
        "ocr_bench": 0.001,
    },
    "llava-onevision-qwen2-0.5b-ov": {
        "ai2d": 0.625,
        "mmmu_val": (0.250, 0.375),  # with/without flash attn
        "muirbench": 0.125,
        "ocr_bench": 0.007,
    },
    "idefics2-8b": {
        "ai2d": 0,
        "mmmu_val": 0.250,
        # "muirbench": None,  OOM on A6000
        "ocr_bench": 0.007,
    },
    "instructblip-vicuna-7b": {
        "ai2d": 0.500,
        "mmmu_val": 0.250,
        "muirbench": 0.250,
        "ocr_bench": 0.008,
    },
    "internvl2-2b": {
        "ai2d": 1.0,
        "mmmu_val": 0.375,
        "muirbench": 0.250,
        "ocr_bench": 0.008,
    },
    "phi3v": {
        "ai2d": 0.875,
        "mmmu_val": 0.500,
        # "muirbench": ???,
        "ocr_bench": 0.007,
    },
    "qwen2-vl-2b": {
        "ai2d": 0.500,
        "mmmu_val": 0.500,
        "muirbench": 0.0,
        "ocr_bench": 0.007,
    },
}


@pytest.mark.slow
@pytest.mark.parametrize("model_id", list(RESULTS.keys()))
@pytest.mark.parametrize("load_in_8bit", [False, True])
def test_model_init(model_id: str, load_in_8bit: bool) -> None:
    """Test initializing models."""
    model_info = get_model_info(model_id)
    model = get_model(model_id, device_map="auto", load_in_8bit=load_in_8bit)

    assert model_info.name is not None
    assert model.batch_size == 1
    assert model.device.type == "cuda"

    if model_id not in ["internvl2-2b"]:
        assert model.device_map == "auto"


@pytest.mark.slow
@pytest.mark.parametrize("num_processes", list(range(1, torch.cuda.device_count() + 1)))
@pytest.mark.parametrize("attn_implementation", [None, "flash_attention_2"])
def test_llava_15_7b(
    default_args: argparse.Namespace, num_processes: int, attn_implementation: str | None
) -> None:
    """Test the LLaVA 1.5 model 7B params regression via the command line entrypoint."""
    args = default_args

    model_id = "llava-1.5-7b"
    expected_results = RESULTS[model_id]

    args.task = list(expected_results.keys())
    args.output_path = f"logs/tests/{int(time.time())}"
    os.makedirs(args.output_path, exist_ok=False)

    python_path = shutil.which("python3")
    assert python_path

    command_args = [python_path, "-m", "accelerate.commands.launch"]
    command_args.extend(["--main_process_port=12580", f"--num_processes={num_processes}"])
    command_args.extend(["-m", "eval_model"])
    command_args.extend(["--model", model_id])
    if attn_implementation:
        command_args.extend(["--model_args", f"attn_implementation={attn_implementation}"])
    command_args.extend(["--tasks", ",".join(args.task)])
    command_args.extend(["--num_fewshot", str(args.num_fewshot)])
    command_args.extend(["--limit", str(args.limit)])
    command_args.extend(["--batch_size", str(args.batch_size)])
    command_args.extend(["--output_path", args.output_path])

    command = " ".join(command_args)
    assert "--model llava-1.5-7b" in command
    assert "--num_fewshot 0" in command
    assert "--batch_size 1" in command

    output = subprocess.run(  # noqa: S603
        command_args,
        capture_output=True,
        shell=False,  # noqa: S603
        text=True,
        check=True,
        env=dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=",".join(map(str, range(num_processes))),
        ),
    )
    with open(f"{args.output_path}/command_output.log", "w") as f:
        f.write(output.stdout)
    with open(f"{args.output_path}/command_output.err", "w") as f:
        f.write(output.stderr)

    assert output.returncode == 0

    json_file_path = next(
        iter(glob(f"{args.output_path}/**/*_results.json", recursive=True)),
        None,
    )
    assert json_file_path is not None, f"Expected to find a JSON file in `{args.output_path}`"

    with open(json_file_path) as f:
        logs = json.load(f)

    assert logs != {}

    for key, value in expected_results.items():
        predicted_value = list(logs["results"][key].values())[1]
        assert predicted_value == value, f"Expected `value={value}` with `task={key}`"


@pytest.mark.slow
@pytest.mark.parametrize("num_processes", list(range(1, torch.cuda.device_count() + 1)))
@pytest.mark.parametrize("attn_implementation", [None, "flash_attention_2"])
def test_llava_onevision_qwen2_0_5b_ov(
    default_args: argparse.Namespace, num_processes: int, attn_implementation: str | None
) -> None:
    """Test the LLaVAOnevision model with Qwen2 0.5B regression via the command line entrypoint."""
    args = default_args

    model_id = "llava-onevision-qwen2-0.5b-ov"
    expected_results = RESULTS[model_id]

    args.task = list(expected_results.keys())
    args.output_path = f"logs/tests/{int(time.time())}"
    os.makedirs(args.output_path, exist_ok=False)

    python_path = shutil.which("python3")
    assert python_path

    command_args = [python_path, "-m", "accelerate.commands.launch"]
    command_args.extend(["--main_process_port=12580", f"--num_processes={num_processes}"])
    command_args.extend(["-m", "eval_model"])
    command_args.extend(["--model", model_id])
    if attn_implementation:
        command_args.extend(["--model_args", f"attn_implementation={attn_implementation}"])
    command_args.extend(["--tasks", ",".join(args.task)])
    command_args.extend(["--num_fewshot", str(args.num_fewshot)])
    command_args.extend(["--limit", str(args.limit)])
    command_args.extend(["--batch_size", str(args.batch_size)])
    command_args.extend(["--output_path", args.output_path])

    command = " ".join(command_args)
    assert "--model llava-onevision-qwen2-0.5b-ov" in command
    assert "--num_fewshot 0" in command
    assert "--batch_size 1" in command

    output = subprocess.run(  # noqa: S603
        command_args,
        capture_output=True,
        shell=False,  # noqa: S603
        text=True,
        check=True,
        env=dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=",".join(map(str, range(num_processes))),
        ),
    )
    with open(f"{args.output_path}/command_output.log", "w") as f:
        f.write(output.stdout)
    with open(f"{args.output_path}/command_output.err", "w") as f:
        f.write(output.stderr)

    assert output.returncode == 0

    json_file_path = next(
        iter(glob(f"{args.output_path}/**/*_results.json", recursive=True)),
        None,
    )
    assert json_file_path is not None, f"Expected to find a JSON file in `{args.output_path}`"

    with open(json_file_path) as f:
        logs = json.load(f)

    assert logs != {}

    for key, value in expected_results.items():
        if isinstance(value, tuple):
            value = value[0] if attn_implementation else value[1]
        predicted_value = list(logs["results"][key].values())[1]
        assert predicted_value == value, f"Expected `value={value}` with `task={key}`"


@pytest.mark.slow
@pytest.mark.parametrize("num_processes", list(range(1, torch.cuda.device_count() + 1)))
def test_idefics2_8b(default_args: argparse.Namespace, num_processes: int) -> None:
    """Test the Idefics2 model with 8B params regression via the command line entrypoint."""
    args = default_args

    model_id = "idefics2-8b"
    expected_results = RESULTS[model_id]

    args.task = list(expected_results.keys())
    args.output_path = f"logs/tests/{int(time.time())}"
    os.makedirs(args.output_path, exist_ok=False)

    python_path = shutil.which("python3")
    assert python_path

    command_args = [python_path, "-m", "accelerate.commands.launch"]
    command_args.extend(["--main_process_port=12580", f"--num_processes={num_processes}"])
    command_args.extend(["-m", "eval_model"])
    command_args.extend(["--model", model_id])
    command_args.extend(["--tasks", ",".join(args.task)])
    command_args.extend(["--num_fewshot", str(args.num_fewshot)])
    command_args.extend(["--limit", str(args.limit)])
    command_args.extend(["--batch_size", str(args.batch_size)])
    command_args.extend(["--output_path", args.output_path])

    command = " ".join(command_args)
    assert "--model idefics2-8b" in command
    assert "--num_fewshot 0" in command
    assert "--batch_size 1" in command

    output = subprocess.run(  # noqa: S603
        command_args,
        capture_output=True,
        shell=False,  # noqa: S603
        text=True,
        check=True,
        env=dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=",".join(map(str, range(num_processes))),
        ),
    )
    with open(f"{args.output_path}/command_output.log", "w") as f:
        f.write(output.stdout)
    with open(f"{args.output_path}/command_output.err", "w") as f:
        f.write(output.stderr)

    assert output.returncode == 0

    json_file_path = next(
        iter(glob(f"{args.output_path}/**/*_results.json", recursive=True)),
        None,
    )
    assert json_file_path is not None, f"Expected to find a JSON file in `{args.output_path}`"

    with open(json_file_path) as f:
        logs = json.load(f)

    assert logs != {}

    for key, value in expected_results.items():
        predicted_value = list(logs["results"][key].values())[1]
        assert predicted_value == value, f"Expected `value={value}` with `task={key}`"


@pytest.mark.slow
@pytest.mark.parametrize("num_processes", list(range(1, torch.cuda.device_count() + 1)))
def test_instructblip_vicuna_7b(default_args: argparse.Namespace, num_processes: int) -> None:
    """Test the InstructBLIP model with Vicuna 7B regression via the command line entrypoint."""
    args = default_args

    model_id = "instructblip-vicuna-7b"
    expected_results = RESULTS[model_id]

    args.task = list(expected_results.keys())
    args.output_path = f"logs/tests/{int(time.time())}"
    os.makedirs(args.output_path, exist_ok=False)

    python_path = shutil.which("python3")
    assert python_path

    command_args = [python_path, "-m", "accelerate.commands.launch"]
    command_args.extend(["--main_process_port=12580", f"--num_processes={num_processes}"])
    command_args.extend(["-m", "eval_model"])
    command_args.extend(["--model", model_id])
    command_args.extend(["--tasks", ",".join(args.task)])
    command_args.extend(["--num_fewshot", str(args.num_fewshot)])
    command_args.extend(["--limit", str(args.limit)])
    command_args.extend(["--batch_size", str(args.batch_size)])
    command_args.extend(["--output_path", args.output_path])

    command = " ".join(command_args)

    assert "--model instructblip-vicuna-7b" in command
    assert "--num_fewshot 0" in command
    assert "--batch_size 1" in command

    output = subprocess.run(  # noqa: S603
        command_args,
        capture_output=True,
        shell=False,  # noqa: S603
        text=True,
        check=True,
        env=dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=",".join(map(str, range(num_processes))),
        ),
    )
    with open(f"{args.output_path}/command_output.log", "w") as f:
        f.write(output.stdout)
    with open(f"{args.output_path}/command_output.err", "w") as f:
        f.write(output.stderr)

    assert output.returncode == 0

    json_file_path = next(
        iter(glob(f"{args.output_path}/**/*_results.json", recursive=True)),
        None,
    )
    assert json_file_path is not None, f"Expected to find a JSON file in `{args.output_path}`"

    with open(json_file_path) as f:
        logs = json.load(f)

    assert logs != {}

    for key, value in expected_results.items():
        predicted_value = list(logs["results"][key].values())[1]
        assert predicted_value == value, f"Expected `value={value}` with `task={key}`"


@pytest.mark.slow
@pytest.mark.parametrize("num_processes", list(range(1, torch.cuda.device_count() + 1)))
def test_internvl2_2b(default_args: argparse.Namespace, num_processes: int) -> None:
    """Test the InternVL2 model with 2B params regression via the command line entrypoint."""
    args = default_args

    model_id = "internvl2-2b"
    expected_results = RESULTS[model_id]

    args.task = list(expected_results.keys())
    args.output_path = f"logs/tests/{int(time.time())}"
    os.makedirs(args.output_path, exist_ok=False)

    python_path = shutil.which("python3")
    assert python_path

    command_args = [python_path, "-m", "accelerate.commands.launch"]
    command_args.extend(["--main_process_port=12580", f"--num_processes={num_processes}"])
    command_args.extend(["-m", "eval_model"])
    command_args.extend(["--model", model_id])
    command_args.extend(["--tasks", ",".join(args.task)])
    command_args.extend(["--num_fewshot", str(args.num_fewshot)])
    command_args.extend(["--limit", str(args.limit)])
    command_args.extend(["--batch_size", str(args.batch_size)])
    command_args.extend(["--output_path", args.output_path])

    command = " ".join(command_args)
    assert "--model internvl2-2b" in command
    assert "--num_fewshot 0" in command
    assert "--batch_size 1" in command

    output = subprocess.run(  # noqa: S603
        command_args,
        capture_output=True,
        shell=False,  # noqa: S603
        text=True,
        check=True,
        env=dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=",".join(map(str, range(num_processes))),
        ),
    )
    with open(f"{args.output_path}/command_output.log", "w") as f:
        f.write(output.stdout)
    with open(f"{args.output_path}/command_output.err", "w") as f:
        f.write(output.stderr)

    assert output.returncode == 0

    json_file_path = next(
        iter(glob(f"{args.output_path}/**/*_results.json", recursive=True)),
        None,
    )
    assert json_file_path is not None, f"Expected to find a JSON file in `{args.output_path}`"

    with open(json_file_path) as f:
        logs = json.load(f)

    assert logs != {}

    for key, value in expected_results.items():
        predicted_value = list(logs["results"][key].values())[1]
        assert predicted_value == value, f"Expected `value={value}` with `task={key}`"


@pytest.mark.slow
@pytest.mark.parametrize("num_processes", list(range(1, torch.cuda.device_count() + 1)))
@pytest.mark.parametrize("attn_implementation", ["eager", "flash_attention_2"])
def test_phi3v(
    default_args: argparse.Namespace, num_processes: int, attn_implementation: str
) -> None:
    """Test the Phi3V model with 4B params regression via the command line entrypoint."""
    args = default_args

    model_id = "phi3v"
    expected_results = RESULTS[model_id]

    args.task = list(expected_results.keys())
    args.output_path = f"logs/tests/{int(time.time())}"
    os.makedirs(args.output_path, exist_ok=False)

    python_path = shutil.which("python3")
    assert python_path

    command_args = [python_path, "-m", "accelerate.commands.launch"]
    command_args.extend(["--main_process_port=12580", f"--num_processes={num_processes}"])
    command_args.extend(["-m", "eval_model"])
    command_args.extend(["--model", model_id])
    command_args.extend(["--model_args", f"attn_implementation={attn_implementation}"])
    command_args.extend(["--tasks", ",".join(args.task)])
    command_args.extend(["--num_fewshot", str(args.num_fewshot)])
    command_args.extend(["--limit", str(args.limit)])
    command_args.extend(["--batch_size", str(args.batch_size)])
    command_args.extend(["--output_path", args.output_path])

    command = " ".join(command_args)
    assert "--model phi3v" in command
    assert "--num_fewshot 0" in command
    assert "--batch_size 1" in command

    output = subprocess.run(  # noqa: S603
        command_args,
        capture_output=True,
        shell=False,  # noqa: S603
        text=True,
        check=True,
        env=dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=",".join(map(str, range(num_processes))),
        ),
    )
    with open(f"{args.output_path}/command_output.log", "w") as f:
        f.write(output.stdout)
    with open(f"{args.output_path}/command_output.err", "w") as f:
        f.write(output.stderr)

    assert output.returncode == 0

    json_file_path = next(
        iter(glob(f"{args.output_path}/**/*_results.json", recursive=True)),
        None,
    )
    assert json_file_path is not None, f"Expected to find a JSON file in `{args.output_path}`"

    with open(json_file_path) as f:
        logs = json.load(f)

    assert logs != {}

    for key, value in expected_results.items():
        predicted_value = list(logs["results"][key].values())[1]
        assert predicted_value == value, f"Expected `value={value}` with `task={key}`"


@pytest.mark.slow
@pytest.mark.parametrize("num_processes", list(range(1, torch.cuda.device_count() + 1)))
@pytest.mark.parametrize("use_flash_attention_2", [True, False])
def test_qwen2_vl_2b(
    default_args: argparse.Namespace, num_processes: int, use_flash_attention_2: bool
) -> None:
    """Test the Qwen2-VL model with 2B params regression via the command line entrypoint."""
    args = default_args

    model_id = "qwen2-vl-2b"
    expected_results = RESULTS[model_id]

    args.task = list(expected_results.keys())
    args.output_path = f"logs/tests/{int(time.time())}"
    os.makedirs(args.output_path, exist_ok=False)

    python_path = shutil.which("python3")
    assert python_path

    command_args = [python_path, "-m", "accelerate.commands.launch"]
    command_args.extend(["--main_process_port=12580", f"--num_processes={num_processes}"])
    command_args.extend(["-m", "eval_model"])
    command_args.extend(["--model", model_id])
    command_args.extend(["--model_args", f"use_flash_attention_2={use_flash_attention_2}"])
    command_args.extend(["--tasks", ",".join(args.task)])
    command_args.extend(["--num_fewshot", str(args.num_fewshot)])
    command_args.extend(["--limit", str(args.limit)])
    command_args.extend(["--batch_size", str(args.batch_size)])
    command_args.extend(["--output_path", args.output_path])

    command = " ".join(command_args)
    assert "--model qwen2-vl-2b" in command
    assert "--num_fewshot 0" in command
    assert "--batch_size 1" in command

    output = subprocess.run(  # noqa: S603
        command_args,
        capture_output=True,
        shell=False,  # noqa: S603
        text=True,
        check=True,
        env=dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=",".join(map(str, range(num_processes))),
        ),
    )
    with open(f"{args.output_path}/command_output.log", "w") as f:
        f.write(output.stdout)
    with open(f"{args.output_path}/command_output.err", "w") as f:
        f.write(output.stderr)

    assert output.returncode == 0

    json_file_path = next(
        iter(glob(f"{args.output_path}/**/*_results.json", recursive=True)),
        None,
    )
    assert json_file_path is not None, f"Expected to find a JSON file in `{args.output_path}`"

    with open(json_file_path) as f:
        logs = json.load(f)

    assert logs != {}

    for key, value in expected_results.items():
        predicted_value = list(logs["results"][key].values())[1]
        assert predicted_value == value, f"Expected `value={value}` with `task={key}`"
