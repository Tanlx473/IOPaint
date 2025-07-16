"""IOPaint 核心包初始化 (Core package initialization).

本模块负责配置 PyTorch 相关的环境变量，并暴露 :func:`entry_point` 函数，
以便通过 ``python -m iopaint`` 或 ``iopaint`` 命令行入口启动程序。
This module configures PyTorch environment variables and exposes
``entry_point`` so the program can be launched via ``python -m iopaint``
or the ``iopaint`` command.

"""

import os
import importlib.util
import shutil
import ctypes
import logging

# 启用 Apple MPS 回退

# Enable fallback for Apple MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# 控制 oneDNN 缓存容量，解决警告
# Control oneDNN cache capacity to suppress warnings
os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = "1"
os.environ["LRU_CACHE_CAPACITY"] = "1"
# 避免在 GPU 上运行模型时出现内存泄漏问题
# Avoid GPU memory leak issues when running models

# 参考：https://github.com/pytorch/pytorch/issues/98688#issuecomment-1869288431
#       https://github.com/pytorch/pytorch/issues/108334#issuecomment-1752763633
os.environ["TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT"] = "1"

import warnings

warnings.simplefilter("ignore", UserWarning)


def fix_window_pytorch():

    """在 Windows 上修补 PyTorch 的兼容性问题。

    Patch PyTorch on Windows to avoid compatibility issues.
    """

    # 来源：https://github.com/comfyanonymous/ComfyUI/blob/5cbaa9e07c97296b536f240688f5a19300ecf30d/fix_torch.py#L4
    # Source: https://github.com/comfyanonymous/ComfyUI/blob/5cbaa9e07c97296b536f240688f5a19300ecf30d/fix_torch.py#L4

    import platform

    try:
        if platform.system() != "Windows":
            return
        torch_spec = importlib.util.find_spec("torch")
        for folder in torch_spec.submodule_search_locations:
            lib_folder = os.path.join(folder, "lib")
            test_file = os.path.join(lib_folder, "fbgemm.dll")
            dest = os.path.join(lib_folder, "libomp140.x86_64.dll")
            if os.path.exists(dest):
                break

            with open(test_file, "rb") as f:
                contents = f.read()
                if b"libomp140.x86_64.dll" not in contents:
                    break
            try:
                mydll = ctypes.cdll.LoadLibrary(test_file)
            except FileNotFoundError:
                logging.warning("Detected pytorch version with libomp issue, patching.")
                shutil.copyfile(os.path.join(lib_folder, "libiomp5md.dll"), dest)
    except:
        pass


def entry_point():
    """程序入口，用于启动命令行界面。

    Entry point used by the CLI.
    """

    # 使 ``os.environ["XDG_CACHE_HOME"] = args.model_cache_dir`` 在不同用户下生效
    # Allow per-user model cache directories to work

    # 参考：https://github.com/huggingface/diffusers/blob/be99201a567c1ccd841dc16fb24e88f7f239c187/src/diffusers/utils/constants.py#L18
    from iopaint.cli import typer_app

    fix_window_pytorch()

    typer_app()
