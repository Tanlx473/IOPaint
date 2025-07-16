"""用于安装可选插件依赖的辅助模块。

Helper module to install optional plugin dependencies.
"""


import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def install_plugins_package():
    install("onnxruntime<=1.19.2")
    install("rembg[cpu]")
