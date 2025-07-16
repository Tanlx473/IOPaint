"""使得可以通过 ``python -m iopaint`` 启动命令行界面。

Allows ``python -m iopaint`` to start the CLI directly.
"""


from iopaint import entry_point

if __name__ == "__main__":
    entry_point()
