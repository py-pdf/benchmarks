import os
import platform
import re
import subprocess
from pathlib import Path


def get_processor_name() -> str:
    """Credits: https://stackoverflow.com/a/13078519/562769"""
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ["PATH"] = os.environ["PATH"] + os.pathsep + "/usr/sbin"
        return (
            subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
            .strip()
            .decode("utf-8")
        )
    elif platform.system() == "Linux":
        all_info = Path("/proc/cpuinfo").read_text()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, count=1)
    return ""
