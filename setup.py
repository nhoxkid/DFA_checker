from pathlib import Path
import platform
from setuptools import Extension, setup

source = Path(__file__).parent / "dfa_checker" / "_accelerator.cpp"
compile_args = []
link_args = []
if platform.system() == "Windows":
    compile_args = ["/std:c++17", "/O2"]
else:
    compile_args = ["-std=c++17", "-O3"]

module = Extension(
    "dfa_checker._accelerator",
    sources=[str(source)],
    language="c++",
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(
    name="dfa-checker",
    version="0.0.0",
    packages=["dfa_checker"],
    ext_modules=[module],
)
