# -*- encoding: utf-8 -*-
# @Author: Jocker1212
# @Contact: xinyijianggo@gmail.com
import sys
from typing import List, Union
from pathlib import Path
from get_pypi_latest_version import GetPyPiLatestVersion

import setuptools


def read_txt(txt_path: Union[Path, str]) -> List[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        data = [v.rstrip("\n") for v in f]
    return data


MODULE_NAME = "rapid_table_det"

obtainer = GetPyPiLatestVersion()
try:
    latest_version = obtainer(MODULE_NAME)
except Exception:
    latest_version = "0.0.0"

VERSION_NUM = obtainer.version_add_one(latest_version)

if len(sys.argv) > 2:
    match_str = " ".join(sys.argv[2:])
    matched_versions = obtainer.extract_version(match_str)
    if matched_versions:
        VERSION_NUM = matched_versions
sys.argv = sys.argv[:2]

setuptools.setup(
    name=MODULE_NAME,
    version=VERSION_NUM,
    platforms="Any",
    description="table detection with onnx model",
    long_description="table detection with onnx model",
    author="jockerK",
    author_email="xinyijianggo@gmail.com",
    url="https://github.com/Joker1212/RapidTableDetection",
    license="Apache-2.0",
    install_requires=read_txt("requirements.txt"),
    include_package_data=False,
    packages=[MODULE_NAME, f"{MODULE_NAME}.models", f"{MODULE_NAME}.utils"],
    package_data={"": [".gitkeep"]},
    keywords=["obj detection,ocr,table-recognition"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8,<3.13",
)
