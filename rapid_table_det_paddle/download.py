# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import tarfile
import zipfile
import requests
from tqdm import tqdm
import logging


# 初始化日志记录器
def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# 获取当前脚本文件的目录
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# 设置存储的根路径
MODELS_DIR = os.path.abspath(os.path.join(current_file_dir, "models"))


def download_with_progressbar(url, save_path):
    logger = get_logger()
    if save_path and os.path.exists(save_path):
        logger.info(f"Path {save_path} already exists. Skipping...")
        return
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size_in_bytes = int(response.headers.get("content-length", 1))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(save_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
    else:
        logger.error("Something went wrong while downloading models")
        sys.exit(0)


def maybe_download(model_storage_directory, url):
    supported_extensions = [".tar", ".zip"]
    extension = os.path.splitext(url)[1].lower()

    if extension not in supported_extensions:
        raise ValueError(
            f"Unsupported file extension: {extension}. Only .tar and .zip are supported."
        )

    tmp_path = os.path.join(model_storage_directory, url.split("/")[-1])
    logger = get_logger()
    logger.info(f"download {url} to {tmp_path}")
    os.makedirs(model_storage_directory, exist_ok=True)
    download_with_progressbar(url, tmp_path)

    if extension == ".tar":
        with tarfile.open(tmp_path, "r") as tarObj:
            tarObj.extractall(path=model_storage_directory)
    elif extension == ".zip":
        with zipfile.ZipFile(tmp_path, "r") as zipObj:
            zipObj.extractall(path=model_storage_directory)

    os.remove(tmp_path)
