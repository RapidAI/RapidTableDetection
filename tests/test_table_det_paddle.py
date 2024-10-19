import sys
from pathlib import Path

import pytest

from rapid_table_det_paddle.inference import TableDetector

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

sys.path.append(str(root_dir))
test_file_dir = cur_dir / "test_files"
table_det = TableDetector(
    obj_model_path=f"{root_dir}/rapid_table_det_paddle/models/obj_det_paddle",
    edge_model_path=f"{root_dir}/rapid_table_det_paddle/models/edge_det_paddle",
    cls_model_path=f"{root_dir}/rapid_table_det_paddle/models/cls_det_paddle",
    use_obj_det=True,
    use_edge_det=True,
    use_cls_det=True,
)


@pytest.mark.parametrize(
    "img_path, expected",
    [("chip.jpg", 1), ("doc.png", 2)],
)
def test_input_normal(img_path, expected):
    img_path = test_file_dir / img_path
    result, elapse = table_det(img_path)
    assert len(result) == expected
