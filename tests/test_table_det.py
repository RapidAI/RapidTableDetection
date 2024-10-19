import sys
from pathlib import Path

import pytest


cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

sys.path.append(str(root_dir))
test_file_dir = cur_dir / "test_files"


@pytest.mark.parametrize(
    "img_path, expected",
    [("chip.jpg", 1), ("doc.png", 2)],
)
def test_input_normal(img_path, expected):
    from rapid_table_det import TableDetector

    table_det = TableDetector()
    img_path = test_file_dir / img_path
    result, elapse = table_det(img_path)
    assert len(result) == expected
