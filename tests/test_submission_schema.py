from pathlib import Path
import tempfile
import unittest
from project_override.submission.kaggle_submission import (
    validate_submission_parquet,
    write_submission_parquet,
)

def test_parquet_schema_validation(tmp_path: Path = Path(tempfile.mkdtemp())) -> None:
    out_file = tmp_path / "submission.parquet"
    write_submission_parquet(
        [{"row_id": "ls20_0", "game_id": "ls20", "end_of_game": True, "score": 54.67}],
        output_path=str(out_file),
    )
    assert validate_submission_parquet(str(out_file)) is True

class TestSchema(unittest.TestCase):
    def test_schema(self) -> None:
        test_parquet_schema_validation()
