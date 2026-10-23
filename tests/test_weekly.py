from pathlib import Path

import pytest

from stockpicker import weekly


def test_weekly_output_rejects_paths_outside_project(tmp_path: Path):
    outside = tmp_path / "packet.json"

    with pytest.raises(ValueError, match="inside the project directory"):
        weekly._project_local_output(outside)
