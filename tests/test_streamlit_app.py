from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_streamlit_app_renders_with_public_dataset() -> None:
    root = Path(__file__).parents[1]
    if not (root / "data/processed/synthstrip_data_v1.5_2d").is_dir():
        pytest.skip("Official dataset is not available in this environment.")
    app = AppTest.from_file(root / "streamlit_app.py", default_timeout=20).run()
    assert not app.exception
    assert len(app.selectbox) >= 2
    assert len(app.slider) == 4
    assert len(app.tabs) == 3
