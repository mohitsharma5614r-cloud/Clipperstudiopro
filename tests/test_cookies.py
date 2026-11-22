import os
import tempfile
from pathlib import Path
from starlette.testclient import TestClient

from clipper.download import get_youtube_cookies_options
from clipper.api import app


def test_cookies_options_empty():
    cfg = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        cf, cb = get_youtube_cookies_options(cfg, workspace_dir=Path(tmpdir))
        # Might be None or empty unless a global cookies.txt exists
        assert cb is None


def test_cookies_options_explicit_custom():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"# Netscape HTTP Cookie File\n")
        f.flush()
        f_path = f.name

    try:
        cf, cb = get_youtube_cookies_options(
            cfg={},
            custom_cookies_file=f_path,
            custom_cookies_browser="firefox",
        )
        assert cf == str(Path(f_path).resolve())
        assert cb == "firefox"
    finally:
        os.unlink(f_path)


def test_cookies_options_autodiscover():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cookie_file = tmp_path / "cookies.txt"
        cookie_file.write_text("# Netscape HTTP Cookie File\n.youtube.com TRUE / TRUE 0 name val\n")

        cf, cb = get_youtube_cookies_options(cfg={}, workspace_dir=tmp_path)
        assert cf == str(cookie_file.resolve())


def test_api_cookies_endpoints():
    client = TestClient(app)
    # Test GET /cookies/status
    res = client.get("/cookies/status")
    assert res.status_code == 200
    data = res.json()
    assert "has_cookies" in data

    # Test POST /cookies/upload with invalid content
    res = client.post("/cookies/upload", json={"content": "invalid cookies content"})
    assert res.status_code == 400

    # Test POST /cookies/upload with valid netscape content
    valid_content = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t2147483647\tSID\tsome_token\n"
    res = client.post("/cookies/upload", json={"content": valid_content})
    assert res.status_code == 200
    upload_data = res.json()
    assert upload_data["success"] is True

    # Check status again
    res = client.get("/cookies/status")
    assert res.status_code == 200
    assert res.json()["has_cookies"] is True

    # Cleanup created cookies.txt in workspace
    if Path("cookies.txt").is_file():
        Path("cookies.txt").unlink()
    if (Path("workspace") / "cookies.txt").is_file():
        (Path("workspace") / "cookies.txt").unlink()
