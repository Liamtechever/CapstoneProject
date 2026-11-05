from scripts.setup_local import setup


def test_local_setup_creates_key_without_overwriting_existing_files(tmp_path):
    (tmp_path / ".env.example").write_text("DJANGO_SECRET_KEY=\nDJANGO_DEBUG=true\n")
    (tmp_path / "settings.example.json").write_text('{"user_vram":4}')
    setup(tmp_path)
    original = (tmp_path / ".env").read_text()
    key = original.splitlines()[0].split("=", 1)[1]
    assert len(key) >= 50
    assert (tmp_path / "settings.json").exists()
    (tmp_path / "settings.json").write_text("keep local settings")
    setup(tmp_path)
    assert (tmp_path / ".env").read_text() == original
    assert (tmp_path / "settings.json").read_text() == "keep local settings"
