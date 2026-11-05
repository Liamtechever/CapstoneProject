import pytest

from core.errors import AssistantError
from experiments.fashion.app import (
    get_all_clothing_items,
    init_db,
    remove_clothing_item,
    save_clothing_item,
    validate_outfit,
)

ITEMS = [
    (1, "shirt.jpg", "shirt", ""),
    (2, "trousers.jpg", "trousers", ""),
    (3, "shoes.jpg", "shoes", ""),
]


@pytest.mark.parametrize(
    "response",
    [
        '{"top":1,"bottom":1,"shoes":3}',
        '{"top":99,"bottom":2,"shoes":3}',
        '{"top":true,"bottom":2,"shoes":3}',
        '{"top":"1","bottom":2,"shoes":3}',
        '{"top":1}',
        "[]",
        "invalid",
    ],
)
def test_invalid_outfit_rejected(response):
    with pytest.raises(AssistantError):
        validate_outfit(response, ITEMS)


def test_outfit_accepts_existing_ids_and_unavailable_category():
    outfit = validate_outfit('{"top":1,"bottom":2,"shoes":null}', ITEMS)
    assert outfit["top"] == ITEMS[0]
    assert outfit["shoes"] is None


def test_database_crud_and_image_deletion(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"sample")
    conn = init_db(tmp_path / "test.db")
    try:
        save_clothing_item(conn, image, "shirt ' quoted", "")
        items = get_all_clothing_items(conn)
        assert items[0][2] == "shirt ' quoted"
        remove_clothing_item(conn, items[0][0], tmp_path)
        assert not image.exists()
        assert get_all_clothing_items(conn) == []
    finally:
        conn.close()


def test_delete_cannot_escape_wardrobe(tmp_path):
    outside = tmp_path / "keep.txt"
    outside.write_text("keep")
    root = tmp_path / "wardrobe"
    conn = init_db(root / "test.db")
    try:
        save_clothing_item(conn, outside, "bad path", "")
        with pytest.raises(AssistantError):
            remove_clothing_item(conn, 1, root)
        assert outside.read_text() == "keep"
        assert len(get_all_clothing_items(conn)) == 1
    finally:
        conn.close()
