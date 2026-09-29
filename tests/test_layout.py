import pytest
from pydantic import ValidationError

from jevex import BBox, Component, DomLocation, ImageLocation, PageLocation


def tree() -> Component:
    return Component(
        id="c0",
        type="section",
        heading_trail=["Specifications"],
        location=DomLocation(dom_path="/html/body/main"),
        children=[
            Component(
                id="c1",
                type="heading",
                text="Performance",
                location=DomLocation(dom_path="/html/body/main/h2"),
            ),
            Component(
                id="c2",
                type="table",
                heading_trail=["Specifications", "Performance"],
                location=PageLocation(page=3, bbox=BBox(x0=10, y0=20, x1=300, y1=400)),
                children=[
                    Component(
                        id="c3",
                        type="image",
                        location=ImageLocation(src="https://example.com/chart.png"),
                    )
                ],
            ),
        ],
    )


def test_walk_is_depth_first_in_reading_order() -> None:
    assert [c.id for c in tree().walk()] == ["c0", "c1", "c2", "c3"]


def test_find() -> None:
    root = tree()
    found = root.find("c3")
    assert found is not None
    assert found.type == "image"
    assert root.find("missing") is None


def test_round_trip_keeps_location_kinds() -> None:
    root = tree()
    restored = Component.model_validate_json(root.model_dump_json())
    assert restored == root
    assert isinstance(restored.children[1].location, PageLocation)
    assert isinstance(restored.children[1].children[0].location, ImageLocation)


def test_bbox_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        BBox(x0=10, y0=0, x1=5, y1=5)


def test_page_is_one_based() -> None:
    with pytest.raises(ValidationError):
        PageLocation(page=0)


def test_unknown_component_type_rejected() -> None:
    with pytest.raises(ValidationError):
        Component.model_validate(
            {"id": "x", "type": "sidebar", "location": {"kind": "dom", "dom_path": "/"}}
        )
