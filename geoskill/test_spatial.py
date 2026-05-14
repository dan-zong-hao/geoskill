import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoskill.spatial import parse_locator, spatial_reward, spatial_violation_type


GT = [100, 100, 200, 200]


def test_parse_locator():
    assert parse_locator("What color is the left-most building?")["axes"] == ["left"]
    assert set(parse_locator("Find the upper-right tank")["axes"]) == {"top", "right"}
    assert parse_locator("What is the largest ship?")["backlog_families"] == ["largest"]


def test_spatial_reward_axes():
    assert spatial_reward([50, 100, 120, 200], GT, parse_locator("left-most object"))["spatial_ok"] == 1.0
    assert spatial_violation_type([230, 100, 300, 200], GT, parse_locator("left-most object")) == "left_violated"
    assert spatial_reward([100, 20, 200, 80], GT, parse_locator("top-most object"))["spatial_ok"] == 1.0
    assert spatial_violation_type([100, 250, 200, 330], GT, parse_locator("top-most object")) == "top_violated"


def test_corner_partial():
    v = spatial_violation_type([220, 20, 300, 80], GT, parse_locator("upper-left object"))
    assert v == "corner_partial:left_violated"


if __name__ == "__main__":
    test_parse_locator()
    test_spatial_reward_axes()
    test_corner_partial()
    print("ok")
