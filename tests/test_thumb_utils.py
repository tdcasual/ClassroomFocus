from tools.thumb_utils import effective_refresh_sec


def test_effective_refresh_sec_links_to_preview_interval():
    assert effective_refresh_sec(60.0, 0.1, 30) == 60.0
    assert effective_refresh_sec(1.0, 0.5, 10) == 5.0
    assert effective_refresh_sec(0.0, 0.2, 0) == 0.0
