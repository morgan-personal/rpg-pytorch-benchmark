from types import SimpleNamespace

from rpgbench import precheck


def make_psutil(plugged=True, names=(), battery=True, raises=False):
    def sensors_battery():
        if raises:
            raise RuntimeError("no sensors")
        if not battery:
            return None
        return SimpleNamespace(power_plugged=plugged)

    def process_iter(attrs=None):
        if raises:
            raise RuntimeError("no processes")
        return [SimpleNamespace(info={"name": n}) for n in names]

    return SimpleNamespace(sensors_battery=sensors_battery,
                           process_iter=process_iter)


def importer(psutil_mod):
    def import_fn(name):
        if name == "psutil" and psutil_mod is not None:
            return psutil_mod
        raise ImportError(name)

    return import_fn


def test_heavy_app_list_size_and_case():
    assert len(precheck.HEAVY_APPS) >= 5
    assert all(a == a.lower() for a in precheck.HEAVY_APPS)


def test_battery_warning():
    res = precheck.run_precheck(importer(make_psutil(plugged=False)),
                                platform_name="linux")
    assert res.on_battery is True
    assert any("battery" in w.lower() for w in res.warnings)


def test_plugged_in_no_warning():
    res = precheck.run_precheck(importer(make_psutil(plugged=True)),
                                platform_name="linux")
    assert res.on_battery is False
    assert res.warnings == []


def test_heavy_process_case_insensitive():
    psutil_mod = make_psutil(names=["explorer.exe", "OBS64.EXE"])
    res = precheck.run_precheck(importer(psutil_mod), platform_name="linux")
    assert res.heavy_apps == ["OBS64.EXE"]
    assert any("OBS64.EXE" in w for w in res.warnings)


def test_psutil_missing_limited_not_raising():
    res = precheck.run_precheck(importer(None), platform_name="linux")
    assert any("skipped" in n for n in res.notes)
    assert res.heavy_apps == []


def test_psutil_missing_windows_battery_fallback():
    res = precheck.run_precheck(importer(None), win_battery_fn=lambda: True,
                                platform_name="win32")
    assert res.on_battery is True
    assert any("battery" in w.lower() for w in res.warnings)


def test_win_battery_fn_raising_is_contained():
    def boom():
        raise OSError("nope")

    res = precheck.run_precheck(importer(None), win_battery_fn=boom,
                                platform_name="win32")
    assert res.on_battery is None


def test_psutil_calls_raising_are_contained():
    res = precheck.run_precheck(importer(make_psutil(raises=True)),
                                platform_name="linux")
    assert res.warnings == []
    assert res.notes


def test_import_fn_raising_oddly_is_contained():
    def import_fn(name):
        raise RuntimeError("weird")

    res = precheck.run_precheck(import_fn, platform_name="linux")
    assert isinstance(res.warnings, list)


def test_never_reads_input(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("input read")

    monkeypatch.setattr("builtins.input", fail)
    precheck.run_precheck(importer(None), platform_name="linux")
