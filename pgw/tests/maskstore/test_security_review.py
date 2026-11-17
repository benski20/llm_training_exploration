"""Security review regressions: external counter required to serve, network filesystems refused."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pgw.core import RING_K40_A12, ConfigError
from pgw.maskstore import MaskStore
from pgw.maskstore import store as store_mod
from pgw.maskstore.store import _mount_fstype, is_network_filesystem

from .conftest import FP, ToyModel, write_segment


def test_reserve_refuses_without_external_counter(tmp_path: Path, open_store: Any, model: ToyModel) -> None:
    """counter='none' + journal on the same volume = an undetectable snapshot restore: refuse to serve."""
    store = open_store(allow_no_external_counter=False)
    write_segment(store, model, 4)
    with pytest.raises(ConfigError, match="external monotonic counter"):
        store.reserve(FP, RING_K40_A12, 1)
    assert store.available_slots(FP, RING_K40_A12) == 4  # nothing consumed
    # a file counter on separate storage makes it servable
    counted = open_store(root=tmp_path / "counted", allow_no_external_counter=False, counter="file",
                         counter_file=tmp_path / "separate-volume" / "counter")
    write_segment(counted, model, 4)
    counted.reserve(FP, RING_K40_A12, 2).close()


def test_default_settings_require_counter() -> None:
    from pgw.maskstore import MaskStoreSettings

    assert MaskStoreSettings.model_fields["allow_no_external_counter"].default is False
    assert MaskStoreSettings.model_fields["allow_unsigned_partner_segments"].default is False
    assert MaskStoreSettings.model_fields["allow_network_filesystem"].default is False


MOUNTINFO = """\
22 1 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw
40 22 0:35 / /var/lib/pgw rw,relatime shared:2 - nfs4 server:/export rw,vers=4.2
41 40 0:36 / /var/lib/pgw/local rw,relatime shared:3 - xfs /dev/sdb1 rw
42 22 0:37 / /mnt/my\\040share rw - fuse.sshfs host:/x rw
"""


def test_mount_fstype_parsing(tmp_path: Path) -> None:
    mi = tmp_path / "mountinfo"
    mi.write_text(MOUNTINFO)
    assert _mount_fstype(Path("/var/lib/pgw/masks"), str(mi)) == "nfs4"
    assert _mount_fstype(Path("/var/lib/pgw/local/masks"), str(mi)) == "xfs"
    assert _mount_fstype(Path("/var/lib/pgwx"), str(mi)) == "ext4"
    assert _mount_fstype(Path("/mnt/my share/m"), str(mi)) == "fuse.sshfs"
    assert _mount_fstype(Path("/x"), str(tmp_path / "missing")) is None
    assert is_network_filesystem("nfs4") and is_network_filesystem("cifs") and is_network_filesystem("fuse.sshfs")
    assert is_network_filesystem("ceph") and not is_network_filesystem("ext4") and not is_network_filesystem(None)


def test_store_refuses_network_filesystem(tmp_path: Path, kp: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store_mod, "_mount_fstype", lambda path, *a: "nfs4")
    with pytest.raises(ConfigError, match="network filesystem"):
        MaskStore.open(tmp_path / "nfs-store", kp, low_water_slots=0)
    s = MaskStore.open(tmp_path / "nfs-store", kp, low_water_slots=0, allow_network_filesystem=True)
    s.close()
