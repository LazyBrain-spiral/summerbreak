import math

from src.ingest.intrinsics import intrinsics_prior, normalise_focal_35mm
from src.ingest.telemetry import _parse_srt_block, parse_srt, synchronize

NEW_DJI = (
    "3\n00:00:00,066 --> 00:00:00,100\n"
    "FrameCnt: 3, DiffTime: 33ms\n2026-09-20 10:00:00.066\n"
    "[iso: 100] [shutter: 1/1000.0] [fnum: 2.8] [ev: 0] [focal_len: 24.00] [latitude: 28.613915] "
    "[longitude: 77.209008] [rel_alt: 52.300 abs_alt: 268.100] [gb_yaw: 90.2 gb_pitch: -60.0 gb_roll: 0.0]"
)

OLD_DJI = (
    "1\n00:00:00,000 --> 00:00:00,033\nSrt:1/180\n"
    "[latitude: 28.613900] [longitude: 77.209000] [altitude: 55.00] [rel_alt: 55.00]\n"
    "[gimbal_pitch: -60.0] [gimbal_roll: 0.0] [gimbal_yaw: 45.0]"
)

PHANTOM = (
    "7\n00:00:06,000 --> 00:00:07,000\nHOME(77.2090,28.6139) 2026.09.20 10:00:06\n"
    "GPS(28.6141,77.2092,19) BAROMETER:61.4\nISO:100 Shutter:500 EV:0 Fnum:F2.8"
)


def test_new_dji_block():
    r = _parse_srt_block(NEW_DJI)
    assert math.isclose(r["lat"], 28.613915) and math.isclose(r["lon"], 77.209008)
    assert r["rel_alt"] == 52.3 and r["abs_alt"] == 268.1
    assert r["gimbal_yaw"] == 90.2 and r["gimbal_pitch"] == -60.0
    assert r["focal_len"] == 24.0 and r["frame_cnt"] == 3
    assert math.isclose(r["t"], 0.066)


def test_old_dji_block():
    r = _parse_srt_block(OLD_DJI)
    assert r["rel_alt"] == 55.0 and r["gimbal_pitch"] == -60.0 and r["gimbal_yaw"] == 45.0


def test_phantom_block():
    r = _parse_srt_block(PHANTOM)
    assert math.isclose(r["lat"], 28.6141) and math.isclose(r["lon"], 77.2092)
    assert r["rel_alt"] == 61.4


def test_parse_file_and_sync(tmp_path):
    blocks = []
    for i in range(10):
        blocks.append(
            f"{i + 1}\n00:00:0{i},000 --> 00:00:0{i},500\n"
            f"<font size=\"28\">[latitude: {28.6 + i * 1e-4:.6f}] [longitude: 77.2] "
            f"[rel_alt: {50 + i}.0 abs_alt: {250 + i}.0] "
            f"[gb_yaw: {((170 + i * 3 + 180) % 360) - 180:.1f} gb_pitch: -60.0 gb_roll: 0.0]</font>\n")
    p = tmp_path / "f.srt"
    p.write_text("\n".join(blocks))
    recs = parse_srt(str(p))
    assert len(recs) == 10
    rows = synchronize(recs, [{"name": "a.jpg", "t": 2.0}, {"name": "b.jpg", "t": 20.0}])
    assert math.isclose(rows[0]["lat"], 28.6 + 2e-4, abs_tol=1e-8)  # caption start time is the frame time
    assert rows[0]["alt_source"] == "abs_alt"
    assert rows[0]["in_span"] and not rows[1]["in_span"]
    # yaw goes 176 -> 179 -> -178 ...; interpolation must wrap, not sweep through 0
    rows = synchronize(recs, [{"name": "c.jpg", "t": 3.75}])
    yaw = rows[0]["gimbal_yaw"]
    assert abs(abs(yaw) - 180.0) < 2.0, yaw


def test_intrinsics_prior():
    assert normalise_focal_35mm(240.0) == 24.0
    p = intrinsics_prior(1920, 1080, 24.0)
    assert p["source"] == "telemetry_focal_len"
    assert math.isclose(p["f"], 24.0 * math.hypot(1920, 1080) / 43.2666, rel_tol=1e-9)
    assert intrinsics_prior(1920, 1080, None)["source"] == "dfov_default"
