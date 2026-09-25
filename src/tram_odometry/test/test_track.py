import math

from tram_odometry.track import GnssInit, LocalFrame, StraightTrack


def test_local_frame_axes():
    f = LocalFrame(55.80, 37.42, 150.0)
    e, n, u = f.forward(55.80, 37.42, 150.0)
    assert max(abs(e), abs(n), abs(u)) < 1e-6
    e, n, _ = f.forward(55.801, 37.42, 150.0)
    assert abs(e) < 0.01 and 110.0 < n < 112.0   # 0,001° широты ≈ 111 м на север
    e, n, _ = f.forward(55.80, 37.421, 150.0)
    assert 62.0 < e < 63.0 and abs(n) < 0.1      # 0,001° долготы на 55,8° с. ш. ≈ 62,6 м
    assert abs(f.forward(55.80, 37.42, 160.0)[2] - 10.0) < 1e-6


def test_straight_track():
    x, y, z, yaw = StraightTrack(1.0, 2.0, 3.0, math.pi / 2).pose(10.0)
    assert abs(x - 1.0) < 1e-9 and abs(y - 12.0) < 1e-9 and z == 3.0


def test_gnss_init_heading_from_motion():
    init = GnssInit(window_s=5.0, min_move_m=3.0)
    init.start(0.0)
    for i in range(6):                           # едем на север 2 м/с
        init.fix(i * 0.9, 55.80 + i * 1.8 / 111320.0, 37.42, 150.0)
    assert init.heading_known
    assert abs(init.track().yaw - math.pi / 2) < 0.01
    assert not init.expired(4.9) and init.expired(5.1)


def test_gnss_init_without_motion_or_fix():
    init = GnssInit()
    init.start(0.0)
    init.fix(0.5, 55.80, 37.42, 150.0)
    init.fix(1.0, 55.8000001, 37.42, 150.0)
    init.fix(1.5, float('nan'), 37.42, 150.0)
    init.fix(2.0, 56.0, 37.42, 150.0, status=-1)  # нет решения: не используем
    assert not init.heading_known
    assert init.track(fallback_yaw=1.0).yaw == 1.0
