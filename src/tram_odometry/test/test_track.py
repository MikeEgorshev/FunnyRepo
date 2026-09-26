import math

from tram_odometry.track import GnssInit, LocalFrame, MgrsLocal, RouteTrack, StraightTrack

M = MgrsLocal()


def test_mgrs_matches_reference_utm():
    # эталон — pyproj, EPSG:32637 минус (300000, 6100000); расхождение < 0,1 мм
    for (lat, lon), (x_ref, y_ref) in {(55.80, 37.42): (100955.1867, 84950.1040),
                                        (55.7995, 37.3885): (98979.4541, 84939.9584)}.items():
        x, y, z = M.forward(lat, lon, 150.0)
        assert abs(x - x_ref) < 1e-3 and abs(y - y_ref) < 1e-3 and z == 150.0


def test_local_frame_axes():
    f = LocalFrame(55.80, 37.42, 150.0)
    e, n, u = f.forward(55.80, 37.42, 150.0)
    assert max(abs(e), abs(n), abs(u)) < 1e-6
    e, n, _ = f.forward(55.801, 37.42, 150.0)
    assert abs(e) < 0.01 and 110.0 < n < 112.0   # 0,001° широты ≈ 111 м на север


def test_straight_track():
    x, y, z, yaw = StraightTrack(1.0, 2.0, 3.0, math.pi / 2).pose(10.0)
    assert abs(x - 1.0) < 1e-9 and abs(y - 12.0) < 1e-9 and z == 3.0


def _latlon_of(x, y):
    """Обратное к MgrsLocal численно (для тестов): точка в сетке -> (lat, lon)."""
    lat, lon = 55.80, 37.42
    for _ in range(20):
        x0, y0, _ = M.forward(lat, lon)
        lat += (y - y0) / 111320.0
        lon += (x - x0) / (111320.0 * math.cos(math.radians(lat)))
    return lat, lon


def _fixes(init, x, y, n=10, dt=0.4, rover=False, t0=0.0, step=0.0):
    for i in range(n):
        lat, lon = _latlon_of(x + step * i, y)
        init.fix(t0 + i * dt, lat, lon, 150.0, rover=rover)


def test_heading_from_antenna_baseline_while_standing():
    init = GnssInit(window_s=5.0)
    x0, y0, _ = M.forward(55.80, 37.42)
    _fixes(init, x0, y0)                         # master
    _fixes(init, x0, y0 + 12.436, rover=True)    # rover впереди, на север
    assert init.still and abs(init.heading() - math.pi / 2) < 0.01
    sx, sy, _ = init.start_point()
    assert math.hypot(sx - x0, sy - y0) < 0.05


def test_heading_from_motion_without_rover_and_gnss_jump_rejected():
    init = GnssInit(window_s=5.0)
    x0, y0, _ = M.forward(55.80, 37.42)
    _fixes(init, x0, y0, n=5)
    lat, lon = _latlon_of(x0 + 50.0, y0)
    init.fix(2.1, lat, lon, 150.0)               # прыжок на 50 м, трамвай стоит
    assert len(init.master) == 5 and init.heading() is None
    moving = GnssInit(window_s=5.0)
    _fixes(moving, x0, y0, n=10, step=1.0)       # едет на восток 2,5 м/с
    assert abs(moving.heading()) < 0.01 and not moving.still


def test_no_fix_means_relative_and_bad_fixes_are_ignored():
    init = GnssInit(window_s=5.0)
    init.start(0.0)
    init.fix(1.0, float('nan'), 37.42, 150.0)
    init.fix(1.5, 55.8, 37.42, 150.0, status=-1)
    init.fix(6.0, 55.8, 37.42, 150.0)            # после окна
    assert init.relative and init.expired(6.0)


def _ring(r=500.0):
    """Кольцо радиусом r в сетке MGRS с шагом 1 м против часовой стрелки."""
    x0, y0, _ = M.forward(55.80, 37.42)
    n = int(2 * math.pi * r)
    s = [float(i) for i in range(n)]
    x = [x0 + r * math.cos(i / r) for i in range(n)]
    y = [y0 + r * math.sin(i / r) for i in range(n)]
    z = [150.0 + 10.0 * math.sin(i / r) for i in range(n)]
    return RouteTrack(s, x, y, z, stops=[(100.0, 1.0), (1500.0, 2.0)]), x0, y0


def test_route_track_wraps_and_gives_grade():
    route, x0, y0 = _ring()
    assert abs(route.length - 2 * math.pi * 500.0) < 2.0
    a, b = route.pose(10.0), route.pose(10.0 + route.length)
    assert math.hypot(a[0] - b[0], a[1] - b[1]) < 1e-6
    assert abs(a[3] - math.pi / 2) < 0.05        # в начале кольца курс на север
    assert abs(route.grade_at(0.0) - 0.02) < 0.001   # dz/ds = 10/500 в начале


def test_route_locate_respects_direction():
    route, x0, y0 = _ring()
    px, py = x0 + 503.0 * math.cos(0.1), y0 + 503.0 * math.sin(0.1)    # 3 м снаружи кольца при s = 50 м
    s, d = route.locate(px, py, yaw=math.pi / 2)
    assert abs(s - 50.0) < 1.0 and abs(d - 3.0) < 0.1
    s_back, _ = route.locate(px, py, yaw=-math.pi / 2)                  # против направления карты
    assert abs(s_back - 50.0) > 100.0


def test_route_load_from_csv(tmp_path):
    lines = ['# карта', 's_m,lat,lon,alt']
    for i in range(200):
        lat, lon = _latlon_of(100955.0 + i, 84950.0)
        lines.append(f'{i}.0,{lat:.9f},{lon:.9f},150.0')
    (tmp_path / 'route.csv').write_text('\n'.join(lines))
    (tmp_path / 'stops.csv').write_text('s_m,sigma_m,count\n50.0,0.5,10\n150.0,1.0,4\n')
    route = RouteTrack.load(tmp_path / 'route.csv', M, tmp_path / 'stops.csv')
    assert route.stops == [(50.0, 0.5), (150.0, 1.0)]
    x, y, _, yaw = route.pose(20.0)
    assert abs(x - 100975.0) < 0.05 and abs(y - 84950.0) < 0.05 and abs(yaw) < 0.01
