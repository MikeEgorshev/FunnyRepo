import math

from tram_odometry.outputs import StampClock, pose_covariance, twist_covariance, yaw_to_quaternion


def test_quaternion_of_yaw():
    x, y, z, w = yaw_to_quaternion(math.pi / 2)
    assert x == y == 0.0 and math.isclose(z, math.sqrt(0.5)) and math.isclose(w, math.sqrt(0.5))


def test_pose_covariance_follows_heading():
    cov = pose_covariance(var_s=100.0, yaw=0.0, var_cross=1.0, var_z=4.0, var_yaw=0.01)
    assert cov[0] == 100.0 and cov[7] == 1.0 and abs(cov[1]) < 1e-12
    cov = pose_covariance(var_s=100.0, yaw=math.pi / 2, var_cross=1.0, var_z=4.0, var_yaw=0.01)
    assert math.isclose(cov[7], 100.0) and math.isclose(cov[0], 1.0, abs_tol=1e-9)
    assert len(cov) == 36 and cov[14] == 4.0 and cov[35] == 0.01
    assert twist_covariance(0.04)[0] == 0.04


def test_stamp_clock_extrapolates_and_never_goes_back():
    c = StampClock(max_extrapolation_s=0.5)
    assert c.output(0.0) is None
    c.input(stamp=100.0, arrival=1.0)
    assert c.output(1.1) == 100.1
    assert c.output(5.0) == 100.5                # не дальше 0,5 с
    c.input(stamp=99.0, arrival=5.1)              # старая метка не двигает часы
    assert c.output(5.2) == 100.5
    c.input(stamp=100.2, arrival=5.3)             # новая метка раньше последнего выхода
    assert c.output(5.3) == 100.5
