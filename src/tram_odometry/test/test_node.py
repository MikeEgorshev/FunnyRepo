"""Нода: сброс оценщика при повторном проигрывании bag и живучесть колбэков."""
import pytest
import rclpy
from builtin_interfaces.msg import Time
from tram_vehicle_msgs.msg import VelocitySensor

from tram_odometry.node import TramOdometryNode


@pytest.fixture
def node():
    rclpy.init()
    n = TramOdometryNode()
    yield n
    n.destroy_node()
    rclpy.shutdown()


def wheel(t, kmh=0.0):
    m = VelocitySensor()
    sec = int(t)
    m.header.stamp = Time(sec=sec, nanosec=int(round((t - sec) * 1e9)))
    m.velocity = kmh
    return m


def feed(node, t0, n, dt=0.05):
    for i in range(n):
        node._wheel(wheel(t0 + i * dt), i % 2 == 0)


def test_replay_of_the_bag_resets_the_estimator(node):
    feed(node, 1000.0, 40)
    old = node.est
    feed(node, 900.0, TramOdometryNode.RESET_BACK_N - 1)
    assert node.est is old          # пока это может быть пачка битых меток
    feed(node, 900.5, 2)
    assert node.est is not old
    assert old.map is not node.est.map and old.model is not node.est.model
    assert 900.0 < node.est.t < 901.0 and node.last_in[0] > 900.0


def test_single_stale_stamp_does_not_reset(node):
    feed(node, 1000.0, 40)
    old = node.est
    node._wheel(wheel(0.0), True)
    feed(node, 1002.0, 40)
    assert node.est is old


def test_exception_in_callback_is_logged_not_raised(node):
    def boom(*args):
        raise ValueError('math domain error')
    node.est.on_wheel = boom
    node._wheel(wheel(1000.0), True)
    assert node.errors == 1
