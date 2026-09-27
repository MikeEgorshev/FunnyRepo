"""ROS adapter tests using deterministic message/clock doubles; not DDS tests."""
import importlib
import math
from pathlib import Path
import sys
import types

import pytest


class Message:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def __getattr__(self, key):
        if key.startswith('__'):
            raise AttributeError(key)
        value = [0.] * 36 if key == 'covariance' else Message()
        setattr(self, key, value)
        return value


@pytest.fixture
def node(monkeypatch):
    clock = [0.]
    published, subscriptions = {}, {}

    class FakeNode:
        def __init__(self, *args):
            self.params = {'use_sim_time': True}

        def declare_parameter(self, name, value):
            self.params[name] = True if name in ('primary_sync', 'gnss_corrections') else value

        def get_parameter(self, name):
            return types.SimpleNamespace(value=self.params[name])

        def create_subscription(self, cls, topic, callback, qos):
            subscriptions[topic] = callback
            return topic

        def destroy_subscription(self, topic):
            subscriptions.pop(topic, None)

        def create_publisher(self, cls, topic, qos):
            return types.SimpleNamespace(publish=lambda m: published.setdefault(topic, []).append(m))

        def create_timer(self, *args):
            pass

        def get_clock(self):
            return types.SimpleNamespace(now=lambda: types.SimpleNamespace(
                nanoseconds=round(clock[0]*1e9), to_msg=lambda: stamp(clock[0])))

        def get_logger(self):
            return types.SimpleNamespace(info=lambda *args: None)

    def module(name, **fields):
        m = types.ModuleType(name)
        m.__dict__.update(fields)
        monkeypatch.setitem(sys.modules, name, m)

    module('rclpy')
    module('rclpy.node', Node=FakeNode)
    module('rclpy.qos', QoSProfile=Message, HistoryPolicy=Message(KEEP_LAST=1),
           ReliabilityPolicy=Message(BEST_EFFORT=1, RELIABLE=2))
    module('ament_index_python')
    module('ament_index_python.packages',
           get_package_share_directory=lambda _: str(Path(__file__).resolve().parents[1]))
    for pkg, names in {'builtin_interfaces': ['Time'], 'nav_msgs': ['Odometry'],
                       'sensor_msgs': ['NavSatFix'],
                       'diagnostic_msgs': ['DiagnosticArray', 'DiagnosticStatus', 'KeyValue'],
                       'tram_vehicle_msgs': ['VelocitySensor', 'DriverControllerCommand']}.items():
        module(pkg)
        module(pkg + '.msg', **{name: Message for name in names})
    Message.OK, Message.WARN = 0, 1
    for name in ['tram_odometry.node', 'tram_odometry.integrated_node']:
        monkeypatch.delitem(sys.modules, name, raising=False)
    cls = importlib.import_module('tram_odometry.integrated_node').TramOdometryNode
    instance = cls()
    yield instance, clock, published, subscriptions
    for name in ['tram_odometry.node', 'tram_odometry.integrated_node']:
        sys.modules.pop(name, None)


def stamp(t):
    sec, nsec = divmod(round(t*1e9), 1_000_000_000)
    return Message(sec=sec, nanosec=nsec)


def test_enforced_reference_isolation_and_long_outage(node):
    n, clock, published, subs = node
    assert '/localization/kinematic_state' not in subs
    n.est.on_gnss(100., 55.8104, 37.4623, 168.)
    for i in range(5):
        clock[0] = i*.1
        msg = Message(header=Message(stamp=stamp(100.+clock[0])), velocity=18.+i*.001)
        n._wheel(msg, False)
        n._wheel(msg, True)
    for i in range(1, 301):
        clock[0] = .4+i*.02
        n._keepalive()
    messages = published['/result/position']
    times = [m.header.stamp.sec+m.header.stamp.nanosec*1e-9 for m in messages]
    assert times[-1] > 106.
    assert max(b-a for a, b in zip(times, times[1:])) <= .100001
    assert all(math.isfinite(m.pose.pose.position.x) for m in messages)
    n._diagnostics()
    assert n.proc_ms.maxlen == 500


def test_paused_sim_clock_does_not_create_future_outputs(node):
    n, clock, published, _ = node
    n.est.on_gnss(100., 55.8104, 37.4623, 168.)
    msg = Message(header=Message(stamp=stamp(100.)), velocity=18.)
    n._wheel(msg, True)
    count = len(published['/result/position'])
    for _ in range(100):
        n._keepalive()
    assert len(published['/result/position']) == count


def test_covariance_is_symmetric_and_nonnegative(node):
    n, clock, published, _ = node
    out = dict(v=1., x=1., y=2., z=3., yaw=.7, var_s=20., var_v=.1, frame='map')
    n._publish(out, stamp(1.), 0.)
    c = published['/result/position'][-1].pose.covariance
    assert c[1] == c[6]
    assert c[0]*c[7] - c[1]*c[6] >= 0


def test_late_fix_subscription_follows_correction_flag(node):
    n, _, _, subs = node
    n.est.on_gnss(100., 55.8104, 37.4623, 168.)
    msg = Message(header=Message(stamp=stamp(110.)), status=Message(status=0),
                  latitude=55.8104, longitude=37.4623, altitude=168.)
    n._gnss(msg)
    assert n.gnss_sub in subs
    n.est.p.gnss_corrections = False
    n._gnss(msg)
    assert n.gnss_sub is None
