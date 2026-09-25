"""Поля nav_msgs/Odometry из состояния фильтра — без ROS, чтобы их можно было тестировать."""
import math


def yaw_to_quaternion(yaw):
    """(x, y, z, w) для поворота вокруг оси z."""
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def pose_covariance(var_s, yaw, var_cross, var_z, var_yaw):
    """Ковариация 6×6 (x, y, z, roll, pitch, yaw) построчно, 36 чисел.

    Неопределённость дистанции var_s ложится вдоль курса, var_cross — поперёк пути.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    cov = [0.0] * 36
    cov[0] = var_s * c * c + var_cross * s * s
    cov[7] = var_s * s * s + var_cross * c * c
    cov[1] = cov[6] = (var_s - var_cross) * s * c
    cov[14] = var_z
    cov[21] = cov[28] = 1e6        # крен и тангаж не оцениваем
    cov[35] = var_yaw
    return cov


def twist_covariance(var_v):
    """Ковариация скорости 6×6: оцениваем только продольную скорость."""
    cov = [0.0] * 36
    cov[0] = var_v
    for i in (7, 14, 21, 28, 35):
        cov[i] = 1e6
    return cov


class StampClock:
    """Метки выхода: последняя метка входа плюс время, прошедшее с её прихода.

    Метки не убывают; экстраполяция не дольше max_extrapolation_s.
    """

    def __init__(self, max_extrapolation_s=0.5):
        self.max_extrapolation_s = max_extrapolation_s
        self.stamp = None
        self.arrival = None
        self.last_out = None

    def reset(self):
        self.stamp = self.arrival = self.last_out = None

    def input(self, stamp, arrival):
        if self.stamp is None or stamp >= self.stamp:
            self.stamp, self.arrival = stamp, arrival

    def output(self, now):
        """Метка для публикации в момент now (часы ноды); None — входов ещё не было."""
        if self.stamp is None:
            return None
        t = self.stamp + min(max(now - self.arrival, 0.0), self.max_extrapolation_s)
        if self.last_out is not None and t < self.last_out:
            t = self.last_out
        self.last_out = t
        return t
