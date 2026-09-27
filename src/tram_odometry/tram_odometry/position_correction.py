"""Position-only map anchoring. Never writes the wheel/model EKF or its map."""
import copy
import math


class PositionCorrection:
    def __init__(self):
        self.route = None
        self.anchor = self.measured = 0.0
        self.direction = 1
        self.orientation = 1
        self.previous = None
        self.last_seen = -math.inf
        self.last_fix = -math.inf
        self.accepted = 0
        self.variance = 0.0

    def pose(self, route, s, facing, travel):
        if self.route is None:
            return (*route.pose(s), facing)
        along = self.measured + self.direction * (travel - self.anchor)
        return (*self.route.pose(along), facing * self.orientation)

    def update(self, estimator, stamp, lat, lon, alt):
        e, p = estimator, estimator.p
        if (not p.gnss_corrections or not e.ready or e.mode == 'relative'
                or e.enu is None or e.t is None or abs(stamp-e.t) > 2.0
                or stamp <= self.last_seen):
            return
        self.last_seen = stamp
        s = e.s + e.v * (stamp-e.t)
        travel = e.travel + e.v * (stamp-e.t)
        x, y, _ = e.enu.forward(lat, lon, alt)
        px, py, _, yaw, _ = self.pose(e.map, s, e.facing, travel)
        if math.hypot(x-px, y-py) > p.gnss_gate_max_m:
            self.previous = None
            return
        # A private map carries branch selection; EKF stop/scale logic is isolated.
        route = copy.copy(e.map)
        measured, distance = route.locate_start(x, y)
        if route.active_spur is None:
            current = self.measured + self.direction*(travel-self.anchor) if self.route else s
            measured, distance = route.locate_near(x, y, current, p.gnss_window_m)
        if distance > p.gnss_lat_max_m:
            self.previous = None
            return
        tangent = route.pose(measured)[3]
        # Spur samples may run opposite to the incoming ring direction.
        direction = self.direction if self.route else 1
        if math.cos(tangent-yaw) < 0:
            direction = -direction
        orientation = self.orientation if self.route else 1
        if math.cos(tangent-yaw) < 0:
            orientation = -orientation
        branch = next((i for i, sp in enumerate(route.spurs)
                       if sp is route.active_spur), -1)
        previous = self.previous
        if previous is not None and branch == previous[1] and abs(travel-previous[2]) > 3.0:
            direction = 1 if (measured-previous[3])*(travel-previous[2]) >= 0 else -1
        self.previous = (stamp, branch, travel, measured)
        # Two agreeing fixes reject isolated jumps, including spur switches.
        if (previous is None or stamp-previous[0] > 30.0
                or branch != previous[1]
                or abs(measured-previous[3]-direction*(travel-previous[2])) > p.gnss_agree_m
                or stamp-self.last_fix < p.gnss_min_dt_s):
            return
        same_path = (self.route or e.map).active_spur is route.active_spur
        variance = (self.variance + p.sigma_u**2 * abs(travel-self.anchor)
                    if self.route else e.P[0][0])
        gain = variance / (variance + p.gnss_sigma_m**2)
        if same_path and direction == self.direction:
            predicted = self.measured + direction*(travel-self.anchor) if self.route else s
            measured = predicted + gain*(measured-predicted)
        else:
            gain = 1.0  # Averaging coordinates on different branches is invalid.
        self.variance = (1-gain)*variance if gain < 1 else p.gnss_sigma_m**2
        self.route = route
        self.anchor, self.measured, self.direction = travel, measured, direction
        self.orientation = orientation
        self.last_fix = stamp
        self.accepted += 1
