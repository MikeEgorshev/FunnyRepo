"""Hybrid competition entry point."""
import rclpy
from .integrated_node import TramOdometryNode as IntegratedNode
from .hybrid import HybridEstimator
from .integrated_model import IntegratedModel


class HybridOdometryNode(IntegratedNode):
    estimator_type = HybridEstimator
    model_type = IntegratedModel


def main(args=None):
    rclpy.init(args=args)
    node = HybridOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
