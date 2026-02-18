import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry, Path
from math import atan2, sin, cos, sqrt, degrees, atan, pi
import time

class PIDControl:
    def __init__(self, Kp=0.35, Ki=0.02, Kd=0.11, brake_gain=0.9):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.brake_gain = brake_gain
        self.int_error_v = 0.0
        self.prev_error_v = 0.0

    def compute(self, target_speed, current_speed, dt=0.1):
        error_v = target_speed - current_speed
        der = (error_v - self.prev_error_v) / dt
        self.int_error_v += error_v * dt
        throttle = self.Kp * error_v + self.Ki * self.int_error_v + self.Kd * der
        self.prev_error_v = error_v
        
        # braking
        if target_speed <= 0.01:
            throttle = -min(1.0, max(0.1, self.brake_gain * current_speed))
        # Clamp throttle to [-1, 1]
        throttle = max(-1.0, min(1.0, float(throttle)))
        return throttle

class AdaptivePurePursuitControl:
    def __init__(self, wheelbase=2.5, max_steer=35.0):
        self.L = wheelbase
        self.max_steer = max_steer
        self.L_min = 2.0
        self.L_max = 12.0
        self.k_v = 0.3
        self.curv_gain = 4.0
        self.k_curv = 25.0
        self.v_min = 0.2
        self.v_max = 8.0
        self.v_absolute_max = 25.0
        self.goal_tolerance = 1.0

    def nearest_index(self, x, y, path):
        if not path:
            return None
        dists = [(sqrt((px - x)**2 + (py - y)**2), i) for i, (px, py) in enumerate(path)]
        _, idx = min(dists)
        return idx

    def curvature_at(self, idx, path):
        n = len(path)
        if n < 3:
            return 0.0
        i0 = max(0, idx - 1)
        i2 = min(n - 1, idx + 1)
        x0, y0 = path[i0]
        x1, y1 = path[idx]
        x2, y2 = path[i2]
        v1x, v1y = x1 - x0, y1 - y0
        v2x, v2y = x2 - x1, y2 - y1
        a1 = atan2(v1y, v1x)
        a2 = atan2(v2y, v2x)
        dtheta = abs(a2 - a1)
        if dtheta > pi:
            dtheta = 2*pi - dtheta
        ds = sqrt((x2 - x0)**2 + (y2 - y0)**2)
        return dtheta / ds if ds > 1e-6 else 0.0

    def lookahead_point(self, Ld, x, y, path):
        if not path:
            return None
        idx = self.nearest_index(x, y, path)
        if idx is None:
            return None
        cum = 0.0
        for i in range(idx, len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            seg = sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if cum + seg >= Ld:
                t = (Ld - cum) / max(seg, 1e-6)
                return x1 + t * (x2 - x1), y1 + t * (y2 - y1)
            cum += seg
        return path[-1]

    def compute(self, x, y, yaw, speed, path, gx, gy):
        idx = self.nearest_index(x, y, path)
        if idx is None:
            return 0.0, 0.0, False
        dist_goal = sqrt((gx - x)**2 + (gy - y)**2)
        curv = self.curvature_at(idx, path)

        #lookahead
        Ld = (self.L_min + self.k_v * max(0.0, speed)) / (1.0 + self.curv_gain * curv)
        Ld = max(self.L_min, min(self.L_max, Ld))
        target = self.lookahead_point(Ld, x, y, path)
        if target is None:
            return 0.0, 0.0, False
        # Target speed based on curvature
        target_speed = max(self.v_min, min(self.v_max, self.v_max / (1.0 + self.k_curv * curv)))
        target_speed = min(target_speed, self.v_absolute_max)
        # Slowing
        if dist_goal < 12.0:
            target_speed = min(target_speed, 1.5)
        if dist_goal < 5.0:
            target_speed = 0.0
        # steering
        tx, ty = target
        dx, dy = tx - x, ty - y
        local_x = cos(-yaw) * dx - sin(-yaw) * dy
        local_y = sin(-yaw) * dx + cos(-yaw) * dy
        alpha = atan2(local_y, local_x)
        steer_rad = atan((2.0 * self.L * sin(alpha)) / max(1e-6, Ld))
        steer_deg = max(-self.max_steer, min(self.max_steer, degrees(steer_rad)))
        return steer_deg, target_speed, dist_goal < self.goal_tolerance

class Controller(Node):
    def __init__(self):
        super().__init__('controller')
        # Subscribers and Publishers
        self.create_subscription(Odometry, '/state', self.state_callback, 10)
        self.create_subscription(Path, '/path', self.path_callback, 10)
        self.pub_steer = self.create_publisher(Float32, '/steer', 10)
        self.pub_throttle = self.create_publisher(Float32, '/throttle', 10)
        self.timer = self.create_timer(0.1, self.control_loop)

        # Controllers
        self.pp = AdaptivePurePursuitControl()
        self.pid = PIDControl()
        # State
        self.x = self.y = self.yaw = self.speed = 0.0
        self.path = []
        self.path_set = False
        self.goal_reached = False
        self.stop_start_time = None
        # Metrics
        self.cross_track_errors = []
        self.heading_errors = []
        self.speed_errors = []
        self.steering_rates = []
        self.prev_steer = 0.0

    #Call back function for state message
    def state_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = atan2(siny, cosy)
        self.speed = msg.twist.twist.linear.x
    # Call back function for path message
    def path_callback(self, msg):
        if not self.path_set:
            self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
            self.path_set = True
    # Metric Calculation
    def compute_metrics(self, steer_deg, target_speed, idx):
        if idx is not None and idx < len(self.path):
            px, py = self.path[idx]
            # Cross-track error
            cte = sqrt((self.x - px)**2 + (self.y - py)**2)
            self.cross_track_errors.append(cte)

            # Heading error
            if idx < len(self.path) - 1:
                dx = self.path[idx + 1][0] - px
                dy = self.path[idx + 1][1] - py
                path_yaw = atan2(dy, dx)
                heading_err = abs((path_yaw - self.yaw + pi) % (2*pi) - pi)
                self.heading_errors.append(degrees(heading_err))

        # Speed error
        speed_err = abs(target_speed - self.speed)
        self.speed_errors.append(speed_err)

        # Steering rate
        steer_rate = (steer_deg - self.prev_steer) / 0.1
        self.prev_steer = steer_deg
        self.steering_rates.append(abs(steer_rate))

        # Path completion
        path_ratio = (idx / len(self.path)) * 100 if self.path else 0.0

        self.get_logger().info(
            f"CTE={cte:.2f} m | Heading_error={self.heading_errors[-1]:.1f}° | "
            f"Speed_error={speed_err:.2f} m/s | SteerRate={steer_rate:.1f}°/s | "
            f"PathCompletion={path_ratio:.1f}%"
        )
        
    def control_loop(self):
        if not self.path_set:
            return

        gx, gy = self.path[-1]

        if self.goal_reached:
            self.hold_position()
            return

        steer_deg, target_speed, reached_goal = self.pp.compute(self.x, self.y, self.yaw, self.speed, self.path, gx, gy)

        if reached_goal:
            self.goal_reached = True
            self.stop_start_time = time.time()
            return

        idx = self.pp.nearest_index(self.x, self.y, self.path)
        self.compute_metrics(steer_deg, target_speed, idx)
        throttle = self.pid.compute(target_speed, self.speed)
        
        # Publishing
        self.pub_steer.publish(Float32(data=float(steer_deg)))
        self.pub_throttle.publish(Float32(data=float(throttle)))

    # Goal 
    def hold_position(self):
        now = time.time()
        if self.stop_start_time and (now - self.stop_start_time < 10.0):
            if self.speed > 0.1:
                brake = -min(1.0, self.pid.brake_gain * self.speed)
                self.pub_steer.publish(Float32(data=0.0))
                self.pub_throttle.publish(Float32(data=brake))
            else:
                self.pub_steer.publish(Float32(data=0.0))
                self.pub_throttle.publish(Float32(data=0.0))
        else:
            self.pub_steer.publish(Float32(data=0.0))
            self.pub_throttle.publish(Float32(data=0.0))

def main(args=None):
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
