#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>
#include <cmath>
#include <optional>

#include <chcnv_cgi_msgs/msg/hcinspvatzcb.hpp>

using MsgT = chcnv_cgi_msgs::msg::Hcinspvatzcb;

class ImuDirectNode : public rclcpp::Node {
public:
  ImuDirectNode() : Node("imu_direct_node") {
    // ------------ 参数 ------------
    frame_map_       = declare_parameter<std::string>("frame_map", "map");
    frame_base_      = declare_parameter<std::string>("frame_base", "base_link");
    odom_topic_      = declare_parameter<std::string>("odom_topic", "odom");
    path_topic_      = declare_parameter<std::string>("path_topic", "/imu/path_predict");
    use_header_stamp_= declare_parameter<bool>("use_header_stamp", true);
    publish_path_    = declare_parameter<bool>("publish_path", true);
    publish_tf_      = declare_parameter<bool>("publish_tf", false);

    gyro_in_deg_     = declare_parameter<bool>("gyro_in_deg", true);     // raw_angular_velocity 单位
    acc_in_g_        = declare_parameter<bool>("acc_in_g",  true);       // vehicle_linear_acceleration_without_g 单位
    euler_in_deg_    = declare_parameter<bool>("euler_in_deg", true);    // roll/pitch/yaw 单位
    gravity_         = declare_parameter<double>("gravity", 9.80665);
    max_dt_          = declare_parameter<double>("max_dt", 0.2);         // 防大间隔爆炸
    min_dt_          = declare_parameter<double>("min_dt", 1e-4);        // 防除零

    // 初始速度/姿态 可参数覆盖
    init_speed_xy_mps_= declare_parameter<double>("init_speed_xy_mps", 0.0);

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(odom_topic_, 10);
    if (publish_path_) path_pub_ = create_publisher<nav_msgs::msg::Path>(path_topic_, 10);
    if (publish_tf_) tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    sub_ = create_subscription<MsgT>(
      "/chcnav/devpvt", rclcpp::SensorDataQoS(),
      std::bind(&ImuDirectNode::onMsg, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "imu_direct_node started. frame: %s -> %s",
                frame_map_.c_str(), frame_base_.c_str());
  }

private:
  // --- WGS84 常量 ---
  static constexpr double a_ = 6378137.0;              // semi-major axis
  static constexpr double f_ = 1.0 / 298.257223563;    // flattening
  static constexpr double b_ = a_ * (1.0 - f_);
  static constexpr double e2_ = (a_*a_ - b_*b_)/(a_*a_);

  struct Vec3 { double x{0}, y{0}, z{0}; };
  struct LLA  { double lat{0}, lon{0}, alt{0}; }; // radians, radians, meters
  struct ECEF { double x{0}, y{0}, z{0}; };

  // ----------------------- 回调 -----------------------
  void onMsg(const MsgT::SharedPtr msg) {
    // 时间戳
    rclcpp::Time stamp = use_header_stamp_ ? rclcpp::Time(msg->header.stamp) : now();
    if (!last_time_.has_value()) {
      // 初始化：姿态、位置原点（ENU）、速度
      initFromMsg(*msg);
      last_time_ = stamp;
      publishAll(stamp);
      return;
    }

    double dt = (stamp - *last_time_).seconds();
    if (dt < min_dt_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "dt too small (%.6f s), skip.", dt);
      return;
    }
    if (dt > max_dt_) {
      RCLCPP_WARN(get_logger(), "Large dt=%.3f s, clamping to %.3f s.", dt, max_dt_);
      dt = max_dt_;
    }
    last_time_ = stamp;

    // ========== 1) 角速度 -> 姿态积分 ==========
    // 使用 raw_angular_velocity（你也可以切到 vehicle_angular_velocity）
    Vec3 gyro{
      static_cast<double>(msg->raw_angular_velocity.x),
      static_cast<double>(msg->raw_angular_velocity.y),
      static_cast<double>(msg->raw_angular_velocity.z)
    };
    if (gyro_in_deg_) {
      gyro.x = gyro.x * M_PI / 180.0;
      gyro.y = gyro.y * M_PI / 180.0;
      gyro.z = gyro.z * M_PI / 180.0;
    }
    integrateAttitude(gyro, dt);

    // ========== 2) 线加速度(去重力，机体系) -> 速度/位置 ==========
    Vec3 acc_body_gfree{
      static_cast<double>(msg->vehicle_linear_acceleration_without_g.x),
      static_cast<double>(msg->vehicle_linear_acceleration_without_g.y),
      static_cast<double>(msg->vehicle_linear_acceleration_without_g.z)
    };
    // 单位：g -> m/s^2
    if (acc_in_g_) {
      acc_body_gfree.x *= gravity_;
      acc_body_gfree.y *= gravity_;
      acc_body_gfree.z *= gravity_;
    }
    // 机体 -> ENU
    Vec3 acc_enu = rotateBodyToENU(acc_body_gfree, q_wb_);

    // 速度、位置积分
    vel_.x += acc_enu.x * dt;
    vel_.y += acc_enu.y * dt;
    vel_.z += acc_enu.z * dt;

    pos_.x += vel_.x * dt;
    pos_.y += vel_.y * dt;
    pos_.z += vel_.z * dt;

    // 发布
    publishAll(stamp);
  }

  void initFromMsg(const MsgT& m) {
    // 姿态初始化（roll/pitch/yaw）
    double roll  = static_cast<double>(m.roll);
    double pitch = static_cast<double>(m.pitch);
    double yaw   = static_cast<double>(m.yaw);
    if (euler_in_deg_) {
      roll  *= M_PI/180.0;
      pitch *= M_PI/180.0;
      yaw   *= M_PI/180.0;
    }
    tf2::Quaternion q;
    q.setRPY(roll, pitch, yaw);
    q.normalize();
    q_wb_ = q;

    // 速度初始化（可根据消息 speed/ENU 速度，也可置零）
    vel_.x = 0.0;
    vel_.y = 0.0;
    vel_.z = 0.0;

    // ENU 原点：第一帧 LLA
    LLA lla0;
    lla0.lat = static_cast<double>(m.latitude)  * M_PI/180.0;
    lla0.lon = static_cast<double>(m.longitude) * M_PI/180.0;
    lla0.alt = static_cast<double>(m.altitude);
    ecef_ref_ = llaToEcef(lla0);
    lla_ref_  = lla0;

    // 初始平移 0
    pos_ = {0,0,0};

    path_msg_.header.frame_id = frame_map_;
    path_msg_.poses.clear();

    RCLCPP_INFO(get_logger(),
      "Initialized with RPY(deg)= [%.3f, %.3f, %.3f], ENU origin = (lat=%.8f, lon=%.8f, alt=%.3f)",
      roll * 180.0/M_PI, pitch * 180.0/M_PI, yaw * 180.0/M_PI,
      m.latitude, m.longitude, m.altitude);
  }

  // 姿态积分（一步指数映射，等价小角度四元数法）
  void integrateAttitude(const Vec3& gyro_rad, double dt) {
    const double wx = gyro_rad.x, wy = gyro_rad.y, wz = gyro_rad.z;
    const double theta = std::sqrt(wx*wx + wy*wy + wz*wz) * dt;
    tf2::Quaternion dq;
    if (theta < 1e-8) {
      // 小角度
      dq.setValue(0.5*wx*dt, 0.5*wy*dt, 0.5*wz*dt, 1.0);
    } else {
      const double ax = wx / (std::sqrt(wx*wx + wy*wy + wz*wz));
      const double ay = wy / (std::sqrt(wx*wx + wy*wy + wz*wz));
      const double az = wz / (std::sqrt(wx*wx + wy*wy + wz*wz));
      const double half = 0.5*theta;
      dq.setX(ax * std::sin(half));
      dq.setY(ay * std::sin(half));
      dq.setZ(az * std::sin(half));
      dq.setW(std::cos(half));
    }
    q_wb_ = dq * q_wb_;
    q_wb_.normalize();
  }

  // 机体到 ENU 的旋转
  Vec3 rotateBodyToENU(const Vec3& v_b, const tf2::Quaternion& q_wb) const {
    tf2::Quaternion q = q_wb;
    tf2::Matrix3x3 R(q);
    Vec3 v_w;
    v_w.x = R[0][0]*v_b.x + R[0][1]*v_b.y + R[0][2]*v_b.z;
    v_w.y = R[1][0]*v_b.x + R[1][1]*v_b.y + R[1][2]*v_b.z;
    v_w.z = R[2][0]*v_b.x + R[2][1]*v_b.y + R[2][2]*v_b.z;
    return v_w;
  }

  // LLA/ECEF/ENU 转换（WGS84）
  static ECEF llaToEcef(const LLA& lla) {
    const double sin_lat = std::sin(lla.lat);
    const double cos_lat = std::cos(lla.lat);
    const double sin_lon = std::sin(lla.lon);
    const double cos_lon = std::cos(lla.lon);
    const double N = a_ / std::sqrt(1.0 - e2_*sin_lat*sin_lat);
    ECEF e;
    e.x = (N + lla.alt) * cos_lat * cos_lon;
    e.y = (N + lla.alt) * cos_lat * sin_lon;
    e.z = (N*(1 - e2_) + lla.alt) * sin_lat;
    return e;
  }

  static Vec3 ecefToEnu(const ECEF& ecef, const LLA& ref, const ECEF& ecef_ref) {
    const double sin_lat = std::sin(ref.lat);
    const double cos_lat = std::cos(ref.lat);
    const double sin_lon = std::sin(ref.lon);
    const double cos_lon = std::cos(ref.lon);

    const double dx = ecef.x - ecef_ref.x;
    const double dy = ecef.y - ecef_ref.y;
    const double dz = ecef.z - ecef_ref.z;

    Vec3 enu;
    enu.x = -sin_lon*dx + cos_lon*dy;                         // East
    enu.y = -sin_lat*cos_lon*dx - sin_lat*sin_lon*dy + cos_lat*dz; // North
    enu.z =  cos_lat*cos_lon*dx + cos_lat*sin_lon*dy + sin_lat*dz; // Up
    return enu;
  }

  // 发布 Odom、TF、Path
  void publishAll(const rclcpp::Time& stamp) {
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = frame_map_;
    odom.child_frame_id  = frame_base_;
    odom.pose.pose.position.x = pos_.x;
    odom.pose.pose.position.y = pos_.y;
    odom.pose.pose.position.z = pos_.z;
    odom.pose.pose.orientation = tf2::toMsg(q_wb_);
    odom.twist.twist.linear.x = vel_.x;
    odom.twist.twist.linear.y = vel_.y;
    odom.twist.twist.linear.z = vel_.z;
    odom_pub_->publish(odom);

    if (publish_tf_) {
      geometry_msgs::msg::TransformStamped tf;
      tf.header.stamp = stamp;
      tf.header.frame_id = frame_map_;
      tf.child_frame_id  = frame_base_;
      tf.transform.translation.x = pos_.x;
      tf.transform.translation.y = pos_.y;
      tf.transform.translation.z = pos_.z;
      tf.transform.rotation = tf2::toMsg(q_wb_);
      tf_broadcaster_->sendTransform(tf);
    }

    if (publish_path_) {
      if ((path_count_++ % path_stride_) == 0) {
      geometry_msgs::msg::PoseStamped p;
      p.header.stamp = stamp;
      p.header.frame_id = frame_map_;
      p.pose.position.x = pos_.x;
      p.pose.position.y = pos_.y;
      p.pose.position.z = pos_.z;
      p.pose.orientation = tf2::toMsg(q_wb_);
      path_msg_.header.stamp = stamp;
      path_msg_.header.frame_id = frame_map_;
      path_msg_.poses.push_back(p);
      path_pub_->publish(path_msg_);
      }
    }

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
      "a_dev=[%.3f %.3f %.3f] m/s^2, v=[%.3f %.3f %.3f] m/s, p=[%.3f %.3f %.3f] m",
      last_acc_enu_.x, last_acc_enu_.y, last_acc_enu_.z,
      vel_.x, vel_.y, vel_.z, pos_.x, pos_.y, pos_.z);
  }

private:
  // pubs/subs
  rclcpp::Subscription<MsgT>::SharedPtr sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // params
  std::string frame_map_, frame_base_, odom_topic_, path_topic_;
  bool use_header_stamp_{true}, publish_path_{true}, publish_tf_{true};
  bool gyro_in_deg_{true}, acc_in_g_{true}, euler_in_deg_{true};
  double gravity_{9.80665}, max_dt_{0.2}, min_dt_{1e-4}, init_speed_xy_mps_{0.0};
  int path_stride_ = declare_parameter<int>("path_stride", 1); // 每隔 N 个点入 Path
  int path_count_ = 0;

  // state
  tf2::Quaternion q_wb_;  // world(ENU) <- body
  Vec3 pos_{0,0,0};
  Vec3 vel_{0,0,0};
  Vec3 last_acc_enu_{0,0,0};
  std::optional<rclcpp::Time> last_time_;

  // ENU 原点
  LLA  lla_ref_{};
  ECEF ecef_ref_{};

  nav_msgs::msg::Path path_msg_;
};
  
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImuDirectNode>());
  rclcpp::shutdown();
  return 0;
}
