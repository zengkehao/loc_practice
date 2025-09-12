#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.h>
#include <Eigen/Dense>
#include <cmath>
#include "chcnv_cgi_msgs/msg/hcinspvatzcb.hpp"
#include <nav_msgs/msg/path.hpp>
#include <algorithm> 

using Msg = chcnv_cgi_msgs::msg::Hcinspvatzcb;
using std::placeholders::_1;

//创建向量的反对称矩阵（用于李代数运算）
static inline Eigen::Matrix3d Skew(const Eigen::Vector3d& v){
  Eigen::Matrix3d S;
  S << 0,-v.z(),v.y(), v.z(),0,-v.x(), -v.y(),v.x(),0; return S;
}

// SO(3) 旋转向量转旋转矩阵
static inline Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& w){
  double th = w.norm(); Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  if (th < 1e-9) return I + Skew(w);
  Eigen::Matrix3d K = Skew(w/th);
  return I + std::sin(th)*K + (1.0-std::cos(th))*K*K;
}
// 将角度归一化到 [-pi, pi]
static inline double wrapToPi(double a){
  while (a >  M_PI) { a -= 2.0*M_PI; }
  while (a < -M_PI) { a += 2.0*M_PI; }
  return a;
}
// 从旋转矩阵中提取 yaw（航向角，rad）
static inline double yawFromR(const Eigen::Matrix3d& R){
  return std::atan2(R(1,0), R(0,0)); // ZYX
}
//将geometry_msgs的Vector3转换为Eigen::Vector3d
static inline Eigen::Vector3d v3(const geometry_msgs::msg::Vector3& v){
  return Eigen::Vector3d(v.x, v.y, v.z);
}
// Z-X-Y 顺序欧拉角（deg）构造旋转：R = Rz(z)*Rx(x)*Ry(y)
static inline Eigen::Matrix3d R_from_ZXY_deg(double z_deg, double x_deg, double y_deg){
  double z = z_deg * M_PI/180.0;
  double x = x_deg * M_PI/180.0;
  double y = y_deg * M_PI/180.0;
  Eigen::AngleAxisd Rz(z, Eigen::Vector3d::UnitZ());
  Eigen::AngleAxisd Rx(x, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd Ry(y, Eigen::Vector3d::UnitY());
  return (Rz * Rx * Ry).toRotationMatrix();
}
// 将近似正交矩阵正交化（SVD 法），确保落回 SO(3)
static inline Eigen::Matrix3d Orthonormalize(const Eigen::Matrix3d& R){
  // 使 R 落回 SO(3)
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();
  Eigen::Matrix3d R_ortho = U * V.transpose();
  if (R_ortho.determinant() < 0) {        // 反射，翻转一列修正
    U.col(2) *= -1.0;
    R_ortho = U * V.transpose();
  }
  return R_ortho;
}

class ImuDirectNode : public rclcpp::Node {
public:
  ImuDirectNode() : Node("imu_direct_node"){
    // params
    frame_id_       = declare_parameter<std::string>("frame_id", "odom");
    child_frame_id_ = declare_parameter<std::string>("child_frame_id", "base_link");
    publish_tf_     = declare_parameter<bool>("publish_tf", true);
    use_yaw_obs_    = declare_parameter<bool>("use_yaw_obs", false); // 用双天线航向作弱观测
    yaw_alpha_      = declare_parameter<double>("yaw_obs_alpha", 0.05);
    init_secs_      = declare_parameter<double>("init_duration_sec", 3.0);
    max_dt_         = declare_parameter<double>("max_dt_sec", 0.3);
    min_dt_         = declare_parameter<double>("min_dt_sec", 1e-5);
    apply_mount_    = declare_parameter<bool>("apply_mount_extrinsic", true);
    auto g_list     = declare_parameter<std::vector<double>>("gravity_vec", {0.0,0.0,-9.81});
    drop_on_warning_     = declare_parameter<bool>("drop_on_warning", false);
    warning_drop_mask_   = declare_parameter<int>("warning_drop_mask", 0);

    acc_unit_is_g_        = declare_parameter<bool>("acc_unit_is_g", true);
    acc_includes_gravity_ = declare_parameter<bool>("acc_includes_gravity", false);
    use_linear_acc_wo_g_  = declare_parameter<bool>("use_linear_acc_wo_g", true);

    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("imu_propagation/path", 10);
    path_msg_.header.frame_id = frame_id_;  // 与 odom 的 frame_id 一致
    g_ = {g_list[0], g_list[1], g_list[2]};

    pub_ = create_publisher<nav_msgs::msg::Odometry>("imu_propagation/odom", 50);
    if (publish_tf_) tf_brd_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    sub_ = create_subscription<Msg>("/chcnav/devpvt", rclcpp::SensorDataQoS(),
           std::bind(&ImuDirectNode::callBack, this, _1));

    reset();
    RCLCPP_INFO(get_logger(), "IMU direct integration from /chcnav/devpvt");
  }

private:
  // state
  Eigen::Matrix3d R_{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d v_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d p_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d bg_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d ba_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d g_{0,0,-9.81};

  bool inited_{false}, have_prev_{false};
  double t0_{NAN}, t_prev_{0.0};
  Eigen::Vector3d w_prev_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d a_prev_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d gyro_sum_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d acc_sum_{Eigen::Vector3d::Zero()};
  size_t cnt_{0};

  bool drop_on_warning_{false};
  uint32_t warning_drop_mask_{0}; 

  bool acc_unit_is_g_{true};              // 原始加计是否用 g 作为单位
  bool acc_includes_gravity_{true};       // 原始加计是否含重力（specific force）
  bool use_linear_acc_wo_g_{false};       // 是否直接用“已去重力”的线加速度

  // params & ROS
  std::string frame_id_, child_frame_id_;
  bool publish_tf_{false}, use_yaw_obs_{true}, apply_mount_{true};
  double yaw_alpha_{0.05}, init_secs_{3.0}, max_dt_{0.3}, min_dt_{1e-5};
  rclcpp::Subscription<Msg>::SharedPtr sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_brd_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  nav_msgs::msg::Path path_msg_;
  int path_stride_ = 3;   // 每 N 帧采一点，防止太密
  int path_count_  = 0;

  void reset(){
    R_.setIdentity(); v_.setZero(); p_.setZero(); bg_.setZero(); ba_.setZero();
    inited_=false; have_prev_=false; t0_=NAN; t_prev_=0; gyro_sum_.setZero(); acc_sum_.setZero(); cnt_=0;
  }

  void callBack(const Msg::ConstSharedPtr m){
    // 时间：优先 header
    double t = rclcpp::Time(m->header.stamp).seconds();
    if (!std::isfinite(t)) return;

    // 原始 IMU（设备系）
    Eigen::Vector3d w_dev = v3(m->raw_angular_velocity) * M_PI/180.0;
    Eigen::Vector3d a_dev = v3(m->vehicle_linear_acceleration_without_g) * 9.80665;
    if (use_linear_acc_wo_g_) {
    // 直接使用“已去重力”的线加速度
    a_dev = v3(m->vehicle_linear_acceleration_without_g);
    // 单位：你的数据看起来像是 g；若是 g，就换算到 m/s^2
    if (acc_unit_is_g_) a_dev *= 9.80665;
    // 后面世界系加速度就不要再 +g 了（见下方）
    } else {
    // 使用原始加计（含重力）
    a_dev = v3(m->raw_acceleration);
    // 单位换算：g -> m/s^2
    if (acc_unit_is_g_) a_dev *= 9.80665;
    }

    // ★ 新增：数值投影回 SO(3)
    R_ = Orthonormalize(R_);

    // 设备系→车辆系（Z-X-Y 顺序安装角：车辆→设备）
    Eigen::Vector3d w = w_dev, a = a_dev;
    if (apply_mount_){
      Eigen::Matrix3d R_vehicle_to_device = R_from_ZXY_deg(
          m->ins2body_angle.z, m->ins2body_angle.x, m->ins2body_angle.y);
      Eigen::Matrix3d R_device_to_vehicle = R_vehicle_to_device.transpose();
      w = R_device_to_vehicle * w_dev;
      a = R_device_to_vehicle * a_dev;
    }

    // warning 处理
    uint32_t warn = static_cast<uint32_t>(m->warning);

    if (warn != 0){
    if (drop_on_warning_ && (warn & warning_drop_mask_)){
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "warning=0x%X, dropped by mask=0x%X.", warn, warning_drop_mask_);
        //return;
    } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "warning=0x%X, continuing (mask=0x%X, drop=%s).",
        warn, warning_drop_mask_, drop_on_warning_ ? "true":"false");
    }
    }

    // // 可用 warning 作质量控制（非零时丢弃/降权）
    // if (m->warning != 0){
    //   RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
    //                        "warning=0x%X, sample ignored.", (unsigned)m->warning);
    //   return;
    // }

    if (!w.allFinite() || !a.allFinite()) return;

    // 初始化：静置估偏 + 用加计对齐重力
    if (!inited_){
      if (!std::isfinite(t0_)) t0_ = t;
      gyro_sum_ += w; acc_sum_ += a; cnt_++;
      if (t - t0_ < init_secs_) return;

      Eigen::Vector3d gyro_mean = gyro_sum_/double(cnt_);
      Eigen::Vector3d acc_mean  = acc_sum_ /double(cnt_);
      bg_ = gyro_mean;
      // 对齐：R * (a - ba) ≈ -g，先以 ba=0 求 R
      Eigen::Vector3d zb = acc_mean.normalized();   // body里“重力方向”(+9.81朝上)
      Eigen::Vector3d zw = (-g_).normalized();      // 世界“重力方向”
      Eigen::Vector3d v = zb.cross(zw); double s=v.norm(), c=zb.dot(zw);
      Eigen::Matrix3d R_align = Eigen::Matrix3d::Identity();
      if (s < 1e-8) { if (c < 0){ R_align = ExpSO3(M_PI*Eigen::Vector3d::UnitX()); } }
      else{
        Eigen::Matrix3d K = Skew(v/s);
        R_align = Eigen::Matrix3d::Identity() + K + K*K*((1.0-c)/(s*s));
      }
      R_ = R_align;
      // 粗估 ba
      ba_ = acc_mean - R_.transpose() * (-g_);
      inited_ = true;
      RCLCPP_INFO(get_logger(),
        "Bias init done. bg=[%.4f %.4f %.4f], ba=[%.4f %.4f %.4f]",
        bg_.x(),bg_.y(),bg_.z(), ba_.x(),ba_.y(),ba_.z());
      return; // 下一帧开始积分
    }

    // 积分
    if (!have_prev_){ t_prev_=t; w_prev_=w; a_prev_=a; have_prev_=true; return; }
    double dt = t - t_prev_;
    if (dt < min_dt_ || dt > max_dt_){
      t_prev_=t; w_prev_=w; a_prev_=a;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "abnormal dt=%.6f", dt);
      return;
    }

    // 1) 去偏置
    Eigen::Vector3d w0 = w_prev_ - bg_, w1 = w - bg_;
    Eigen::Vector3d a0 = a_prev_ - ba_, a1 = a - ba_;

    // 2) 姿态（中值角速）
    Eigen::Vector3d w_mid = 0.5*(w0+w1);
    Eigen::Matrix3d R1 = R_ * ExpSO3(w_mid*dt);

    // 2.5) 航向弱观测（双天线 yaw，deg）
    if (use_yaw_obs_){
      double yaw_meas = m->yaw * M_PI/180.0; // [-180,180] deg
      double yaw_cur  = yawFromR(R1);
      double dyaw     = wrapToPi(yaw_meas - yaw_cur);
      double a = std::clamp(yaw_alpha_, 0.0, 1.0);
      Eigen::Matrix3d Rz = ExpSO3(Eigen::Vector3d(0,0,a*dyaw));
      R1 = Rz * R1;
      //再正交化一次
      R1 = Orthonormalize(R1);
    }

    // 3) 加速度到世界 + g（中值）
    Eigen::Vector3d aw0 = R_  * a0;
    Eigen::Vector3d aw1 = R1 * a1;
    Eigen::Vector3d a_bar = 0.5*(aw0+aw1);

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
    "a_dev=[%.3f %.3f %.3f] m/s^2, a_bar.z=%.3f, p.z=%.3f",
    a_dev.x(), a_dev.y(), a_dev.z(), a_bar.z(), p_.z());

    if (!use_linear_acc_wo_g_ && acc_includes_gravity_) {
    a_bar += g_;   // 比力 → 加 g，得到世界系线加速度
    }

    // 4) v/p
    p_ += v_*dt + 0.5*a_bar*dt*dt;
    v_ += a_bar*dt;
    R_  = R1;

    publishOdom(m->header.stamp);
    t_prev_=t; w_prev_=w; a_prev_=a;
  }

  void publishOdom(const rclcpp::Time& stamp){
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = frame_id_;
    odom.child_frame_id  = child_frame_id_;
    //Eigen::Quaterniond q(R_);

    //odom.pose.pose.orientation = tf2::toMsg(q);

    Eigen::Matrix3d Rpub = Orthonormalize(R_);
    Eigen::Quaterniond q(Rpub);
    // ★ 兜底 & 归一化
    if (!q.coeffs().allFinite() || q.squaredNorm() < 1e-12) {
    q = Eigen::Quaterniond::Identity();
    } else {
    q.normalize();
    }
    
    odom.pose.pose.position.x = p_.x();
    odom.pose.pose.position.y = p_.y();
    odom.pose.pose.position.z = p_.z();
    odom.twist.twist.linear.x = v_.x();
    odom.twist.twist.linear.y = v_.y();
    odom.twist.twist.linear.z = v_.z();
    pub_->publish(odom);

    geometry_msgs::msg::PoseStamped ps;
    ps.header = odom.header;
    ps.pose = odom.pose.pose;

    if ((path_count_++ % path_stride_) == 0) {
        path_msg_.header.stamp = odom.header.stamp;
        path_msg_.poses.push_back(ps);
        if (path_msg_.poses.size() > 5000) {            // 可选限长度
            path_msg_.poses.erase(path_msg_.poses.begin(),
                                path_msg_.poses.begin() + 1000);
        }
        path_pub_->publish(path_msg_);
    }

    if (publish_tf_){
      geometry_msgs::msg::TransformStamped tf;
      tf.header = odom.header;
      tf.child_frame_id = child_frame_id_;
      tf.transform.translation.x = p_.x();
      tf.transform.translation.y = p_.y();
      tf.transform.translation.z = p_.z();
      tf.transform.rotation = odom.pose.pose.orientation;
      tf_brd_->sendTransform(tf);
    }
  }
};

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImuDirectNode>());
  rclcpp::shutdown();
  return 0;
}
