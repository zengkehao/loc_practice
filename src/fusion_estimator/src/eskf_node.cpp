#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include "std_msgs/msg/float64.hpp"


#include "fusion_estimator/eskf.hpp"

class EskfNode : public rclcpp::Node {
public:
    EskfNode() : Node("eskf_node") {
        // 参数
        imu_topic_ = declare_parameter<std::string>("imu_topic", "imu/data");
        gnss_odom_topic_ = declare_parameter<std::string>("gnss_odom_topic", "/gnss_ins/odom");
        odom_topic_ = declare_parameter<std::string>("odom_topic", "fusion/odom");
        path_topic_ = declare_parameter<std::string>("path_topic", "fusion/path");


        frame_map_ = declare_parameter<std::string>("frame_map", "map");
        frame_base_ = declare_parameter<std::string>("frame_base", "base_link");


        publish_tf_ = declare_parameter<bool>("publish_tf", true);
        publish_path_= declare_parameter<bool>("publish_path", true);


        // 预测（IMU-only）可视化开关与话题
        publish_predict_topics_   = declare_parameter<bool>("publish_predict_topics", true);
        odom_predict_topic_       = declare_parameter<std::string>("odom_predict_topic", "fusion/odom_predict");
        path_predict_topic_       = declare_parameter<std::string>("path_predict_topic", "fusion/path_predict");

        // 预测 path 发布控制
        path_predict_publish_stride_ = declare_parameter<int>("path_predict_publish_stride", 5); // 每N帧IMU发一次path
        path_predict_max_points_     = declare_parameter<int>("path_predict_max_points", 0);     // 0=不限制


        // GNSS/外部里程量测噪声（若来源是融合解，适当放大些）
        double pos_std_x = declare_parameter<double>("meas_pos_std_x", 0.20);
        double pos_std_y = declare_parameter<double>("meas_pos_std_y", 0.20);
        double pos_std_z = declare_parameter<double>("meas_pos_std_z", 0.40);
        Rpos_.setZero();
        Rpos_(0,0) = pos_std_x*pos_std_x;
        Rpos_(1,1) = pos_std_y*pos_std_y;
        Rpos_(2,2) = pos_std_z*pos_std_z;


        use_yaw_update_ = declare_parameter<bool>("use_yaw_update", true);
        yaw_var_default_ = declare_parameter<double>("yaw_var_default", std::pow(5.0*M_PI/180.0, 2));


        // 发布器
        odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(odom_topic_, 10);
        path_pub_ = create_publisher<nav_msgs::msg::Path>(path_topic_, 10);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);


        if (publish_predict_topics_) {
        odom_predict_pub_ = create_publisher<nav_msgs::msg::Odometry>(odom_predict_topic_, 10);

        // Path 用 transient_local，让 RViz 晚加入也能拿到最新完整Path
        rclcpp::QoS path_qos(rclcpp::KeepLast(1));
        path_qos.transient_local();
        path_predict_pub_ = create_publisher<nav_msgs::msg::Path>(path_predict_topic_, path_qos);

        path_predict_msg_.header.frame_id = frame_map_;
        }


        // 订阅器
        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_, rclcpp::SensorDataQoS(),
        std::bind(&EskfNode::ImuCallback, this, std::placeholders::_1));


        gnss_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        gnss_odom_topic_, rclcpp::QoS(10),
        std::bind(&EskfNode::GnssOdomCallback, this, std::placeholders::_1));


        // 初始化 path header
        path_msg_.header.frame_id = frame_map_;

        // 放大系数、下界、降采样、NIS 开关
        scale_pos_R_      = declare_parameter<double>("meas_pos_scale", 4.0);   // 放大位置协方差倍数
        floor_pos_xy_var_ = declare_parameter<double>("meas_pos_floor_xy", 0.04); // m^2 (≥0.2m std)^2
        floor_pos_z_var_  = declare_parameter<double>("meas_pos_floor_z",  0.25); // m^2 (≥0.5m std)^2
        subsample_N_pos_  = declare_parameter<int>("meas_pos_subsample_N", 60);    // 每 N 帧更新一次

        scale_yaw_var_    = declare_parameter<double>("yaw_var_scale", 4.0);
        floor_yaw_var_    = declare_parameter<double>("yaw_var_floor", std::pow(5.0*M_PI/180.0, 2));
        subsample_N_yaw_  = declare_parameter<int>("yaw_subsample_N", 5);

        //publish_predict_topics_ = declare_parameter<bool>("publish_predict_topics", true);
        publish_corrected_immediately_ = declare_parameter<bool>("publish_corrected_immediately", true);

        debug_publish_nis_ = declare_parameter<bool>("debug_publish_nis", true);


        RCLCPP_INFO(get_logger(), "ESKF node up. imu=%s, gnss_odom=%s → odom=%s",
        imu_topic_.c_str(), gnss_odom_topic_.c_str(), odom_topic_.c_str());

    }

private:
    void ImuCallback(const sensor_msgs::msg::Imu::SharedPtr m) {
        const rclcpp::Time stamp = m->header.stamp;
        if (!last_imu_stamp_.nanoseconds()) { last_imu_stamp_ = stamp; return; }
        double dt = (stamp - last_imu_stamp_).seconds();
        last_imu_stamp_ = stamp;
        if (!(dt > 0.0) || dt > 0.2) return; // 防抖


        // specific force (m/s^2) & angular rate (rad/s)
        Eigen::Vector3d f(m->linear_acceleration.x, m->linear_acceleration.y, m->linear_acceleration.z);
        Eigen::Vector3d w(m->angular_velocity.x, m->angular_velocity.y, m->angular_velocity.z);


        eskf_.Predict(f, w, dt);

        // 先发布“预测态”（此时尚未被GNSS回调纠正）
        if (publish_predict_topics_) {
        PublishPredictOutputs(m->header.stamp);
        }

        PublishOutputs(stamp);
        //RCLCPP_INFO("imuCallback");
    }

    void GnssOdomCallback(const nav_msgs::msg::Odometry::SharedPtr m) {
        // 1) 降采样：每 N 帧做一次位置更新
        if (++gnss_odom_counter_ % std::max(1, subsample_N_pos_) != 0) return;

        // 位置量测
        Eigen::Vector3d pz(m->pose.pose.position.x,
        m->pose.pose.position.y,
        m->pose.pose.position.z);


        // 若对方提供协方差，则覆盖默认 Rpos_
        const auto &C = m->pose.covariance; // row-major 6x6

        auto vxx = (std::isfinite(C[0])  && C[0]  > 0) ? C[0]  : floor_pos_xy_var_;
        auto vyy = (std::isfinite(C[7])  && C[7]  > 0) ? C[7]  : floor_pos_xy_var_;
        auto vzz = (std::isfinite(C[14]) && C[14] > 0) ? C[14] : floor_pos_z_var_;
          Eigen::Matrix3d Rpos = Eigen::Matrix3d::Zero();
        Rpos(0,0) = std::max(floor_pos_xy_var_, vxx) * scale_pos_R_;
        Rpos(1,1) = std::max(floor_pos_xy_var_, vyy) * scale_pos_R_;
        Rpos(2,2) = std::max(floor_pos_z_var_,  vzz) * scale_pos_R_;
        
        // Eigen::Matrix3d Rpos = Rpos_;
        // if (std::isfinite(C[0]) && C[0] > 0 && std::isfinite(C[7]) && C[7] > 0 && std::isfinite(C[14]) && C[14] > 0) {
        // Rpos(0,0) = C[0]; // var(x)
        // Rpos(1,1) = C[7]; // var(y)
        // Rpos(2,2) = C[14]; // var(z)
        // }
        eskf_.UpdatePos(pz, Rpos);


        // 可选：用 odom 的 yaw 约束（roll/pitch 不建议）
        if (use_yaw_update_) {
        const auto &q = m->pose.pose.orientation;
        const double qw = q.w, qx = q.x, qy = q.y, qz = q.z;
        const double yaw = std::atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz));
        double var_yaw = yaw_var_default_;
        if (std::isfinite(C[35]) && C[35] > 0) var_yaw = C[35]; // pose.cov yaw 分量
        eskf_.UpdateYaw(yaw, var_yaw);
        }

          // 7) 量测后立即发布一次“修正后”的轨迹（可选）
        if (publish_corrected_immediately_) {
            PublishOutputs(m->header.stamp);
        }
    }

    void PublishOutputs(const rclcpp::Time& stamp) {
        const auto &X = eskf_.X();


        nav_msgs::msg::Odometry od;
        od.header.stamp = stamp;
        od.header.frame_id = frame_map_;
        od.child_frame_id = frame_base_;
        od.pose.pose.position.x = X.p.x();
        od.pose.pose.position.y = X.p.y();
        od.pose.pose.position.z = X.p.z();
        od.pose.pose.orientation.x = X.q.x();
        od.pose.pose.orientation.y = X.q.y();
        od.pose.pose.orientation.z = X.q.z();
        od.pose.pose.orientation.w = X.q.w();
        od.twist.twist.linear.x = X.v.x();
        od.twist.twist.linear.y = X.v.y();
        od.twist.twist.linear.z = X.v.z();
        odom_pub_->publish(od);


        if (publish_tf_) {
        geometry_msgs::msg::TransformStamped tf;
        tf.header = od.header;
        tf.child_frame_id = frame_base_;
        tf.transform.translation.x = X.p.x();
        tf.transform.translation.y = X.p.y();
        tf.transform.translation.z = X.p.z();
        tf.transform.rotation = od.pose.pose.orientation;
        tf_broadcaster_->sendTransform(tf);
        }


        if (publish_path_) {
        if (path_msg_.header.frame_id.empty()) path_msg_.header.frame_id = frame_map_;
        path_msg_.header.stamp = stamp;
        geometry_msgs::msg::PoseStamped ps;
        ps.header = od.header;
        ps.pose = od.pose.pose;
        path_msg_.poses.push_back(ps);
        if (path_msg_.poses.size() > 2000) {
        path_msg_.poses.erase(path_msg_.poses.begin(), path_msg_.poses.begin()+1000);
        }
        if (path_pub_->get_subscription_count() > 0) path_pub_->publish(path_msg_);
        }
    }

    void PublishPredictOutputs(const rclcpp::Time& stamp) {
        const auto& X = eskf_.X();

        // 1) 预测 Odom
        nav_msgs::msg::Odometry od;
        od.header.stamp = stamp;
        od.header.frame_id = frame_map_;
        od.child_frame_id  = frame_base_;
        od.pose.pose.position.x = X.p.x();
        od.pose.pose.position.y = X.p.y();
        od.pose.pose.position.z = X.p.z();
        od.pose.pose.orientation.x = X.q.x();
        od.pose.pose.orientation.y = X.q.y();
        od.pose.pose.orientation.z = X.q.z();
        od.pose.pose.orientation.w = X.q.w();
        od.twist.twist.linear.x = X.v.x();
        od.twist.twist.linear.y = X.v.y();
        od.twist.twist.linear.z = X.v.z();
        if (odom_predict_pub_) odom_predict_pub_->publish(od);

        // 2) 预测 Path（控频/限长）
        if (path_predict_pub_) {
            path_predict_msg_.header.frame_id = frame_map_;
            path_predict_msg_.header.stamp = stamp;

            geometry_msgs::msg::PoseStamped ps;
            ps.header = od.header;
            ps.pose   = od.pose.pose;
            path_predict_msg_.poses.push_back(ps);

            if (path_predict_max_points_ > 0 &&
                path_predict_msg_.poses.size() > static_cast<size_t>(path_predict_max_points_)) {
            size_t drop = path_predict_msg_.poses.size() - static_cast<size_t>(path_predict_max_points_);
            path_predict_msg_.poses.erase(path_predict_msg_.poses.begin(),
                                            path_predict_msg_.poses.begin() + drop);
            }

            if ((++path_predict_counter_ % std::max(1, path_predict_publish_stride_)) == 0) {
            path_predict_pub_->publish(path_predict_msg_);
            }
        }
    }


    // --- members ---
    ESKF eskf_;
    Eigen::Matrix3d Rpos_ = Eigen::Matrix3d::Identity();
    bool use_yaw_update_ = true;
    double yaw_var_default_ = std::pow(5.0*M_PI/180.0, 2);


    std::string imu_topic_, gnss_odom_topic_, odom_topic_, path_topic_, frame_map_, frame_base_;
    bool publish_tf_{true}, publish_path_{true};


    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr gnss_odom_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    nav_msgs::msg::Path path_msg_;


    rclcpp::Time last_imu_stamp_{};


    double scale_pos_R_{4.0}, floor_pos_xy_var_{0.04}, floor_pos_z_var_{0.25};
    int    subsample_N_pos_{60};

    double scale_yaw_var_{4.0}, floor_yaw_var_{std::pow(5.0*M_PI/180.0,2)};
    int    subsample_N_yaw_{5};

    bool   publish_predict_topics_{true};
    bool   publish_corrected_immediately_{true};
    bool   debug_publish_nis_{true};

    size_t gnss_odom_counter_{0};
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr nis_pos_pub_;

    // 预测（IMU-only）发布器与缓存
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_predict_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr     path_predict_pub_;
    nav_msgs::msg::Path path_predict_msg_;
    int path_predict_publish_stride_{5};
    int path_predict_max_points_{0};
    size_t path_predict_counter_{0};

    // 参数
    std::string odom_predict_topic_, path_predict_topic_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EskfNode>());
    rclcpp::shutdown();
    return 0;
}