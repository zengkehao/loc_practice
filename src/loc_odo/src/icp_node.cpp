#include <memory>
#include <chrono>
#include <string>
#include <Eigen/Dense>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>   

#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>


namespace ph = std::placeholders;

class IcpNode : public rclcpp::Node
{
public:
  IcpNode()// 构造函数创建节点
  : Node("icp_node")  
  , leaf_size_(declare_parameter<double>("voxel_leaf_size", 0.2)) // 用于体素降采样的叶大小（米）
  , max_corresp_dist_(declare_parameter<double>("max_correspondence_dist", 2.0))  // ICP 中对应点最大距离阈值
  , max_iter_(declare_parameter<int>("max_iterations", 50))  // ICP 算法最大迭代次数
  , trans_eps_(declare_parameter<double>("transformation_epsilon", 1e-6))  // 变换收敛阈值
  , euclid_fitness_eps_(declare_parameter<double>("euclidean_fitness_epsilon", 1e-6))  // 欧几里得适应度收敛阈值
  {
    world_frame_  = this->declare_parameter<std::string>("world_frame", "map");
    lidar_frame_  = this->declare_parameter<std::string>("lidar_frame", "rslidar");
    z_min_        = this->declare_parameter<double>("z_min", -1.5);
    z_max_        = this->declare_parameter<double>("z_max",  1.5);
    use_z_weight_ = this->declare_parameter<bool>  ("use_z_weight", false); // 是否启用 z 降权
    z_weight_     = this->declare_parameter<double>("z_weight", 0.10);      // z 权重 (<1 降权)


    // 订阅 bag 播放出的点云
    std::string topic = declare_parameter<std::string>("topic", "/delphin_m1p_points"); 
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(   
      topic, rclcpp::SensorDataQoS(),                               // 使用 SensorDataQoS() 服务质量策略（适用于传感器数据）
      std::bind(&IcpNode::cloudCallback, this, ph::_1));            // 绑定回调函数 cloudCallback，当收到消息时调用该函数处理

    // 发布对齐后的点云和累计位姿
    pub_aligned_ = create_publisher<sensor_msgs::msg::PointCloud2>("/icp/aligned_points", 10);  
    pub_pose_    = create_publisher<geometry_msgs::msg::PoseStamped>("/icp/pose", 10);  
    pub_path_   = this->create_publisher<nav_msgs::msg::Path>("icp_path", 10);

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    path_msg_.header.frame_id = world_frame_;  

    // 累计位姿（世界坐标）初始化为单位矩阵
    T_world_curr_.setIdentity();  

    RCLCPP_INFO(get_logger(), "ICP node started. Subscribing: %s", topic.c_str());  
  }

private:
  using CloudT = pcl::PointCloud<pcl::PointXYZ>;


    // 点云回调函数
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {

    
    // 转为 PCL 并降采样
    CloudT::Ptr curr_raw = toPclXYZ(*msg);
    if (curr_raw->empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Empty cloud, skip.");
      return;
    }

    //CloudT::Ptr curr = voxelDownsample(curr_raw, leaf_size_);
    // 过滤：高度窗口 + 体素
    CloudT::Ptr curr = filterCloud(curr_raw, static_cast<float>(leaf_size_), z_min_, z_max_); 
    // 第一帧只缓存，不做配准
    if (!prev_) {
      prev_ = curr;
      T_world_curr_.setIdentity();  // world->rslidar = I

      // 广播 TF: world_frame_ -> lidar_frame_（用bag的时间戳）
      auto tf0 = eigMatToTf(T_world_curr_, world_frame_, lidar_frame_, msg->header.stamp);
      tf_broadcaster_->sendTransform(tf0);

      // 初始化 Path，并发布初始位姿（可视化起点）
      path_msg_.header.frame_id = world_frame_;
      path_msg_.header.stamp = msg->header.stamp;
      path_msg_.poses.clear();
 
      geometry_msgs::msg::PoseStamped ps0;
      ps0.header.frame_id = world_frame_;
      ps0.header.stamp = msg->header.stamp;
      matToPoseSE2(T_world_curr_, ps0.pose);                  // 需要: Eigen 4x4 -> geometry_msgs::Pose
      path_msg_.poses.push_back(ps0);
      pub_path_->publish(path_msg_);

      return;
    }

    // z 降权：把源/目标都缩放到 z_weight_ 域内做 ICP
    CloudT::Ptr src = curr, tgt = prev_;
    float z_scale = 1.0f;
    if (use_z_weight_) {
      z_scale = static_cast<float>(z_weight_);
      src = boost::make_shared<CloudT>(*curr);
      tgt = boost::make_shared<CloudT>(*prev_);
      scaleCloudZ(src, z_scale);
      scaleCloudZ(tgt, z_scale);
    }

    // 3) ICP: source=curr, target=prev_
    Eigen::Matrix4f T_prev_curr;
    CloudT aligned;   // 对齐到 "prev_" 坐标系（即 rslidar）的点云，仅用于可视化
    try {
        // 打印输入点云大小
         RCLCPP_INFO(this->get_logger(),
              "ICP start: source pts=%zu, target pts=%zu, "
              "max_corr=%.2f, max_iter=%d, trans_eps=%.1e, fit_eps=%.1e",
              curr->size(), prev_ ? prev_->size() : 0,
              max_corresp_dist_, max_iter_,
              trans_eps_, euclid_fitness_eps_);

      // 设置 ICP 参数并执行 icp.align(aligned)
      T_prev_curr = runICP(src, tgt, &aligned,
                          max_corresp_dist_, max_iter_,
                          trans_eps_, euclid_fitness_eps_);
    } catch (const std::exception &e) {
      RCLCPP_WARN(this->get_logger(), "ICP failed: %s. Keep last pose.", e.what());

      // 失败也广播一次TF，保证RViz不报 TF 断链
      auto tf_keep = eigMatToTf(T_world_curr_, world_frame_, lidar_frame_, msg->header.stamp);
      tf_broadcaster_->sendTransform(tf_keep);

      pub_path_->publish(path_msg_);
      return;

      // //仍发布当前帧为 aligned（未对齐）
      // sensor_msgs::msg::PointCloud2 out_fail;
      // pcl::toROSMsg(*curr, out_fail);
      // out_fail.header = msg->header;
      // out_fail.header.frame_id = lidar_frame_;
      // pub_aligned_->publish(out_fail);

      // prev_ = curr;  // 滚动参考
      return;
    }

    // 若做了 z 降权：T = S^{-1} * T_scaled * S
    if (use_z_weight_) {
      Eigen::Matrix4f S = Eigen::Matrix4f::Identity();  S(2,2) = z_scale;
      Eigen::Matrix4f S_inv = Eigen::Matrix4f::Identity(); S_inv(2,2) = 1.0f / z_scale;
      T_prev_curr = S_inv * T_prev_curr * S;
    }

    // === 只保留 SE(2) ===
    projectToSE2(T_prev_curr);

    // 累计到世界位姿： T_world_curr = T_world_prev * T_prev_curr
    T_world_curr_ = T_world_curr_ * T_prev_curr;

    // 5) 广播 TF: world_frame_ -> lidar_frame_（核心：让RViz能把所有坐标系放一起）
    auto tf_msg = eigMatToTf(T_world_curr_, world_frame_, lidar_frame_, msg->header.stamp);
    tf_broadcaster_->sendTransform(tf_msg);

    RCLCPP_INFO(get_logger(), "ICP node started pub");    

    // 6) 发布 PoseStamped + Path（header.frame_id 必须是 world_frame_）
    geometry_msgs::msg::PoseStamped ps;
    ps.header.frame_id = world_frame_;
    ps.header.stamp = msg->header.stamp;
    matToPoseSE2(T_world_curr_, ps.pose);

    path_msg_.header.frame_id = world_frame_;
    path_msg_.header.stamp = ps.header.stamp;
    path_msg_.poses.push_back(ps);
    pub_path_->publish(path_msg_);

    // 7) （可选）发布对齐后的点云（仍然在 rslidar 坐标系，RViz 会用TF变到 map）
    sensor_msgs::msg::PointCloud2 out;
    pcl::toROSMsg(aligned, out);
    out.header = msg->header;
    out.header.frame_id = lidar_frame_;  // 和 bag 一致，方便TF统一管理
    pub_aligned_->publish(out);

    // 8) 滚动参考
    prev_ = curr;
  }

  //将ROS的sensor_msgs::msg::PointCloud2转换为PCL点云格式
  static CloudT::Ptr toPclXYZ(const sensor_msgs::msg::PointCloud2 & msg)
  {
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(msg, pcl_pc2);//将ROS的sensor_msgs::PointCloud2消息转换为PCL的PCLPointCloud2格式。
    CloudT::Ptr cloud(new CloudT);
    pcl::fromPCLPointCloud2(pcl_pc2, *cloud); //将PCLPointCloud2格式的数据转换为PointCloudpcl::PointXYZ格式，存储在cloud对象中。
    // 去掉 NaN
    std::vector<int> idx;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, idx);
    return cloud;
  }

  // 体素降采样
  CloudT::Ptr voxelDownsample(const CloudT::Ptr & in, double leaf)
  {
    if (leaf <= 0.0) return in;
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(in);
    vg.setLeafSize(leaf, leaf, leaf);
    CloudT::Ptr out(new CloudT);
    vg.filter(*out);
    return out;
  }



  //用于发布对齐后的点云和位姿
  void publishAlignedAndPose(const CloudT & cloud,
                             const std::string & frame_id,
                             const rclcpp::Time & stamp,
                             bool keep_pose_identity)
  {
    sensor_msgs::msg::PointCloud2 out;
    pcl::toROSMsg(cloud, out); //将PCL点云格式转换为ROS PointCloud2消息格式。
    out.header.frame_id = frame_id;
    out.header.stamp = stamp;
    pub_aligned_->publish(out);

    if (keep_pose_identity) {
      publishPose(frame_id, stamp, T_world_curr_); // 初始为 I
    }
  }

  // 将 4x4 变换矩阵转换为 ROS Pose
  static void matToPose(const Eigen::Matrix4f & T,
                        geometry_msgs::msg::Pose & pose)
  {   
    Eigen::Matrix3f R = T.block<3,3>(0,0); //从4x4变换矩阵T中提取左上角3x3的旋转子矩阵，存储在R中。
    Eigen::Quaternionf q(R);  
    q.normalize();//对四元数进行归一化处理，确保其为单位四元数。
    pose.position.x = T(0,3);
    pose.position.y = T(1,3);
    pose.position.z = T(2,3);
    pose.orientation.x = q.x();
    pose.orientation.y = q.y();
    pose.orientation.z = q.z();
    pose.orientation.w = q.w();
  }

  // 将 4x4 变换矩阵转换为 TF
  static geometry_msgs::msg::TransformStamped
  eigMatToTf(const Eigen::Matrix4f &T, const std::string &parent, const std::string &child,
           const rclcpp::Time &stamp)
  {
    Eigen::Matrix3f R = T.block<3,3>(0,0);
    Eigen::Vector3f t = T.block<3,1>(0,3);
    Eigen::Quaternionf q(R);
    q.normalize();

    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = stamp;          // 用 bag 的时间 /clock
    tf.header.frame_id = parent;      // world_frame_
    tf.child_frame_id = child;        // lidar_frame_
    tf.transform.translation.x = t.x();
    tf.transform.translation.y = t.y();
    tf.transform.translation.z = t.z();
    tf.transform.rotation.x = q.x();
    tf.transform.rotation.y = q.y();
    tf.transform.rotation.z = q.z();
    tf.transform.rotation.w = q.w();
    return tf;
  }

  // ICP 算法
  Eigen::Matrix4f runICP(const CloudT::Ptr & source,
                       const CloudT::Ptr & target,
                       CloudT *aligned_out,
                       double max_corr,
                       int max_iter,
                       double trans_eps,
                       double fit_eps)
  {
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(max_corr);
    icp.setMaximumIterations(max_iter);
    icp.setTransformationEpsilon(trans_eps);
    icp.setEuclideanFitnessEpsilon(fit_eps);

    //创建临时点云存储对齐结果
    CloudT aligned_tmp;
    icp.align(aligned_tmp);

    if (aligned_out) {
      *aligned_out = aligned_tmp;
    }

    if (!icp.hasConverged()) {
      throw std::runtime_error("ICP did not converge");
    }

    return icp.getFinalTransformation();
  }


  // 发布位姿和路径
  void publishPose(const std::string & frame_id,
                   const rclcpp::Time & stamp,
                   const Eigen::Matrix4f & T_world_curr)
  {
    geometry_msgs::msg::PoseStamped ps;
    
    ps.header.stamp = stamp;
    ps.header.frame_id = "rslidar"; // 这里简单用点云 frame，当作“世界”坐标系
    matToPoseSE2(T_world_curr, ps.pose);

    //发布单帧 pose
    pub_pose_->publish(ps);

    // 累加到 path
    path_msg_.header.stamp = stamp;
    path_msg_.poses.push_back(ps);
    pub_path_->publish(path_msg_);
  }

  // 3.1 高度窗口 + 体素下采样（curr/prev 都要一致）
  CloudT::Ptr filterCloud(const CloudT::Ptr& in, float leaf, double zmin, double zmax) {
    // z 高度窗口
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(in);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(static_cast<float>(zmin), static_cast<float>(zmax));
    CloudT::Ptr zf(new CloudT);
    pass.filter(*zf);

    // 体素
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(zf);
    vg.setLeafSize(leaf, leaf, leaf);
    CloudT::Ptr out(new CloudT);
    vg.filter(*out);
    return out;
  }


  // 3.2 （可选）对点云的 z 做缩放（z 降权）
  void scaleCloudZ(CloudT::Ptr& cloud, float scale) {
    for (auto& p : cloud->points) p.z *= scale;
  }

  // 3.3 把 4x4 位姿投影到 SE(2)：仅保留 yaw & xy，清零 z/roll/pitch
  void projectToSE2(Eigen::Matrix4f& T) {
    Eigen::Matrix3f R = T.block<3,3>(0,0);
    float yaw = std::atan2(R(1,0), R(0,0));
    Eigen::Matrix3f Rz = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()).toRotationMatrix();
    T.block<3,3>(0,0) = Rz;
    T(2,3) = 0.0f;
  }


  // 将 4x4 变换矩阵（已近似平面）转为 Pose（z=0 + 纯 yaw）
  static void matToPoseSE2(const Eigen::Matrix4f& T, geometry_msgs::msg::Pose& pose) {
    // 位置
    pose.position.x = T(0,3);
    pose.position.y = T(1,3);
    pose.position.z = 0.0;  // 固定为 0

    // 纯 yaw
    Eigen::Matrix3f R = T.block<3,3>(0,0);
    float yaw = std::atan2(R(1,0), R(0,0));
    Eigen::Quaternionf q(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
    q.normalize();
    pose.orientation.x = q.x();
    pose.orientation.y = q.y();
    pose.orientation.z = q.z();
    pose.orientation.w = q.w();
  }

  // 生成 TF（z=0 + 纯 yaw）
  static geometry_msgs::msg::TransformStamped
  eigMatToTfSE2(const Eigen::Matrix4f& T, const std::string& parent, const std::string& child,
                const rclcpp::Time& stamp) {
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = stamp;
    tf.header.frame_id = parent;
    tf.child_frame_id  = child;

    tf.transform.translation.x = T(0,3);
    tf.transform.translation.y = T(1,3);
    tf.transform.translation.z = 0.0;  // 固定为 0

    Eigen::Matrix3f R = T.block<3,3>(0,0);
    float yaw = std::atan2(R(1,0), R(0,0));
    Eigen::Quaternionf q(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
    q.normalize();
    tf.transform.rotation.x = q.x();
    tf.transform.rotation.y = q.y();
    tf.transform.rotation.z = q.z();
    tf.transform.rotation.w = q.w();
    return tf;
  }

private:
  // 参数
  double leaf_size_;
  double max_corresp_dist_;
  int    max_iter_;
  double trans_eps_;
  double euclid_fitness_eps_;

  double z_min_ = -1.5, z_max_ = 1.5;
  bool   use_z_weight_ = false;
  double z_weight_ = 0.10;


  // 订阅 / 发布
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_aligned_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
  nav_msgs::msg::Path path_msg_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::string world_frame_ = "map";   // "map"
  std::string lidar_frame_ = "rslidar";   // "rslidar"

  // 上一帧点云与时间戳
  CloudT::Ptr prev_;
  rclcpp::Time last_stamp_;

  // 累计世界位姿
  Eigen::Matrix4f T_world_curr_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<IcpNode>());
  std::cout << "IcpNode shutdown" << std::endl;
  rclcpp::shutdown();
  return 0;
}
