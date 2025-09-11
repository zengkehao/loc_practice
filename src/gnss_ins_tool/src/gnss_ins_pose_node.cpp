#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>

#include <GeographicLib/UTMUPS.hpp>  // UTM
#include <cmath>
#include <memory>

// 自定义消息
#include <chcnv_cgi_msgs/msg/hcinspvatzcb.hpp>

// ---------- WGS84 <-> ECEF/ENU 工具 ----------
namespace geo {
static constexpr double a  = 6378137.0;           // WGS84 semi-major axis
static constexpr double f  = 1.0 / 298.257223563; // flattening
static constexpr double b  = a * (1.0 - f);
static constexpr double e2 = (a*a - b*b) / (a*a);
static constexpr double ep2= (a*a - b*b) / (b*b);

// WGS84 -> ECEF/ENU   ​​​
inline void geodeticToECEF(double lat, double lon, double h,
                           double &x, double &y, double &z) {
  const double sLat = sin(lat), cLat = cos(lat);
  const double sLon = sin(lon), cLon = cos(lon);
  const double N = a / sqrt(1.0 - e2 * sLat * sLat);
  x = (N + h) * cLat * cLon;
  y = (N + h) * cLat * sLon;
  z = (b*b / (a*a) * N + h) * sLat;
}

// ECEF（地心直角坐标系） -> ENU（​局部平面直角坐标系）
inline void ecefToENU(double x, double y, double z,
                      double x0, double y0, double z0,
                      double lat0, double lon0,
                      double &e, double &n, double &u) {
  // 以原点(lat0, lon0, h0)的 ECEF 为参考
  double dx = x - x0, dy = y - y0, dz = z - z0;
  const double sLat0 = sin(lat0), cLat0 = cos(lat0);
  const double sLon0 = sin(lon0), cLon0 = cos(lon0);
  // ENU 旋转
  e = -sLon0*dx + cLon0*dy;
  n = -sLat0*cLon0*dx - sLat0*sLon0*dy + cLat0*dz;
  u =  cLat0*cLon0*dx + cLat0*sLon0*dy + sLat0*dz;
}
} // namespace geo

class GnssInsPoseNode : public rclcpp::Node {
public:
  GnssInsPoseNode() : Node("gnss_ins_pose_node") {
    // 参数
    use_first_fix_as_origin_ = declare_parameter<bool>("use_first_fix_as_origin", true);//是否使用首次定位作为坐标原点
    heading_is_from_north_   = declare_parameter<bool>("heading_is_from_north", true);//航向角是否以正北为参考
    map_frame_id_            = declare_parameter<std::string>("map_frame_id", "map");//地图坐标系id
    base_frame_id_           = declare_parameter<std::string>("base_frame_id", "base_link");//机体坐标系id
    publish_tf_              = declare_parameter<bool>("publish_tf", true);//是否发布TF变换
    use_utm_                 = declare_parameter<bool>("use_utm", false);//是否使用UTM坐标系

    //预设原点的经纬度和高度
    origin_lat_ = declare_parameter<double>("origin_lat", 0.0);  
    origin_lon_ = declare_parameter<double>("origin_lon", 0.0);
    origin_alt_ = declare_parameter<double>("origin_alt", 0.0);

    //不使用首次定位作为原点，转换为ECEF坐标
    if (!use_first_fix_as_origin_) {
        if (use_utm_) {//如果使用UTM坐标系
        int z; bool n;
        double e, nn;
        GeographicLib::UTMUPS::Forward(origin_lat_, origin_lon_, z, n, e, nn);
        utm_zone_ = z; utm_northp_ = n; e0_ = e; n0_ = nn; u0_ = origin_alt_;
        origin_set_ = true;
        RCLCPP_INFO(get_logger(), "[UTM] Fixed origin: zone=%d %s  E0=%.3f N0=%.3f H0=%.3f",
                    utm_zone_, utm_northp_?"N":"S", e0_, n0_, u0_);
        } else {
        origin_set_ = true;
        origin_lat_rad_ = deg2rad(origin_lat_);
        origin_lon_rad_ = deg2rad(origin_lon_);
        geo::geodeticToECEF(origin_lat_rad_, origin_lon_rad_, origin_alt_,
                            x0_ecef_, y0_ecef_, z0_ecef_);
        RCLCPP_INFO(get_logger(), "Using fixed origin (%.8f, %.8f, %.3f m).",
                    origin_lat_, origin_lon_, origin_alt_);
    }
  }

    // 发布者
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/gnss_ins/pose", 10);
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/gnss_ins/odom", 10);
    path_pub_ = create_publisher<nav_msgs::msg::Path>("/gnss_ins/path", 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // 订阅 GNSS/INS
    sub_ = create_subscription<chcnv_cgi_msgs::msg::Hcinspvatzcb>(
      "/chcnav/devpvt", rclcpp::SensorDataQoS(),
      std::bind(&GnssInsPoseNode::callBack, this, std::placeholders::_1));

    path_msg_.header.frame_id = map_frame_id_;

    RCLCPP_INFO(get_logger(), "Gnss_ins_pose node started. Subscribing: %s mode=%s", 
        sub_->get_topic_name(),use_utm_?"UTM":"ENU"); 
  }

private:
  void callBack(const chcnv_cgi_msgs::msg::Hcinspvatzcb::SharedPtr msg) {
    // 1) 经纬度从角度制转换为弧度制
    const double lat_rad = deg2rad(msg->latitude);
    const double lon_rad = deg2rad(msg->longitude);
    const double alt     = static_cast<double>(msg->altitude);

    double px = 0;
    double py = 0;
    double pz = 0;

    if (use_utm_) {
      // ----- UTM 模式 -----
      int zone; 
      bool northp;
      double easting, northing;
      GeographicLib::UTMUPS::Forward(msg->latitude, msg->longitude,
                                     zone, northp, easting, northing); // 输入度

      if (!origin_set_) {
        origin_set_ = true;
        utm_zone_ = zone; 
        utm_northp_ = northp;
        e0_ = easting; 
        n0_ = northing; 
        u0_ = alt;
        RCLCPP_INFO(get_logger(), "[UTM] Origin set from first fix: zone=%d %s  E0=%.3f N0=%.3f H0=%.3f",
                    utm_zone_, utm_northp_?"N":"S", e0_, n0_, u0_);
      } else if (zone != utm_zone_ || northp != utm_northp_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 3000,
          "UTM zone changed (old %d%s -> new %d%s). Using old zone origin.",
          utm_zone_, utm_northp_?"N":"S", zone, northp?"N":"S");
      }
      
      px = easting  - e0_;
      py = northing - n0_;
      pz = alt      - u0_;
    }
    else {
      // ----- ENU 模式 -----
      if (!origin_set_) {// 2) 原点初始化（首次有效定位）
        origin_set_ = true;
        origin_lat_rad_ = lat_rad;
        origin_lon_rad_ = lon_rad;
        //原点转换为ECEF坐标
        geo::geodeticToECEF(origin_lat_rad_, origin_lon_rad_, alt, x0_ecef_, y0_ecef_, z0_ecef_);
        RCLCPP_INFO(get_logger(), "[ENU] Origin set from first fix: lat=%.8f lon=%.8f alt=%.3f",
                    msg->latitude, msg->longitude, msg->altitude);
      }
      // 3) WGS84 -> ECEF -> ENU or UTM
      double x, y, z;
      geo::geodeticToECEF(lat_rad, lon_rad, alt, x, y, z);
      double e, n, u;
      geo::ecefToENU(x, y, z, x0_ecef_, y0_ecef_, z0_ecef_, origin_lat_rad_, origin_lon_rad_, e, n, u);
      px = e; py = n; pz = u;
    }

    // 4) 姿态：roll/pitch/yaw 从角度制转为弧度制，并处理“航向定义”差异
    double roll  = deg2rad(msg->roll);
    double pitch = deg2rad(msg->pitch);
    double yaw   = deg2rad(msg->yaw);

    // 若 yaw 为“相对北的航向角（逆时针为正）”，需转换成 ENU 内部偏航（相对东轴）
    // ENU: x=East, y=North。若 heading 从北逆时针，则与 ENU yaw 关系为：yaw_enu = heading - 90°（即 -pi/2）
    if (heading_is_from_north_) {
      yaw = yaw - M_PI_2; // -90 度
    }

    // 以 ZYX (yaw-pitch-roll) 生成四元数（ENU，Z 轴朝上）
    tf2::Quaternion q;
    q.setRPY(roll, pitch, yaw);
    geometry_msgs::msg::Quaternion q_msg = tf2::toMsg(q);

    // 5) 发布 PoseStamped
    geometry_msgs::msg::PoseStamped ps;
    ps.header.stamp = this->now();
    ps.header.frame_id = map_frame_id_;
    ps.pose.position.x = px;
    ps.pose.position.y = py;
    ps.pose.position.z = pz;
    ps.pose.orientation = q_msg;
    pose_pub_->publish(ps);

    // 6) 发布 Odometry（协方差）
    nav_msgs::msg::Odometry odom;
    odom.header = ps.header;
    odom.child_frame_id = base_frame_id_;
    odom.pose.pose = ps.pose;

    // 位置协方差：由 position_stdev (lat,lon,alt) 近似映射到 ENU。这里简化为对角阵。
    for (int i = 0; i < 36; ++i) odom.pose.covariance[i] = 0.0;
    if (msg->position_stdev.size() == 3) {
      const double sx = static_cast<double>(msg->position_stdev[1]); // lon -> E 近似
      const double sy = static_cast<double>(msg->position_stdev[0]); // lat -> N 近似
      const double sz = static_cast<double>(msg->position_stdev[2]); // alt -> U
      odom.pose.covariance[0]  = sx*sx; // xx
      odom.pose.covariance[7]  = sy*sy; // yy
      odom.pose.covariance[14] = sz*sz; // zz
    }

    // 姿态协方差：由 euler_stdev (deg) 近似到 rad^2，对角阵。
    if (msg->euler_stdev.size() == 3) {
      const double sr = deg2rad(static_cast<double>(msg->euler_stdev[0]));
      const double sp = deg2rad(static_cast<double>(msg->euler_stdev[1]));
      const double sy = deg2rad(static_cast<double>(msg->euler_stdev[2]));
      odom.pose.covariance[21] = sr*sr; // rr
      odom.pose.covariance[28] = sp*sp; // pp
      odom.pose.covariance[35] = sy*sy; // yy
    }

    odom_pub_->publish(odom);

    // 7) Path（可视化轨迹）
    path_msg_.header.stamp = this->now();
    path_msg_.poses.push_back(ps);
    path_pub_->publish(path_msg_);

    // 8) TF: map -> base_link
    if (publish_tf_) {
      geometry_msgs::msg::TransformStamped tf;
      tf.header = ps.header;
      tf.child_frame_id = base_frame_id_;
      tf.transform.translation.x = px;
      tf.transform.translation.y = py;
      tf.transform.translation.z = pz;
      tf.transform.rotation = q_msg;
      tf_broadcaster_->sendTransform(tf);
    }
  }

  static inline double deg2rad(double d){ return d * M_PI / 180.0; }

  // 参数/状态
  bool use_utm_{false};
  bool use_first_fix_as_origin_{true};
  bool heading_is_from_north_{true};
  bool publish_tf_{true};
  bool origin_set_{false};
  std::string map_frame_id_{"map"};
  std::string base_frame_id_{"base_link"};
  //ENU原点
  double origin_lat_{0.0}, origin_lon_{0.0}, origin_alt_{0.0};
  double origin_lat_rad_{0.0}, origin_lon_rad_{0.0};
  double x0_ecef_{0.0}, y0_ecef_{0.0}, z0_ecef_{0.0};
  //UTM原点
  int utm_zone_{0};
  bool utm_northp_{true};
  double e0_{0.0}, n0_{0.0}, u0_{0.0};

  // ROS IO
  rclcpp::Subscription<chcnv_cgi_msgs::msg::Hcinspvatzcb>::SharedPtr sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  nav_msgs::msg::Path path_msg_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GnssInsPoseNode>());
  rclcpp::shutdown();
  return 0;
}
