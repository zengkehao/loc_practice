#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

class BagSubscriber : public rclcpp::Node
{
public:
    BagSubscriber()
    : Node("test_subscriber")
    {
        /*
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "topic", 10,
            std::bind(&BagSubcriber::topic_callback, this, std::placeholders::_1));*/

        // 选择要订阅的话题
        std::string topic_name = this->declare_parameter<std::string>("topic", "/delphin_m1p_points");

        sub_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            topic_name, 10,
            std::bind(&BagSubscriber::pc_callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "Subscribed to PointCloud2 topic: %s", topic_name.c_str());
    }

private:
    void topic_callback(const std_msgs::msg::String::SharedPtr msg) const
    {
        RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
    }

    // 订阅 PointCloud2 回调
    void pc_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) const
    {
        RCLCPP_INFO(this->get_logger(),
            "[PointCloud2] width=%u, height=%u, fields=%zu, total_points=%u",
            msg->width, msg->height, msg->fields.size(),
            msg->width * msg->height);
    }

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
    // 三种订阅对象
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pc_;

};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BagSubscriber>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
