from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        
        Node(package='imu_direct', executable='imu_direct_node',  name='imu_node',  output='screen'),
        Node(package='gnss_ins_tool', executable='gnss_ins_pose_node',name='gnss_node',output='screen',
             parameters=[{'publish_tf': False}]),  # 避免 TF 冲突

        # 融合节点
        Node(
            package='fusion_estimator',
            executable='eskf_node',
            name='fusion_eskf',
            output='screen',
            parameters=[{
                'imu_topic': '/imu/data',
                'gnss_odom_topic': '/gnss_ins/odom',
                'frame_map': 'map',
                'frame_base': 'base_link',
                'publish_tf': True,
                'publish_path': True,
                'use_yaw_meas': True,
                'yaw_std_deg': 2.0,
                'gravity': 9.80665,
                'acc_noise': 0.08,
                'gyro_noise': 0.005,
                'ba_walk': 0.001,
                'bg_walk': 0.0002,
            }]
        )
    ])
