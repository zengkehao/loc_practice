#include <Eigen/Dense>


class ESKF {
    public:
    struct State {
    Eigen::Vector3d p{Eigen::Vector3d::Zero()}; // world
    Eigen::Vector3d v{Eigen::Vector3d::Zero()}; // world
    Eigen::Quaterniond q{Eigen::Quaterniond::Identity()}; // body->world
    Eigen::Vector3d ba{Eigen::Vector3d::Zero()}; // accel bias (body)
    Eigen::Vector3d bg{Eigen::Vector3d::Zero()}; // gyro bias (body)
    };


    ESKF();


    void setGravity(double g);
    void setProcessNoise(double sigma_na, double sigma_ng,
    double sigma_nba, double sigma_nbg);


    // IMU predict: f_b(m/s^2) is specific force (含重力效应的IMU读数), w_b(rad/s)
    void Predict(const Eigen::Vector3d& f_b_mps2,
    const Eigen::Vector3d& w_b_rps,
    double dt);


    // 外部量测更新（把 /gnss_ins/odom 当作位置/可选航向）
    void UpdatePos(const Eigen::Vector3d& pz_world, const Eigen::Matrix3d& Rpos);
    void UpdateVel(const Eigen::Vector3d& vz_world, const Eigen::Matrix3d& Rvel); // 可选
    void UpdateYaw(double yaw_world_rad, double var_yaw);


    const State& X() const { return X_; }
    State& mutable_state() { return X_; }


    void ResetCov(double v = 1e-3);


private:
    static inline Eigen::Matrix3d Skew(const Eigen::Vector3d& a) {
    Eigen::Matrix3d S; S << 0, -a.z(), a.y(), a.z(), 0, -a.x(), -a.y(), a.x(), 0; return S;
    }


    State X_{};
    Eigen::Matrix<double,15,15> P_{}; // [dp dv dtheta dba dbg]
    double gravity_ = 9.80665;


    // 连续时间过程噪声标准差
    double sigma_na_ = 0.50; // m/s^2
    double sigma_ng_ = 0.02; // rad/s
    double sigma_nba_ = 0.01; // m/s^2/s
    double sigma_nbg_ = 0.001; // rad/s^2
};