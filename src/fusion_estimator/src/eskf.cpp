#include "fusion_estimator/eskf.hpp"
#include <algorithm>
#include <cmath>

ESKF::ESKF() {
    ResetCov();
}

void ESKF::setGravity(double g) { gravity_ = g; }


void ESKF::setProcessNoise(double sigma_na, double sigma_ng,
    double sigma_nba, double sigma_nbg) {
    sigma_na_ = sigma_na; sigma_ng_ = sigma_ng;
    sigma_nba_ = sigma_nba; sigma_nbg_ = sigma_nbg;
}


void ESKF::ResetCov(double v) {
    P_.setIdentity();
    P_ *= v;
}


void ESKF::Predict(const Eigen::Vector3d& f_b, const Eigen::Vector3d& w_b, double dt) {
    if (!(dt > 0.0) || dt > 0.2) return;


    // 去偏置
    Eigen::Vector3d omega = w_b - X_.bg; // rad/s
    Eigen::Vector3d f_b_unbiased = f_b - X_.ba; // m/s^2


    // 姿态右乘增量
    Eigen::Quaterniond dq(1.0, 0.5*omega.x()*dt, 0.5*omega.y()*dt, 0.5*omega.z()*dt);
    X_.q = (X_.q * dq).normalized();


    // 加速度旋到世界并加重力
    const Eigen::Vector3d g(0.0, 0.0, -gravity_);
    const Eigen::Matrix3d Rwb = X_.q.toRotationMatrix();
    Eigen::Vector3d acc_w = Rwb * f_b_unbiased + g;


    // 积分 v, p
    X_.v += acc_w * dt;
    X_.p += X_.v * dt + 0.5 * acc_w * dt * dt;


    // 线性化传播
    Eigen::Matrix<double,15,15> F = Eigen::Matrix<double,15,15>::Zero();
    Eigen::Matrix<double,15,12> G = Eigen::Matrix<double,15,12>::Zero(); // [na ng nba nbg]


    F.block<3,3>(0,3) = Eigen::Matrix3d::Identity();
    F.block<3,3>(3,6) = - Rwb * Skew(f_b_unbiased);
    F.block<3,3>(3,9) = - Rwb;
    F.block<3,3>(6,6) = - Skew(omega);
    F.block<3,3>(6,12) = - Eigen::Matrix3d::Identity();


    G.block<3,3>(3,0) = Rwb; // dv/na
    G.block<3,3>(6,3) = Eigen::Matrix3d::Identity(); // dtheta/ng
    G.block<3,3>(9,6) = Eigen::Matrix3d::Identity(); // dba/nba
    G.block<3,3>(12,9) = Eigen::Matrix3d::Identity(); // dbg/nbg


    Eigen::Matrix<double,15,15> Phi = Eigen::Matrix<double,15,15>::Identity() + F * dt;


    Eigen::Matrix<double,12,12> Qc = Eigen::Matrix<double,12,12>::Zero();
    Qc.block<3,3>(0,0) = (sigma_na_ * sigma_na_) * Eigen::Matrix3d::Identity();
    Qc.block<3,3>(3,3) = (sigma_ng_ * sigma_ng_) * Eigen::Matrix3d::Identity();
    Qc.block<3,3>(6,6) = (sigma_nba_ * sigma_nba_) * Eigen::Matrix3d::Identity();
    Qc.block<3,3>(9,9) = (sigma_nbg_ * sigma_nbg_) * Eigen::Matrix3d::Identity();


    Eigen::Matrix<double,15,15> Qd = G * Qc * G.transpose() * dt;


    P_ = Phi * P_ * Phi.transpose() + Qd;
}

void ESKF::UpdatePos(const Eigen::Vector3d& pz, const Eigen::Matrix3d& Rpos) {
    Eigen::Matrix<double,3,15> H = Eigen::Matrix<double,3,15>::Zero();
    H.block<3,3>(0,0) = Eigen::Matrix3d::Identity();


    Eigen::Vector3d r = pz - X_.p;
    Eigen::Matrix3d S = H * P_ * H.transpose() + Rpos;
    Eigen::Matrix<double,15,3> K = P_ * H.transpose() * S.inverse();


    Eigen::Matrix<double,15,1> dx = K * r;


    X_.p += dx.segment<3>(0);
    X_.v += dx.segment<3>(3);
    Eigen::Vector3d dth = dx.segment<3>(6);
    Eigen::Quaterniond dq(1, 0.5*dth.x(), 0.5*dth.y(), 0.5*dth.z());
    X_.q = (X_.q * dq).normalized();
    X_.ba += dx.segment<3>(9);
    X_.bg += dx.segment<3>(12);


    const auto I = Eigen::Matrix<double,15,15>::Identity();
    P_ = (I - K*H) * P_ * (I - K*H).transpose() + K * Rpos * K.transpose();
}

void ESKF::UpdateVel(const Eigen::Vector3d& vz, const Eigen::Matrix3d& Rvel) {
    Eigen::Matrix<double,3,15> H = Eigen::Matrix<double,3,15>::Zero();
    H.block<3,3>(0,3) = Eigen::Matrix3d::Identity();


    Eigen::Vector3d r = vz - X_.v;
    Eigen::Matrix3d S = H * P_ * H.transpose() + Rvel;
    Eigen::Matrix<double,15,3> K = P_ * H.transpose() * S.inverse();


    Eigen::Matrix<double,15,1> dx = K * r;


    X_.p += dx.segment<3>(0);
    X_.v += dx.segment<3>(3);
    Eigen::Vector3d dth = dx.segment<3>(6);
    Eigen::Quaterniond dq(1, 0.5*dth.x(), 0.5*dth.y(), 0.5*dth.z());
    X_.q = (X_.q * dq).normalized();
    X_.ba += dx.segment<3>(9);
    X_.bg += dx.segment<3>(12);


    const auto I = Eigen::Matrix<double,15,15>::Identity();
    P_ = (I - K*H) * P_ * (I - K*H).transpose() + K * Rvel * K.transpose();
}


void ESKF::UpdateYaw(double yaw, double var_yaw) {
    // 估计的 yaw（ENU）
    const auto R = X_.q.toRotationMatrix();
    double yaw_est = std::atan2(R(1,0), R(0,0));
    double r = std::atan2(std::sin(yaw - yaw_est), std::cos(yaw - yaw_est));


    Eigen::Matrix<double,1,15> H = Eigen::Matrix<double,1,15>::Zero();
    H(0, 6+2) = 1.0; // 仅约束 z 轴小角度


    double S = (H * P_ * H.transpose())(0,0) + var_yaw;
    Eigen::Matrix<double,15,1> K = P_ * H.transpose() * (1.0 / S);


    Eigen::Matrix<double,15,1> dx = K * Eigen::Matrix<double,1,1>(r);


    Eigen::Vector3d dth = dx.segment<3>(6);
    Eigen::Quaterniond dq(1, 0.5*dth.x(), 0.5*dth.y(), 0.5*dth.z());
    X_.q = (X_.q * dq).normalized();


    const auto I = Eigen::Matrix<double,15,15>::Identity();
    P_ = (I - K*H) * P_ * (I - K*H).transpose() + K * var_yaw * K.transpose();
}