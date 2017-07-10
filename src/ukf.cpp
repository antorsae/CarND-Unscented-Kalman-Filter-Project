#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;
  
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;
  
  // State dimension
  n_x_ = 5;
  
  // Augmented state dimension
  n_aug_ = n_x_ + 2;
  
  // Number of sigma points
  n_sigma_points_ = 2 * n_aug_ + 1;
  
  ///* Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  // initial predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, n_sigma_points_);
  
  // time when the state is true, in us
  time_us_ = 0;
  
  // weights of sigma points
  weights_ = VectorXd(n_sigma_points_);
  // initialize weights
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  
  for (int i = 1; i< n_sigma_points_; i++) {
    weights_(i) = 0.5 / (n_aug_ + lambda_);
  }
  
  // NIS
  NIS_radar_ = 0;
  NIS_laser_ = 0;
  
  // measurement noise covariance matrix
  R_laser_ = MatrixXd(2,2);
  
  R_laser_ << pow(std_laspx_, 2), 0,
              0,                  pow(std_laspy_, 2);
  

  //add measurement noise covariance matrix
  R_radar_ = MatrixXd(3,3);
  
  R_radar_ << pow(std_radr_, 2 ),   0,                       0,
              0,                    pow(std_radphi_, 2),     0,
              0,                    0,                       pow(std_radrd_, 2);
  
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  
  if (!is_initialized_) {
    // first measurement
    VectorXd x_init(4);
    
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      double rho   = meas_package.raw_measurements_[0];
      double phi   = meas_package.raw_measurements_[1];
      
      // we cannot use rho_dt since it's the radar measured relative speed
      // not the object's actual rho speed
      
      x_ << rho * cos(phi), rho * sin(phi), 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      double px = meas_package.raw_measurements_[0];
      double py = meas_package.raw_measurements_[1];
      
      x_ << px, py, 0, 0, 0;
    }
    
    P_ << 0.5, 0, 0,  0,  0,
          0, 0.5, 0,  0,  0,
          0, 0,  10,  0,  0,
          0, 0,   0, 10,  0,
          0, 0,   0,  0, 10;
    
    time_us_ = meas_package.timestamp_;
    
    is_initialized_ = true;
    return;
  }
  
  if (((meas_package.sensor_type_ == MeasurementPackage::RADAR) && (!use_radar_)) ||
    ((meas_package.sensor_type_ == MeasurementPackage::LASER) && (!use_laser_)) )
    return;
  
  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0f;
  time_us_ = meas_package.timestamp_;
  
  Prediction(delta_t);
  
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  
  //create augmented mean vector
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;
  
  //create augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  
  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug(n_aug_ - 2, n_aug_ - 2) = pow(std_a_, 2);
  P_aug(n_aug_ - 1, n_aug_ - 1) = pow(std_yawdd_, 2);
  
  MatrixXd L = P_aug.llt().matrixL();
  
  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sigma_points_);
  
  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++) {
    Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }
  
  //predict sigma points
  for (int i = 0; i< n_sigma_points_; i++) {
    
    const double p_x  = Xsig_aug(0,i);
    const double p_y  = Xsig_aug(1,i);
    
    const double v    = Xsig_aug(2,i);
    
    const double yaw  = Xsig_aug(3,i);
    const double yawd = Xsig_aug(4,i);
    
    const double nu_a = Xsig_aug(5,i);
    const double nu_yawdd = Xsig_aug(6,i);
    
    //predicted state values
    double px_p, py_p;
    
    //avoid division by zero
    if (fabs(yawd) >= UKF_EPSILON) {
      px_p = p_x + v/yawd * ( sin (yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v/yawd * ( cos (yaw) - cos(yaw + yawd * delta_t) );
    } else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }
    
    double v_p = v;
    
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;
    
    //add noise model
    px_p   += 0.5 * nu_a * pow(delta_t, 2) * cos(yaw);
    py_p   += 0.5 * nu_a * pow(delta_t, 2) * sin(yaw);
    v_p    += nu_a * delta_t;
    
    yaw_p  += 0.5 * nu_yawdd * pow(delta_t, 2);
    yawd_p += nu_yawdd * delta_t;
    
    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
  
  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < n_sigma_points_; i++) {  //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }
  
  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < n_sigma_points_; i++) {  //iterate over sigma points
    
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    x_diff(3) = tools.ConstrainAngle(x_diff(3));
    
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  const Eigen::VectorXd z = meas_package.raw_measurements_;
  int n_z = z.size();
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sigma_points_);
  
  //transform sigma points into measurement space
  for (int i = 0; i < n_sigma_points_; i++) {  //2n+1 simga points
    
    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    
    // measurement model
    Zsig(0,i) = p_x;
    Zsig(1,i) = p_y;
  }
 
  NIS_laser_ = Update(z, Zsig, R_laser_);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  const Eigen::VectorXd z = meas_package.raw_measurements_;
  
  int n_z = z.size();
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sigma_points_);
  
  //transform sigma points into measurement space
  for (int i = 0; i < n_sigma_points_; i++) {  //2n+1 simga points
    
    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v   = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    
    double v_x = cos(yaw)*v;
    double v_y = sin(yaw)*v;
    double rho = hypot(p_x, p_y);
    
    // measurement model
    Zsig(0,i) = rho;                                    //r
    Zsig(1,i) = atan2(p_y, p_x);                        //phi
    if (fabs(rho) >= UKF_EPSILON)
      Zsig(2,i) = (p_x * v_x + p_y * v_y ) / rho ;      //r_dot or don't update
  }
  
  NIS_radar_ = Update(z, Zsig, R_radar_);
}

double UKF::Update(const VectorXd &z, const MatrixXd &Zsig, const MatrixXd &R) {
  
  int n_z = z.size();
  
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < n_sigma_points_; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < n_sigma_points_; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    
    //angle normalization
    z_diff(1) = tools.ConstrainAngle(z_diff(1));
    
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  
  S = S + R;
  
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  
  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_sigma_points_; i++) {  //2n+1 simga points
    
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    z_diff(1) = tools.ConstrainAngle(z_diff(1));
    
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    x_diff(3) = tools.ConstrainAngle(x_diff(3));
    
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  
  //Kalman gain K;
  MatrixXd Si = S.inverse();
  MatrixXd K = Tc * Si;
  
  //residual
  VectorXd z_diff = z - z_pred;
  
  //angle normalization
  z_diff(1) = tools.ConstrainAngle(z_diff(1));
  
  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
  
  return z_diff.transpose() * Si * z_diff;
}
