#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  
  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if(estimations.size() != ground_truth.size()
     || estimations.size() == 0){
    rmse << -1,-1,-1,-1;
    return rmse;
  }
  
  //accumulate squared residuals
  for(unsigned int i=0; i < estimations.size(); ++i){
    
    VectorXd residual = estimations[i] - ground_truth[i];
    
    //coefficient-wise multiplication
    residual = residual.array() * residual.array();
    rmse += residual;
  }
  
  //calculate the mean
  rmse = rmse/estimations.size();
  
  //calculate the squared root
  rmse = rmse.array().sqrt();
  
  //return the result
  return rmse;
}

// from http://stackoverflow.com/questions/11498169/dealing-with-angle-wrap-in-c-code
double Tools::ConstrainAngle(double x){
  x = fmod(x + M_PI, 2 * M_PI);
  if (x < 0)
    x += 2 * M_PI;
  return x - M_PI;
}
