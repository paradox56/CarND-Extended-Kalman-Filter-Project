#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * This function takes in Groud Truth and Estimations from the Kalman Fitler
     and return Root Mean Squared Error for Evaluation
   */

  //Initialize rmse in a vector with the same dimension of the state vector
  VectorXd rmse_vec(4);
  rmse_vec << 0,0,0,0;

  if (estimations.size() == 0 && estimations.size() != ground_truth.size()){
    cout << "Error: Dimension Mismatch between Ground Truth and Estimations, or empty Estimations" << endl;
    return rmse_vec;
    }

  for (int i=0; i < estimations.size(); ++i) {
    VectorXd diff = estimations[i]-ground_truth[i];
    diff = (diff.array()*diff.array())/4; //Normalized Square
    rmse_vec += diff;
    }

    // Calculate Mean
    rmse_vec = rmse_vec/estimations.size();

    // calculate the squared root
    rmse_vec = rmse_vec.array().sqrt();

    // return the result
    return rmse_vec;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * This function calculates the jacobian matrix for linearizing the functon h through
   * First-order Taylor Expansion
   */

  //The dimension of the Jacobian matrix is based on the state vecotr x and radar measurement vector z
  MatrixXd Hj_mat(3,4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // Define Constant Parameters as
  float c1 = px*px+py*py;
  float c2 = sqrt(c1);
  float c3 = (c1*c2);

  // check division by zero
  if (fabs(c1) < 0.00001) {
    cout << "Error: Division by zero" << endl;
    return Hj_mat;
  }

  // compute the Jacobian matrix
  Hj_mat << (px/c2), (py/c2), 0, 0,
           -(py/c1), (px/c1), 0, 0,
           py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return Hj_mat;
}
