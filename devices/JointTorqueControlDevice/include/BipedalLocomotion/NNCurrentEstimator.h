/**
 * @file NNCurrentEstimator.h
 * @authors Filippo Passerini
 * @copyright 2025 Istituto Italiano di Tecnologia (IIT). This software may be modified and
 * distributed under the terms of the BSD-3-Clause license.
 */

#ifndef BIPEDAL_LOCOMOTION_FRAMEWORK_NN_CURRENT_ESTIMATOR_H
#define BIPEDAL_LOCOMOTION_FRAMEWORK_NN_CURRENT_ESTIMATOR_H

#include <memory>
#include <mutex>
#include <yarp/sig/Vector.h>

/**
 * NNCurrentEstimator is a class that performs residual current estimation
 * using a neural network model.
 * This class uses NN models exported as ONNX files
 * Such NNs take 3 or 4 quantities as input:
 * - Joint position
 * - Joint velocity
 * - Desired force/torque (from an high level controller)
 * - Measured motor current (optional input)
 */
class NNCurrentEstimator
{
public:
    NNCurrentEstimator();
    ~NNCurrentEstimator();

    /**
     * Initialize the estimator
     * @param[in] modelPath a string representing the path to the ONNX model
	 * @param[in] intraOpNumThreads a std::size_t representing the number of threads to be used for intra-op parallelism
	 * @param[in] interOpNumThreads a std::size_t representing the number of threads to be used for inter-op parallelism
     * @return true if the initialization is successful, false otherwise
     */
    bool initialize(const std::string& modelPath,
                const std::size_t intraOpNumThreads = 1,
                const std::size_t interOpNumThreads = 1);

    /**
     * Reset the estimator
     * This function clears the buffers
     * and resets the internal state of the estimator
     * @return void
     */
    void resetEstimator();

    /**
     * Estimate the current residual starting from raw data
     * @param[in] inputJointPosition a double representing the joint position
     * @param[in] inputJointVelocity a double representing the joint velocity
     * @param[in] inputForce a double representing the force
     * @param[in] inputMotorCurrent a double representing the motor current
     * @param[out] output a double representing the current compensation
     * @return true if the estimation is successful, false otherwise
     */
    bool estimate(double inputJointPosition,
                  double inputJointVelocity,
                  double inputForce,
                  double inputMotorCurrent,
                  double& output);


private:
    struct Impl;
    std::unique_ptr<Impl> m_pimpl;
};

#endif // BIPEDAL_LOCOMOTION_FRAMEWORK_NN_CURRENT_ESTIMATOR_H
