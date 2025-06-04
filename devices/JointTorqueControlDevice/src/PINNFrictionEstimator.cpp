/**
 * @file PINNFrictionEstimator.cpp
 * @authors Ines Sorrentino
 * @copyright 2024 Istituto Italiano di Tecnologia (IIT). This software may be modified and
 * distributed under the terms of the BSD-3-Clause license.
 */

#include <memory>
#include <deque>
#include <string>
#include <cmath>
#include <numeric>

#include <Eigen/Dense>

// onnxruntime
#include <onnxruntime_cxx_api.h>

#include <BipedalLocomotion/ParametersHandler/IParametersHandler.h>
#include <BipedalLocomotion/System/VariablesHandler.h>
#include <BipedalLocomotion/TextLogging/Logger.h>

#include <BipedalLocomotion/PINNFrictionEstimator.h>


struct PINNFrictionEstimator::Impl
{
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo;

    std::deque<float> jointVelocityBuffer;
    std::deque<float> motorVelocityBuffer;
    std::deque<float> jointPositionBuffer;
    std::deque<float> motorPositionBuffer;
    std::deque<float> motorTemperatureBuffer;

    size_t historyLength;
    size_t inputType = 0;

    struct DataStructured
    {
        std::vector<float> rawData;

        Ort::Value tensor{nullptr};
        std::array<int64_t, 2> shape;
    };

    DataStructured structuredInput;
    DataStructured structuredOutput;

    Impl()
        : memoryInfo(::Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU))
    {
    }
};

PINNFrictionEstimator::PINNFrictionEstimator()
{
    m_pimpl = std::make_unique<PINNFrictionEstimator::Impl>();
}

PINNFrictionEstimator::~PINNFrictionEstimator() = default;

bool PINNFrictionEstimator::initialize(const std::string& networkModelPath,
                                       const std::size_t intraOpNumThreads,
                                       const std::size_t interOpNumThreads,
                                       const std::size_t inputType)
{
    std::basic_string<ORTCHAR_T> networkModelPathAsOrtString(networkModelPath.begin(),
                                                             networkModelPath.end());

    Ort::SessionOptions sessionOptions;

	// Set the number of intra-op threads
	if (intraOpNumThreads > 0)
    {
        sessionOptions.SetIntraOpNumThreads(intraOpNumThreads);
    }
	// Set the number of inter-op threads
	if (interOpNumThreads > 0)
    {
        sessionOptions.SetInterOpNumThreads(interOpNumThreads);
    }
	m_pimpl->session = std::make_unique<Ort::Session>(m_pimpl->env,
                                                      networkModelPathAsOrtString.c_str(),
                                                      sessionOptions);

    if (m_pimpl->session == nullptr)
    {
        BipedalLocomotion::log()->error("Unable to load the model from the file: {}", networkModelPath);
        return false;
    }

    m_pimpl->inputType = inputType;
    // case inputType = 1: model takes as input motor velocity (a sequence), joint velocity (a sequence) and motor temperature (a single sample)
    // case inputType = 2: model takes as input motor velocity (a sequence) and joint velocity (a sequence)
    // case inputType = 3: model takes as input motor position (a sequence), joint position (a sequence) and motor temperature (a single sample)
    // case inputType = 4: model takes as input motor velocity (a sequence), joint velocity (a sequence), motor position (a sequence), joint position (a sequence) and motor temperature (a single sample)

    // Get model input size
    std::vector<int64_t> inputShape = m_pimpl->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    // inputCount is the total number of inputs
    // For example, if inputType==1, inputCount = historyLength * 2 + 1 
    // (because we have 2 sequences, motor and joint velocity, and 1 single sample, motor temperature)
    // If inputType==2, inputCount = historyLength * 2
    // If inputType==3, inputCount = historyLength * 2 + 1
    // If inputType==4, inputCount = historyLength * 4 + 1
    const std::size_t inputCount = inputShape[1];

    // Compte the historyLength
    if (inputType == 1 || inputType == 3)
    {
        // Remove one element (single motor temperature timestamp) and calculate historyLength
        m_pimpl->historyLength = (inputCount - 1) / 2;
    }else if (inputType == 4)
    {
        // Remove one element (single motor temperature timestamp) and calculate historyLength by dividing by 4
        m_pimpl->historyLength = (inputCount - 1) / 4;
    }else{
        // Since there is no motor temperature, there is no need to remove any element, so historyLength is just inputCount / 2
        m_pimpl->historyLength = inputCount / 2;
    }

    // format the input
    m_pimpl->structuredInput.rawData.resize(inputCount);
    m_pimpl->structuredInput.shape[0] = 1; // batch
    m_pimpl->structuredInput.shape[1] = inputCount;

    m_pimpl->jointVelocityBuffer.resize(m_pimpl->historyLength);
    m_pimpl->motorVelocityBuffer.resize(m_pimpl->historyLength);
    m_pimpl->jointPositionBuffer.resize(m_pimpl->historyLength);
    m_pimpl->motorPositionBuffer.resize(m_pimpl->historyLength);
    m_pimpl->motorTemperatureBuffer.resize(m_pimpl->historyLength);

    // create tensor required by onnx
    m_pimpl->structuredInput.tensor
        = Ort::Value::CreateTensor<float>(m_pimpl->memoryInfo,
                                          m_pimpl->structuredInput.rawData.data(),
                                          m_pimpl->structuredInput.rawData.size(),
                                          m_pimpl->structuredInput.shape.data(),
                                          m_pimpl->structuredInput.shape.size());

    // format the output
    const std::size_t outputSize = 1;

    // resize the output
    m_pimpl->structuredOutput.rawData.resize(outputSize);
    m_pimpl->structuredOutput.shape[0] = 1; // batch
    m_pimpl->structuredOutput.shape[1] = outputSize;

    // create tensor required by onnx
    m_pimpl->structuredOutput.tensor
        = Ort::Value::CreateTensor<float>(m_pimpl->memoryInfo,
                                          m_pimpl->structuredOutput.rawData.data(),
                                          m_pimpl->structuredOutput.rawData.size(),
                                          m_pimpl->structuredOutput.shape.data(),
                                          m_pimpl->structuredOutput.shape.size());

    return true;
}

void PINNFrictionEstimator::resetEstimator()
{
    m_pimpl->motorVelocityBuffer.clear();
    m_pimpl->jointVelocityBuffer.clear();
    m_pimpl->motorPositionBuffer.clear();
    m_pimpl->jointPositionBuffer.clear();
    m_pimpl->motorTemperatureBuffer.clear();
}

bool PINNFrictionEstimator::estimate(double inputMotorVelocity,
                                     double inputJointVelocity,
                                     double inputMotorPosition,
                                     double inputJointPosition,
                                     double inputMotorTemperature,
                                     double& adjustedMotorTemperature,
                                     double& output)
{
    if (m_pimpl->motorVelocityBuffer.size() == m_pimpl->historyLength)
    {
        // The buffer is full, remove the oldest element
        m_pimpl->motorVelocityBuffer.pop_front();
        m_pimpl->jointVelocityBuffer.pop_front();
        m_pimpl->motorPositionBuffer.pop_front();
        m_pimpl->jointPositionBuffer.pop_front();
        m_pimpl->motorTemperatureBuffer.pop_front();
    }

    // Push element into the queue
    m_pimpl->motorVelocityBuffer.push_back(inputMotorVelocity);
    m_pimpl->jointVelocityBuffer.push_back(inputJointVelocity);
    m_pimpl->motorTemperatureBuffer.push_back(inputMotorTemperature);
    m_pimpl->motorPositionBuffer.push_back(inputMotorPosition);
    m_pimpl->jointPositionBuffer.push_back(inputJointPosition);

    // Check if the buffer is full
    if (m_pimpl->motorVelocityBuffer.size() < m_pimpl->historyLength)
    {
        // The buffer is not full yet
        return false;
    }

    // Detect outlier in motor temperature
    adjustedMotorTemperature = inputMotorTemperature;
    if (m_pimpl->motorTemperatureBuffer.size() >= (m_pimpl->historyLength))
    {
        double sum = std::accumulate(m_pimpl->motorTemperatureBuffer.begin(),
                                     m_pimpl->motorTemperatureBuffer.end(),
                                     0.0);
        double mean = sum / m_pimpl->motorTemperatureBuffer.size();

        double sqDiffSum = 0.0;
        for (double x : m_pimpl->motorTemperatureBuffer) {
            double diff = x - mean;
            sqDiffSum += diff * diff;
        }

        double variance = sqDiffSum / m_pimpl->motorTemperatureBuffer.size();
        double stdDev = std::sqrt(variance) + 1e-1;

        // Define the threshold for outlier detection (e.g., 3 standard deviations)
        double lowerBound = mean - 3.0 * stdDev;
        double upperBound = mean + 3.0 * stdDev;

        // We need to remove the last element from the buffer
        m_pimpl->motorTemperatureBuffer.pop_back();

        if (inputMotorTemperature < lowerBound || inputMotorTemperature > upperBound)
        {
            // Replace outlier with the last valid value
            adjustedMotorTemperature = m_pimpl->motorTemperatureBuffer.back();
        }
    }

    // Add the adjusted value to the buffer
    m_pimpl->motorTemperatureBuffer.push_back(adjustedMotorTemperature);

    // Fill the input
    // Copy the joint positions and then the motor positions in the
    // structured input without emptying the buffer
    // Use iterators to copy the data to the vector
    std::size_t index = 0;
    if (m_pimpl->inputType == 1 || m_pimpl->inputType == 2 || m_pimpl->inputType == 4)
    {
        std::copy(m_pimpl->motorVelocityBuffer.cbegin(),
                m_pimpl->motorVelocityBuffer.cend(),
                m_pimpl->structuredInput.rawData.begin() + index);
        index += m_pimpl->historyLength;
        std::copy(m_pimpl->jointVelocityBuffer.cbegin(),
                m_pimpl->jointVelocityBuffer.cend(),
                m_pimpl->structuredInput.rawData.begin() + index);
    }
    if (m_pimpl->inputType == 3 || m_pimpl->inputType == 4)
    {
        std::copy(m_pimpl->motorPositionBuffer.cbegin(),
                m_pimpl->motorPositionBuffer.cend(),
                m_pimpl->structuredInput.rawData.begin() + index);
        index += m_pimpl->historyLength;
        std::copy(m_pimpl->jointPositionBuffer.cbegin(),
                m_pimpl->jointPositionBuffer.cend(),
                m_pimpl->structuredInput.rawData.begin() + index);
        index += m_pimpl->historyLength;
    }
    if (m_pimpl->inputType == 1 || m_pimpl->inputType == 3 || m_pimpl->inputType == 4)
    {
        m_pimpl->structuredInput.rawData[index] = static_cast<float>(adjustedMotorTemperature);
        index += 1;
    }

    // perform the inference
    const char* inputNames[] = {"input"};
    const char* outputNames[] = {"output"};

    try
    {
        m_pimpl->session->Run(Ort::RunOptions(),
                            inputNames,
                            &(m_pimpl->structuredInput.tensor),
                            1,
                            outputNames,
                            &(m_pimpl->structuredOutput.tensor),
                            1);

    } catch (const Ort::Exception& e) {
        BipedalLocomotion::log()->error("Error during the inference: {}", e.what());
        return false;
    }

    // copy the output
    output = static_cast<double>(m_pimpl->structuredOutput.rawData[0]);

    return true;
}
