/**
 * @file NNCurrentEstimator.cpp
 * @authors Filippo Passerini
 * @copyright 2025 Istituto Italiano di Tecnologia (IIT). This software may be modified and
 * distributed under the terms of the BSD-3-Clause license.
 */

#include <memory>
#include <deque>
#include <string>
#include <cmath>

#include <Eigen/Dense>

// onnxruntime
#include <onnxruntime_cxx_api.h>

#include <BipedalLocomotion/ParametersHandler/IParametersHandler.h>
#include <BipedalLocomotion/System/VariablesHandler.h>
#include <BipedalLocomotion/TextLogging/Logger.h>

#include <BipedalLocomotion/NNCurrentEstimator.h>


struct NNCurrentEstimator::Impl
{
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo;

    int m_modelNumber;  // Number extracted from model filename
    std::size_t m_inputCount;

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

NNCurrentEstimator::NNCurrentEstimator()
{
    m_pimpl = std::make_unique<NNCurrentEstimator::Impl>();
}

NNCurrentEstimator::~NNCurrentEstimator() = default;

bool NNCurrentEstimator::initialize(const std::string& networkModelPath,
                                       const std::size_t intraOpNumThreads,
                                       const std::size_t interOpNumThreads)
{
    // Extract model number from filename (e.g., "3_model.onnx" -> 3)
    size_t lastSlash = networkModelPath.find_last_of("/\\");
    std::string filename = networkModelPath.substr(lastSlash + 1);
    if (!filename.empty() && std::isdigit(filename[0])) {
        m_pimpl->m_modelNumber = filename[0] - '0';
    } else {
        BipedalLocomotion::log()->error("Model filename does not start with a number: {}", filename);
        return false;
    }

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

    // Get model input size
    std::vector<int64_t> inputShape = m_pimpl->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    m_pimpl->m_inputCount = inputShape[1];

    // format the input
    m_pimpl->structuredInput.rawData.resize(m_pimpl->m_inputCount);
    m_pimpl->structuredInput.shape[0] = 1; // batch
    m_pimpl->structuredInput.shape[1] = m_pimpl->m_inputCount;

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

void NNCurrentEstimator::resetEstimator()
{
    m_pimpl->m_inputCount = 0;
    // Clear input
    std::fill(m_pimpl->structuredInput.rawData.begin(),
                m_pimpl->structuredInput.rawData.end(), 0.0f);
}

bool NNCurrentEstimator::estimate(double inputJointPosition,
                                  double inputJointVelocity,
                                  double inputForce,
                                  double inputMotorCurrent,
                                  double inputDesiredMotorCurrent,
                                  double& output)
{
    // Compute the sign of desired current using a threshold
    constexpr double threshold = 0.015; // Amps
    int signDesiredCurrent = (std::abs(inputDesiredMotorCurrent) <= threshold) ? 
                            0 : (inputDesiredMotorCurrent > 0 ? 1 : -1);

    // Fill the input vector based on model number
    std::size_t index = 0;
    
    // Common inputs for all models
    m_pimpl->structuredInput.rawData[index++] = static_cast<float>(inputJointPosition);
    m_pimpl->structuredInput.rawData[index++] = static_cast<float>(inputJointVelocity);
    
    // Model specific inputs
    switch(m_pimpl->m_modelNumber) {
        case 2:
            // Model 2: position, velocity, force, measured current
            m_pimpl->structuredInput.rawData[index++] = static_cast<float>(inputForce);
            m_pimpl->structuredInput.rawData[index++] = static_cast<float>(inputMotorCurrent);
            break;
            
        case 3:
            // Model 3: position, velocity, signDesiredCurrent
            m_pimpl->structuredInput.rawData[index++] = static_cast<int>(signDesiredCurrent);
            break;
            
        case 4:
            // Model 4: position, velocity, measured current, signDesiredCurrent
            m_pimpl->structuredInput.rawData[index++] = static_cast<float>(inputMotorCurrent);
            m_pimpl->structuredInput.rawData[index++] = static_cast<int>(signDesiredCurrent);
            break;
            
        default:
            BipedalLocomotion::log()->error("Unsupported model number: {}", m_pimpl->m_modelNumber);
            return false;
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