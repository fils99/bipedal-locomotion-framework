/**
 * @file UnicycleTrajectoryPlanner.cpp
 * @authors Lorenzo Moretti, Diego Ferigo, Giulio Romualdi, Stefano Dafarra
 * @copyright 2021 Istituto Italiano di Tecnologia (IIT). This software may be modified and
 * distributed under the terms of the BSD-3-Clause license.
 */

#include <BipedalLocomotion/Contacts/ContactList.h>
#include <BipedalLocomotion/Contacts/ContactPhaseList.h>
#include <BipedalLocomotion/ContinuousDynamicalSystem/DynamicalSystem.h>
#include <BipedalLocomotion/ContinuousDynamicalSystem/LinearTimeInvariantSystem.h>
#include <BipedalLocomotion/ContinuousDynamicalSystem/RK4.h>
#include <BipedalLocomotion/Conversions/ManifConversions.h>
#include <BipedalLocomotion/Math/Constants.h>
#include <BipedalLocomotion/Planners/UnicycleTrajectoryPlanner.h>
#include <BipedalLocomotion/Planners/UnicycleUtilities.h>
#include <BipedalLocomotion/TextLogging/Logger.h>

#include <iDynTree/KinDynComputations.h>
#include <iDynTree/Model.h>

#include <chrono>
#include <mutex>
#include <string>
#include <vector>

using namespace BipedalLocomotion;

class Planners::UnicycleTrajectoryPlanner::Impl
{

public:
    enum class FSM
    {
        NotInitialized,
        Initialized,
        Running,
    };

    FSM state{FSM::NotInitialized};

    UnicycleTrajectoryPlannerOutput output;

    UnicycleTrajectoryPlannerInput input;

    UnicycleTrajectoryPlannerParameters parameters;

    UnicycleGenerator generator;

    std::mutex mutex;

    /*
    The CoM model is the Linear Inverted Pendulum Model, described by the equations:

           | xd  |   | -w  0  0  0  |   | x  |   | +w  0  0  0  |    | Xdcm  |
           | yd  | = |  0 -w  0  0  | * | y  | + |  0 +w  0  0  |  * | Ydcm  |
           | xdd |   |  0  0 -w  0  |   | xd |   |  0  0 +w  0  |    | Xdcmd |
           | ydd |   |  0  0  0 -w  |   | yd |   |  0  0  0 +w  |    | Xdcmd |

    where:
           {x,y} is the CoM planar position

           dcm is the Divergent Component of Motion

           w is the angular frequency of the Linear Inverted Pendulum, computed as sqrt(g/z), with z
           being the CoM constant height
    */
    struct COMSystem
    {
        std::shared_ptr<BipedalLocomotion::ContinuousDynamicalSystem::LinearTimeInvariantSystem>
            dynamics;
        std::shared_ptr<BipedalLocomotion::ContinuousDynamicalSystem::RK4<
            BipedalLocomotion::ContinuousDynamicalSystem::LinearTimeInvariantSystem>>
            integrator;
    };

    COMSystem comSystem;

    std::chrono::nanoseconds initTime{std::chrono::nanoseconds::zero()}; /**< init time of the
                                                                            trajectory generated by
                                                                            the planner */
    struct COMHeightTrajectory
    {
        std::vector<double> position, velocity, acceleration;
    };

    COMHeightTrajectory comHeightTrajectory;
};

BipedalLocomotion::Planners::UnicycleTrajectoryPlannerInput BipedalLocomotion::Planners::
    UnicycleTrajectoryPlannerInput::generateDummyUnicycleTrajectoryPlannerInput()
{
    UnicycleTrajectoryPlannerInput input;

    input.plannerInput = Eigen::VectorXd::Zero(3);

    iDynTree::Vector2 dcmInitialPosition, dcmInitialVelocity;
    dcmInitialPosition.zero();
    dcmInitialVelocity.zero();
    input.dcmInitialState.initialPosition = dcmInitialPosition;
    input.dcmInitialState.initialVelocity = dcmInitialVelocity;

    input.isLeftLastSwinging = false;

    input.initTime = std::chrono::nanoseconds::zero();

    input.measuredTransform = manif::SE3d::Identity();
    input.measuredTransform.translation(Eigen::Vector3d(0.0, -0.1, 0.0));

    return input;
}

bool Planners::UnicycleTrajectoryPlanner::setUnicycleControllerFromString(
    const std::string& unicycleControllerAsString, UnicycleController& unicycleController)
{
    if (unicycleControllerAsString == "personFollowing")
    {
        unicycleController = UnicycleController::PERSON_FOLLOWING;
    } else if (unicycleControllerAsString == "direct")
    {
        unicycleController = UnicycleController::DIRECT;
    } else
    {
        log()->error("[UnicycleTrajectoryPlanner::setUnicycleControllerFromString] Invalid "
                     "controller type.");
        return false;
    }

    return true;
}

Planners::UnicycleTrajectoryPlanner::UnicycleTrajectoryPlanner()
{
    m_pImpl = std::make_unique<UnicycleTrajectoryPlanner::Impl>();
}

Planners::UnicycleTrajectoryPlanner::~UnicycleTrajectoryPlanner() = default;

bool Planners::UnicycleTrajectoryPlanner::setRobotContactFrames(const iDynTree::Model& model)
{

    const auto logPrefix = "[UnicycleTrajectoryPlanner::setRobotContactFrames]";

    if (m_pImpl->state == Impl::FSM::NotInitialized)
    {
        log()->error("{} The Unicycle planner has not been initialized. Initialize it first.",
                     logPrefix);
        return false;
    }

    iDynTree::KinDynComputations kinDyn;

    if (!kinDyn.loadRobotModel(model))
    {
        log()->error("{} Unable to load the robot model.", logPrefix);
        m_pImpl->state = Impl::FSM::NotInitialized;
        return false;
    }

    m_pImpl->parameters.leftContactFrameIndex
        = kinDyn.model().getFrameIndex(m_pImpl->parameters.leftContactFrameName);
    if (m_pImpl->parameters.leftContactFrameIndex == iDynTree::FRAME_INVALID_INDEX)
    {
        log()->error("{} Unable to find the frame named {}.",
                     logPrefix,
                     m_pImpl->parameters.leftContactFrameName);
        m_pImpl->state = Impl::FSM::NotInitialized;
        return false;
    }

    m_pImpl->parameters.rightContactFrameIndex
        = kinDyn.model().getFrameIndex(m_pImpl->parameters.rightContactFrameName);
    if (m_pImpl->parameters.rightContactFrameIndex == iDynTree::FRAME_INVALID_INDEX)
    {
        log()->error("{} Unable to find the frame named {}.",
                     logPrefix,
                     m_pImpl->parameters.rightContactFrameName);
        m_pImpl->state = Impl::FSM::NotInitialized;
        return false;
    }

    return true;
}

bool Planners::UnicycleTrajectoryPlanner::initialize(
    std::weak_ptr<const ParametersHandler::IParametersHandler> handler)
{
    constexpr auto logPrefix = "[UnicycleTrajectoryPlanner::initialize]";

    auto ptr = handler.lock();

    if (ptr == nullptr)
    {
        log()->error("{} Invalid parameter handler.", logPrefix);
        return false;
    }

    // lambda function to parse parameters
    auto loadParam = [ptr, logPrefix](const std::string& paramName, auto& param) -> bool {
        if (!ptr->getParameter(paramName, param))
        {
            log()->error("{} Unable to get the parameter named '{}'.", logPrefix, paramName);
            return false;
        }
        return true;
    };

    // lambda function to parse parameters with fallback option
    auto loadParamWithFallback =
        [ptr, logPrefix](const std::string& paramName, auto& param, const auto& fallback) -> bool {
        if (!ptr->getParameter(paramName, param))
        {
            log()->info("{} Unable to find the parameter named '{}'. The default one with value "
                        "[{}] will be used.",
                        logPrefix,
                        paramName,
                        fallback);
            param = fallback;
        }
        return true;
    };

    // initialize parameters
    std::string unicycleControllerAsString;

    double unicycleGain;
    double slowWhenTurningGain;
    double slowWhenBackwardFactor;
    double slowWhenSidewaysFactor;

    double positionWeight;
    double timeWeight;

    std::string leftContactFrameName;
    std::string rightContactFrameName;

    double maxStepLength;
    double minStepLength;
    double maxLengthBackwardFactor;
    double minWidth;
    double minStepDuration;
    double maxStepDuration;
    double nominalDuration;
    double maxAngleVariation;
    double minAngleVariation;

    Eigen::Vector2d saturationFactors;

    bool startWithLeft{true};
    bool startWithSameFoot{true};
    bool terminalStep{true};

    Eigen::Vector2d mergePointRatios;
    double switchOverSwingRatio;
    double lastStepSwitchTime;
    bool isPauseActive{true};

    double comHeight;
    double comHeightDelta;
    Eigen::Vector2d leftZMPDelta;
    Eigen::Vector2d rightZMPDelta;
    double lastStepDCMOffset;

    // parse initialization parameters
    bool ok = true;

    ok = ok && loadParam("referencePosition", m_pImpl->parameters.referencePointDistance);
    ok = ok && loadParamWithFallback("controlType", unicycleControllerAsString, "direct");
    ok = ok && loadParamWithFallback("unicycleGain", unicycleGain, 10.0);
    ok = ok && loadParamWithFallback("slowWhenTurningGain", slowWhenTurningGain, 2.0);
    ok = ok && loadParamWithFallback("slowWhenBackwardFactor", slowWhenBackwardFactor, 0.4);
    ok = ok && loadParamWithFallback("slowWhenSidewaysFactor", slowWhenSidewaysFactor, 0.2);
    double dt, plannerHorizon;
    ok = ok && loadParamWithFallback("dt", dt, 0.002);
    m_pImpl->parameters.dt = std::chrono::nanoseconds(static_cast<int64_t>(dt * 1e9));
    ok = ok && loadParamWithFallback("plannerHorizon", plannerHorizon, 20.0);
    m_pImpl->parameters.plannerHorizon
        = std::chrono::nanoseconds(static_cast<int64_t>(plannerHorizon * 1e9));
    ok = ok && loadParamWithFallback("positionWeight", positionWeight, 1.0);
    ok = ok && loadParamWithFallback("timeWeight", timeWeight, 2.5);
    ok = ok && loadParamWithFallback("maxStepLength", maxStepLength, 0.32);
    ok = ok && loadParamWithFallback("minStepLength", minStepLength, 0.01);
    ok = ok && loadParamWithFallback("maxLengthBackwardFactor", maxLengthBackwardFactor, 0.8);
    ok = ok && loadParamWithFallback("nominalWidth", m_pImpl->parameters.nominalWidth, 0.20);
    ok = ok && loadParamWithFallback("minWidth", minWidth, 0.14);
    ok = ok && loadParamWithFallback("minStepDuration", minStepDuration, 0.65);
    ok = ok && loadParamWithFallback("maxStepDuration", maxStepDuration, 1.5);
    ok = ok && loadParamWithFallback("nominalDuration", nominalDuration, 0.8);
    ok = ok && loadParamWithFallback("maxAngleVariation", maxAngleVariation, 18.0);
    ok = ok && loadParamWithFallback("minAngleVariation", minAngleVariation, 5.0);
    ok = ok && loadParam("saturationFactors", saturationFactors);
    ok = ok
         && loadParamWithFallback("leftYawDeltaInDeg", m_pImpl->parameters.leftYawDeltaInRad, 0.0);
    ok = ok
         && loadParamWithFallback("rightYawDeltaInDeg",
                                  m_pImpl->parameters.rightYawDeltaInRad,
                                  0.0);
    m_pImpl->parameters.leftYawDeltaInRad
        = iDynTree::deg2rad(m_pImpl->parameters.leftYawDeltaInRad);
    m_pImpl->parameters.rightYawDeltaInRad
        = iDynTree::deg2rad(m_pImpl->parameters.rightYawDeltaInRad);
    ok = ok && loadParamWithFallback("swingLeft", startWithLeft, false);
    ok = ok && loadParamWithFallback("startAlwaysSameFoot", startWithSameFoot, true);
    ok = ok && loadParamWithFallback("terminalStep", terminalStep, true);
    ok = ok && loadParam("mergePointRatios", mergePointRatios);
    ok = ok && loadParamWithFallback("switchOverSwingRatio", switchOverSwingRatio, 0.2);
    ok = ok && loadParamWithFallback("lastStepSwitchTime", lastStepSwitchTime, 0.3);
    ok = ok && loadParamWithFallback("isPauseActive", isPauseActive, true);
    ok = ok && loadParamWithFallback("comHeight", comHeight, 0.70);
    ok = ok && loadParamWithFallback("comHeightDelta", comHeightDelta, 0.01);
    ok = ok && loadParam("leftZMPDelta", leftZMPDelta);
    ok = ok && loadParam("rightZMPDelta", rightZMPDelta);
    ok = ok && loadParamWithFallback("lastStepDCMOffset", lastStepDCMOffset, 0.5);
    ok = ok && loadParam("leftContactFrameName", m_pImpl->parameters.leftContactFrameName);
    ok = ok && loadParam("rightContactFrameName", m_pImpl->parameters.rightContactFrameName);

    // try to configure the planner
    auto unicyclePlanner = m_pImpl->generator.unicyclePlanner();

    ok = ok
         && unicyclePlanner->setDesiredPersonDistance(m_pImpl->parameters.referencePointDistance[0],
                                                      m_pImpl->parameters.referencePointDistance[1]);
    ok = ok && unicyclePlanner->setPersonFollowingControllerGain(unicycleGain);
    ok = ok && unicyclePlanner->setSlowWhenTurnGain(slowWhenTurningGain);
    ok = ok && unicyclePlanner->setSlowWhenBackwardFactor(slowWhenBackwardFactor);
    ok = ok && unicyclePlanner->setSlowWhenSidewaysFactor(slowWhenBackwardFactor);
    ok = ok && unicyclePlanner->setMaxStepLength(maxStepLength, maxLengthBackwardFactor);
    ok = ok && unicyclePlanner->setMaximumIntegratorStepSize(m_pImpl->parameters.dt.count() * 1e-9);
    ok = ok && unicyclePlanner->setWidthSetting(minWidth, m_pImpl->parameters.nominalWidth);
    ok = ok && unicyclePlanner->setMaxAngleVariation(maxAngleVariation);
    ok = ok && unicyclePlanner->setMinimumAngleForNewSteps(minAngleVariation);
    ok = ok && unicyclePlanner->setCostWeights(positionWeight, timeWeight);
    ok = ok && unicyclePlanner->setStepTimings(minStepDuration, maxStepDuration, nominalDuration);
    ok = ok && unicyclePlanner->setPlannerPeriod(m_pImpl->parameters.dt.count() * 1e-9);
    ok = ok && unicyclePlanner->setMinimumStepLength(minStepLength);
    ok = ok
         && unicyclePlanner->setSaturationsConservativeFactors(saturationFactors(0),
                                                               saturationFactors(1));
    unicyclePlanner->setLeftFootYawOffsetInRadians(m_pImpl->parameters.leftYawDeltaInRad);
    unicyclePlanner->setRightFootYawOffsetInRadians(m_pImpl->parameters.rightYawDeltaInRad);
    unicyclePlanner->addTerminalStep(terminalStep);
    unicyclePlanner->startWithLeft(startWithLeft);
    unicyclePlanner->resetStartingFootIfStill(startWithSameFoot);

    UnicycleController unicycleController;
    ok = ok && setUnicycleControllerFromString(unicycleControllerAsString, unicycleController);
    ok = ok && unicyclePlanner->setUnicycleController(unicycleController);

    ok = ok && m_pImpl->generator.setSwitchOverSwingRatio(switchOverSwingRatio);
    ok = ok && m_pImpl->generator.setTerminalHalfSwitchTime(lastStepSwitchTime);
    ok = ok && m_pImpl->generator.setPauseConditions(maxStepDuration, nominalDuration);
    ok = ok && m_pImpl->generator.setMergePointRatio(mergePointRatios[0], mergePointRatios[1]);

    m_pImpl->generator.setPauseActive(isPauseActive);

    auto comHeightGenerator = m_pImpl->generator.addCoMHeightTrajectoryGenerator();
    ok = ok && comHeightGenerator->setCoMHeightSettings(comHeight, comHeightDelta);

    auto dcmGenerator = m_pImpl->generator.addDCMTrajectoryGenerator();
    iDynTree::Vector2 leftZMPDeltaVec{leftZMPDelta};
    iDynTree::Vector2 rightZMPDeltaVec{rightZMPDelta};
    dcmGenerator->setFootOriginOffset(leftZMPDeltaVec, rightZMPDeltaVec);
    double omega = sqrt(BipedalLocomotion::Math::StandardAccelerationOfGravitation / comHeight);
    dcmGenerator->setOmega(omega);
    dcmGenerator->setFirstDCMTrajectoryMode(FirstDCMTrajectoryMode::FifthOrderPoly);
    ok = ok && dcmGenerator->setLastStepDCMOffsetPercentage(lastStepDCMOffset);

    // initialize the COM system
    m_pImpl->comSystem.dynamics
        = std::make_shared<BipedalLocomotion::ContinuousDynamicalSystem::LinearTimeInvariantSystem>();
    m_pImpl->comSystem.integrator
        = std::make_shared<BipedalLocomotion::ContinuousDynamicalSystem::RK4<
            BipedalLocomotion::ContinuousDynamicalSystem::LinearTimeInvariantSystem>>();

    // Set dynamical system matrices
    Eigen::Matrix4d A = -omega * Eigen::Matrix4d::Identity();
    Eigen::Matrix4d B = -A;
    ok = ok && m_pImpl->comSystem.dynamics->setSystemMatrices(A, B);
    // Set the initial state
    ok = ok && m_pImpl->comSystem.dynamics->setState({Eigen::Vector4d::Zero()});
    // Set the dynamical system to the integrator
    ok = ok && m_pImpl->comSystem.integrator->setDynamicalSystem(m_pImpl->comSystem.dynamics);
    ok = ok && m_pImpl->comSystem.integrator->setIntegrationStep(m_pImpl->parameters.dt);

    // generateFirstTrajectory;
    ok = ok && generateFirstTrajectory();

    // debug information
    auto leftSteps = m_pImpl->generator.getLeftFootPrint()->getSteps();

    for (const auto& step : leftSteps)
    {
        BipedalLocomotion::log()->debug("Left step at initialization: position: {}, angle: {}, "
                                        "impact time: {}",
                                        step.position.toString(),
                                        step.angle,
                                        step.impactTime);
    }

    auto rightSteps = m_pImpl->generator.getRightFootPrint()->getSteps();

    for (const auto& step : rightSteps)
    {
        BipedalLocomotion::log()->debug("Right step at initialization: position: {}, angle: {}, "
                                        "impact time: {}",
                                        step.position.toString(),
                                        step.angle,
                                        step.impactTime);
    }

    std::vector<StepPhase> leftPhases, rightPhases;
    m_pImpl->generator.getStepPhases(leftPhases, rightPhases);

    for (size_t i = 0; i < leftPhases.size(); i++)
    {
        BipedalLocomotion::log()->debug("Left phase at initialization: {}",
                                        static_cast<int>(leftPhases.at(i)));
    }

    for (size_t i = 0; i < rightPhases.size(); i++)
    {
        BipedalLocomotion::log()->debug("Right phase at initialization: {}",
                                        static_cast<int>(rightPhases.at(i)));
    }

    if (ok)
    {
        m_pImpl->state = Impl::FSM::Initialized;
    }

    return ok;
}

const Planners::UnicycleTrajectoryPlannerOutput&
Planners::UnicycleTrajectoryPlanner::getOutput() const
{
    constexpr auto logPrefix = "[UnicycleTrajectoryPlanner::getOutput]";

    std::lock_guard<std::mutex> lock(m_pImpl->mutex);

    return m_pImpl->output;
}

bool Planners::UnicycleTrajectoryPlanner::isOutputValid() const
{
    return m_pImpl->state == Impl::FSM::Running;
}

bool Planners::UnicycleTrajectoryPlanner::setInput(const UnicycleTrajectoryPlannerInput& input)
{
    constexpr auto logPrefix = "[UnicycleTrajectoryPlanner::setInput]";

    if (m_pImpl->state == Impl::FSM::NotInitialized)
    {
        log()->error("{} The Unicycle planner has never been initialized.", logPrefix);
        return false;
    }

    m_pImpl->input = input;

    return true;
}

bool Planners::UnicycleTrajectoryPlanner::advance()
{
    constexpr auto logPrefix = "[UnicycleTrajectoryPlanner::advance]";

    if (m_pImpl->state == Impl::FSM::NotInitialized)
    {
        log()->error("{} The Unicycle planner has never been initialized.", logPrefix);
        return false;
    }

    auto unicyclePlanner = m_pImpl->generator.unicyclePlanner();
    auto dcmGenerator = m_pImpl->generator.addDCMTrajectoryGenerator();

    double initTime{m_pImpl->input.initTime.count() * 1e-9};
    m_pImpl->initTime = m_pImpl->input.initTime;
    double dt{m_pImpl->parameters.dt.count() * 1e-9};

    // check if it is not the first run
    if (m_pImpl->state == Impl::FSM::Running)
    {
        bool correctLeft{!m_pImpl->input.isLeftLastSwinging};

        // compute end time of trajectory
        double endTime = initTime + m_pImpl->parameters.plannerHorizon.count() * 1e-9;

        // set desired point
        Eigen::Vector2d desiredPointInRelativeFrame, desiredPointInAbsoluteFrame;
        desiredPointInRelativeFrame(0) = m_pImpl->input.plannerInput(0);
        desiredPointInRelativeFrame(0) = m_pImpl->input.plannerInput(1);

        // left foot
        Eigen::Vector2d measuredPositionLeft;
        double measuredAngleLeft;
        double leftYawDeltaInRad;
        measuredPositionLeft(0) = m_pImpl->input.measuredTransform.x();
        measuredPositionLeft(1) = m_pImpl->input.measuredTransform.y();
        measuredAngleLeft
            = Conversions::toiDynTreeRot(m_pImpl->input.measuredTransform.asSO3()).asRPY()(2);
        leftYawDeltaInRad = m_pImpl->parameters.leftYawDeltaInRad;

        // right foot
        Eigen::Vector2d measuredPositionRight;
        double measuredAngleRight;
        double rightYawDeltaInRad;
        measuredPositionRight(0) = m_pImpl->input.measuredTransform.x();
        measuredPositionRight(1) = m_pImpl->input.measuredTransform.y();
        measuredAngleRight
            = Conversions::toiDynTreeRot(m_pImpl->input.measuredTransform.asSO3()).asRPY()(2);
        rightYawDeltaInRad = m_pImpl->parameters.rightYawDeltaInRad;

        // get unicycle pose
        double measuredAngle;
        measuredAngle = correctLeft ? measuredAngleLeft : measuredAngleRight;
        Eigen::Vector2d measuredPosition = correctLeft ? measuredPositionLeft
                                                       : measuredPositionRight;
        Eigen::Vector2d unicyclePositionFromStanceFoot, footPosition, unicyclePosition;
        unicyclePositionFromStanceFoot(0) = 0.0;

        Eigen::Matrix2d unicycleRotation;
        double unicycleAngle;

        if (correctLeft)
        {
            unicyclePositionFromStanceFoot(1) = -m_pImpl->parameters.nominalWidth / 2;
            unicycleAngle = measuredAngleLeft - leftYawDeltaInRad;
            footPosition = measuredPositionLeft;
        } else
        {
            unicyclePositionFromStanceFoot(1) = m_pImpl->parameters.nominalWidth / 2;
            unicycleAngle = measuredAngleRight - rightYawDeltaInRad;
            footPosition = measuredPositionRight;
        }

        double s_theta = std::sin(unicycleAngle);
        double c_theta = std::cos(unicycleAngle);

        unicycleRotation(0, 0) = c_theta;
        unicycleRotation(0, 1) = -s_theta;
        unicycleRotation(1, 0) = s_theta;
        unicycleRotation(1, 1) = c_theta;

        unicyclePosition = unicycleRotation * unicyclePositionFromStanceFoot + footPosition;

        // apply the homogeneous transformation w_H_{unicycle}
        desiredPointInAbsoluteFrame
            = unicycleRotation
                  * (m_pImpl->parameters.referencePointDistance + desiredPointInRelativeFrame)
              + unicyclePosition;

        // clear the old trajectory
        unicyclePlanner->clearPersonFollowingDesiredTrajectory();

        // add new point
        if (!unicyclePlanner
                 ->addPersonFollowingDesiredTrajectoryPoint(endTime,
                                                            iDynTree::Vector2(
                                                                desiredPointInAbsoluteFrame)))
        {
            log()->error("{} Error while setting the new reference.", logPrefix);
            return false;
        }

        // set the desired direct control
        unicyclePlanner->setDesiredDirectControl(m_pImpl->input.plannerInput(0),
                                                 m_pImpl->input.plannerInput(1),
                                                 m_pImpl->input.plannerInput(2));

        // set the initial state of the DCM trajectory generator

        if (!dcmGenerator->setDCMInitialState(m_pImpl->input.dcmInitialState))
        {
            log()->error("{} Failed to set the initial state.", logPrefix);
            return false;
        }

        // generate the new trajectory
        if (!(m_pImpl->generator.reGenerate(initTime,
                                            dt,
                                            endTime,
                                            correctLeft,
                                            iDynTree::Vector2(measuredPosition),
                                            measuredAngle)))
        {
            log()->error("{} Failed in computing new trajectory.", logPrefix);
            return false;
        }
    }

    // get the output
    std::lock_guard<std::mutex> lock(m_pImpl->mutex);

    // get the feet contact status
    m_pImpl->generator.getFeetStandingPeriods(m_pImpl->output.contactStatus.leftFootInContact,
                                              m_pImpl->output.contactStatus.rightFootInContact);

    m_pImpl->generator.getWhenUseLeftAsFixed(m_pImpl->output.contactStatus.UsedLeftAsFixed);

    // get the footsteps
    m_pImpl->output.steps.leftSteps = m_pImpl->generator.getLeftFootPrint()->getSteps();
    m_pImpl->output.steps.rightSteps = m_pImpl->generator.getRightFootPrint()->getSteps();

    // get the DCM trajectory
    auto convertToEigen
        = [](const std::vector<iDynTree::Vector2>& inputVect) -> std::vector<Eigen::Vector2d> {
        std::vector<Eigen::Vector2d> outputVect;
        outputVect.reserve(inputVect.size());

        for (const auto& v : inputVect)
        {
            outputVect.push_back(iDynTree::toEigen(v));
        };

        return outputVect;
    };

    m_pImpl->output.dcmTrajectory.position = convertToEigen(dcmGenerator->getDCMPosition());
    m_pImpl->output.dcmTrajectory.velocity = convertToEigen(dcmGenerator->getDCMVelocity());

    // get the CoM planar trajectory
    std::chrono::nanoseconds time = m_pImpl->input.initTime;
    Eigen::Vector4d state;
    state.head<2>() = m_pImpl->input.comInitialState.initialPlanarPosition;
    state.tail<2>() = m_pImpl->input.comInitialState.initialPlanarVelocity;
    m_pImpl->comSystem.dynamics->setState({state.head<4>()});
    using namespace BipedalLocomotion::GenericContainer::literals;
    auto stateDerivative = BipedalLocomotion::GenericContainer::make_named_tuple(
        BipedalLocomotion::GenericContainer::named_param<"dx"_h, Eigen::VectorXd>());
    Eigen::Vector4d controlInput;

    m_pImpl->output.comTrajectory.position.resize(m_pImpl->output.dcmTrajectory.position.size());
    m_pImpl->output.comTrajectory.velocity.resize(m_pImpl->output.dcmTrajectory.position.size());
    m_pImpl->output.comTrajectory.acceleration.resize(
        m_pImpl->output.dcmTrajectory.position.size());

    for (size_t i = 0; i < m_pImpl->output.dcmTrajectory.position.size(); i++)
    {
        // populate CoM planar position
        m_pImpl->output.comTrajectory.position[i].head<2>() = state.head<2>();

        // set the control input, u
        controlInput << m_pImpl->output.dcmTrajectory.position.at(i),
            m_pImpl->output.dcmTrajectory.velocity.at(i);
        m_pImpl->comSystem.dynamics->setControlInput({controlInput});

        // compute the state derivative xdot = Ax + Bu
        m_pImpl->comSystem.dynamics->dynamics(time, stateDerivative);

        // populate CoM planar velocity and acceleration
        m_pImpl->output.comTrajectory.acceleration[i].head<2>()
            = stateDerivative.get_from_hash<"dx"_h>().tail<2>();
        m_pImpl->output.comTrajectory.velocity[i].head<2>()
            = stateDerivative.get_from_hash<"dx"_h>().head<2>();

        // advance the integrator for one step
        m_pImpl->comSystem.integrator->oneStepIntegration(time, m_pImpl->parameters.dt);
        state.head<4>() = std::get<0>(m_pImpl->comSystem.integrator->getSolution());

        // update the system state
        m_pImpl->comSystem.dynamics->setState({state});
        time += m_pImpl->parameters.dt;
    }

    // get the CoM height trajectory
    auto comHeightGenerator = m_pImpl->generator.addCoMHeightTrajectoryGenerator();
    comHeightGenerator->getCoMHeightTrajectory(m_pImpl->comHeightTrajectory.position);
    comHeightGenerator->getCoMHeightVelocity(m_pImpl->comHeightTrajectory.velocity);
    comHeightGenerator->getCoMHeightAccelerationProfile(m_pImpl->comHeightTrajectory.acceleration);

    // stack the CoM planar and the height trajectory
    for (size_t i = 0; i < m_pImpl->output.comTrajectory.position.size(); i++)
    {
        m_pImpl->output.comTrajectory.position[i].z() = m_pImpl->comHeightTrajectory.position[i];
        m_pImpl->output.comTrajectory.velocity[i].z() = m_pImpl->comHeightTrajectory.velocity[i];
        m_pImpl->output.comTrajectory.acceleration[i].z()
            = m_pImpl->comHeightTrajectory.acceleration[i];
    }

    // get the merge points
    m_pImpl->generator.getMergePoints(m_pImpl->output.mergePoints);

    m_pImpl->state = Impl::FSM::Running;

    return true;
}

bool BipedalLocomotion::Planners::UnicycleTrajectoryPlanner::generateFirstTrajectory()
{

    constexpr auto logPrefix = "[UnicycleTrajectoryPlanner::generateFirstTrajectory]";

    // clear the all trajectory
    auto unicyclePlanner = m_pImpl->generator.unicyclePlanner();
    unicyclePlanner->clearPersonFollowingDesiredTrajectory();
    unicyclePlanner->setDesiredDirectControl(0.0, 0.0, 0.0);

    // clear left and right footsteps
    m_pImpl->generator.getLeftFootPrint()->clearSteps();
    m_pImpl->generator.getRightFootPrint()->clearSteps();

    // set initial and final times
    double initTime = 0;
    double endTime = initTime + m_pImpl->parameters.plannerHorizon.count() * 1e-9;
    double dt = m_pImpl->parameters.dt.count() * 1e-9;

    // at the beginning ergoCub has to stop
    Eigen::Vector2d m_personFollowingDesiredPoint;
    m_personFollowingDesiredPoint(0) = m_pImpl->parameters.referencePointDistance(0);
    m_personFollowingDesiredPoint(1) = m_pImpl->parameters.referencePointDistance(1);

    // add the initial point
    if (!unicyclePlanner
             ->addPersonFollowingDesiredTrajectoryPoint(initTime,
                                                        iDynTree::Vector2(
                                                            m_personFollowingDesiredPoint)))
    {
        log()->error("{} Error while setting the initial point.", logPrefix);
        return false;
    }

    // add the final point
    if (!unicyclePlanner
             ->addPersonFollowingDesiredTrajectoryPoint(endTime,
                                                        iDynTree::Vector2(
                                                            m_personFollowingDesiredPoint)))
    {
        log()->error("{} Error while setting the final point.", logPrefix);
        return false;
    }

    // generate the first trajectories
    if (!m_pImpl->generator.generate(initTime, dt, endTime))
    {

        log()->error("{} Error while computing the first trajectories.", logPrefix);

        return false;
    }

    return true;
}

Contacts::ContactPhaseList
BipedalLocomotion::Planners::UnicycleTrajectoryPlanner::getContactPhaseList()
{
    constexpr auto logPrefix = "[UnicycleTrajectoryPlanner::getContactPhaseList]";

    Contacts::ContactPhaseList contactPhaseList;

    if (isOutputValid() == false)
    {
        log()->error("{} The output is not valid. Returning an empty Contact Phase List.",
                     logPrefix);
        return contactPhaseList;
    }

    // get the contact phase lists
    BipedalLocomotion::Contacts::ContactListMap ContactListMap;
    std::vector<StepPhase> leftStepPhases, rightStepPhases;
    m_pImpl->generator.getStepPhases(leftStepPhases, rightStepPhases);

    BipedalLocomotion::Contacts::ContactList leftContactList, rightContactList;

    if (!Planners::UnicycleUtilities::getContactList(m_pImpl->initTime,
                                                     m_pImpl->parameters.dt,
                                                     m_pImpl->output.contactStatus.leftFootInContact,
                                                     m_pImpl->output.steps.leftSteps,
                                                     m_pImpl->parameters.leftContactFrameIndex,
                                                     "left_foot",
                                                     leftContactList))
    {
        log()->error("{} Error while getting the left contact list. Returning an empty Contact "
                     "Phase List.",
                     logPrefix);
        return contactPhaseList;
    };

    if (!Planners::UnicycleUtilities::getContactList(m_pImpl->initTime,
                                                     m_pImpl->parameters.dt,
                                                     m_pImpl->output.contactStatus
                                                         .rightFootInContact,
                                                     m_pImpl->output.steps.rightSteps,
                                                     m_pImpl->parameters.rightContactFrameIndex,
                                                     "right_foot",
                                                     rightContactList))
    {
        log()->error("{} Error while getting the right contact list. Returning an empty Contact "
                     "Phase List.",
                     logPrefix);
        return contactPhaseList;
    };

    ContactListMap["left_foot"] = leftContactList;
    ContactListMap["right_foot"] = rightContactList;
    contactPhaseList.setLists(ContactListMap);

    return contactPhaseList;
};