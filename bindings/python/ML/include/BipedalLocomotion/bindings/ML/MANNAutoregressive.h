/**
 * @file MANNAutoregressive.h
 * @authors Giulio Romualdi
 * @copyright 2023 Istituto Italiano di Tecnologia (IIT). This software may be modified and
 * distributed under the terms of the BSD-3-Clause license.
 */

#ifndef BIPEDAL_LOCOMOTION_BINDINGS_ML_MANN_TRAJECTORY_GENERATION_H
#define BIPEDAL_LOCOMOTION_BINDINGS_ML_MANN_TRAJECTORY_GENERATION_H

#include <pybind11/pybind11.h>

namespace BipedalLocomotion
{
namespace bindings
{
namespace ML
{

void CreateMANNAutoregressive(pybind11::module& module);

} // namespace ML
} // namespace bindings
} // namespace BipedalLocomotion

#endif // BIPEDAL_LOCOMOTION_BINDINGS_ML_MANN_TRAJECTORY_GENERATION_H
