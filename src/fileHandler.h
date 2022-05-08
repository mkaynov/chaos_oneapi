// File andler saves and reads JSON files
//
// Semyon Malykh
// malykhsm@gmail.com
// Nizhniy Novgorod 2020

#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>

#include "include/json/json.hpp"

#include "consts.h"
#include "singleton.h"
#include "tool_params.h"
#include "Task.h"


using json = nlohmann::json;

int32_t
saveCalculatedData(ToolID tool_id, const std::string& modelName, const Parameters *params, const Task* taskr);

// Save one trajectory
int32_t save(std::string sysName, int32_t dimension, double* initVals, int32_t paramsNum, double* params, int32_t numOfIterations, double* trajectory, std::string fName);

// Save kneading diagram
int32_t save(std::map < std::vector<double>, std::string> kneadingDiagram, std::string fName);

//Save Hamiltonian Normal Form codes
int32_t save(std::map < std::vector<double>, std::vector<double> > hamNFCodes, std::string fName);

//Save Rossler system codes
int32_t save(std::map < std::vector<double>, double > codes, std::string fName);

// Save kneading diagram with name
int32_t save(std::string sysName, std::map < std::vector<double>, std::string> kneadingDiagram, std::string fName);