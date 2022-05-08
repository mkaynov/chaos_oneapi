
#include <iostream>
#include "types.h"
#include "fileHandler.h"

#include <string>
#include "RaiiTimer.h"
#include "TaskLyapunov.h"


int main() {
    int model_id = 0;
    ModelFactoryRegistry::Storage &storage = Singleton<ModelFactoryRegistry::Storage>::Instance();

    std::cout << "Available models: " << std::endl;
    int count = 0;
    std::vector<std::string> model_names;
    model_names.reserve(storage._data.size());
    for (const auto &element : storage._data) {
        model_names.push_back(element.first);
        std::cout << "Model[" << count << "]: \"" << element.first << "\"";
        std::cout << "  Dimension: " << element.second._dimension;
        std::cout << " Number of parameters: " << element.second._param_number << std::endl;
        count++;
    }

    if (model_id > storage._data.size() - 1) {
        std::cout << "Number of model is incorrect: " << model_id << " Please try again" << std::endl;
        exit(1);
    }
    const auto &modelName = model_names.at(model_id);

    auto toolId = ToolID::LYAPUNOV;

    uint32_t t;
    auto timer = RaiiTimer(&t);

    std::vector<std::vector<double>> results;
    std::vector<std::string> results_str;
    int done = 0;
    auto params = ConfigFactory::paramsFromConfigFile(toolId, modelName);
    if (not params) {
        printf("error: config error\n");
        exit(1);
    }

    json json_config;
    std::ifstream lyapunov_config("config.json");
    if (!lyapunov_config.good()) {
        std::cout << "File with initial parameters does not exist";
        exit(1);
    }
    try {
        lyapunov_config >> json_config;
    }
    catch (...) {
        exit(1);
    }

    params->Print();

    std::unique_ptr<Task> task;
    if (toolId == ToolID::LYAPUNOV) {
        task = std::make_unique<TaskLyapunov>();
    } else {
        throw std::runtime_error("wrong tool");
    }

    task->init(params, modelName);
    task->allocate();
    task->execute();

    if (task->isReady()) {
        saveCalculatedData(toolId, modelName, params.get(), task.get());
    }
    task->clear();
    task->printTimeReport();

    return 0;
}
