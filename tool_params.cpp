#include <fstream>
#include <iostream>
#include "tool_params.h"

void LyapunovParameters::Print() const {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Lyapunov's parameters: " << std::endl;
    std::cout << "Number of lyapunov's exponent: " << lyapunov_exponent_num << std::endl;
    std::cout << "Epsilon: " << epsilon << std::endl;
    std::cout << "Step of integration: " << integration_step << std::endl;
    std::cout << "Time skip to reach attractor: " << skip_time_attractor << std::endl;
    std::cout << "Time skip for complementary trajectories: " << skip_time_slave_trajectory << std::endl;
    std::cout << "Time skip for Lyapunov's normalization: " << skip_time_normalization << std::endl;
    std::cout << "Total time: " << total_time << std::endl;
    std::cout << "Samples num: " << samples_num << std::endl;
    std::cout << "Initial state: ";
    int count = 0;
    for (const auto &element : initial_state) {
        std::cout << " [" << count << "] = " << element << " ";
        count++;
    }
    std::cout << std::endl;

    std::cout << "Initial parameters of model: ";
    count = 0;
    for (const auto &element : initial_model_params) {
        std::cout << " [" << count << "] = " << element << " ";
        count++;
    }
    std::cout << std::endl;

    std::cout << "First sweep parameter: ";
    count = 0;
    for (const auto &element : first_sweep_param) {
        std::cout << " [" << count << "] = " << element << " ";
        count++;
    }
    std::cout << std::endl;

    std::cout << "Second sweep parameter: ";
    count = 0;
    for (const auto &element : second_sweep_param) {
        std::cout << " [" << count << "] = " << element << " ";
        count++;
    }
    std::cout << std::endl;
    std::cout << "Number of GPUs: " << num_GPUs << std::endl;
    std::cout << "Using QR: " << std::boolalpha << is_qr << std::endl;
    std::cout << "Using VAR: " << std::boolalpha << is_var << std::endl;
    std::cout << "************************************************************************" << std::endl << std::endl;
}

json LyapunovParameters::paramsToJson() const {
    json params;
    params["lyapunov_exponent_num"] = lyapunov_exponent_num;
    params["epsilon"] = epsilon;
    params["integration_step"] = integration_step;
    params["skip_time_attractor"] = skip_time_attractor;
    params["skip_time_slave_trajectory"] = skip_time_slave_trajectory;
    params["skip_time_normalization"] = skip_time_normalization;
    params["total_time"] = total_time;
    params["samples_num"] = samples_num;
    params["initial_state"] = initial_state;
    params["initial_model_params"] = initial_model_params;

    params["first_sweep_param"] = first_sweep_param;

    params["second_sweep_param"] = second_sweep_param;

    params["num_GPUs"] = num_GPUs;
    params["using_qr"] = is_qr;
    params["using_var"] = is_var;

    return params;
}

LyapunovParameters::LyapunovParameters(const json &jsonConfig, const std::string &modelName) {
    bool isParamsInFile = false;
    try {
        auto conf = jsonConfig.at("hardware");
        conf.at("num_GPUs").get_to(num_GPUs);
        conf = jsonConfig.at("tool").at("lyapunov");
        conf.at("samples_num").get_to(samples_num);
        conf.at("using_qr").get_to(is_qr);
        conf.at("using_var").get_to(is_var);

        conf = jsonConfig.at("systems");
        for (const auto &c: conf.items()) {
            if (c.value().at("model_name") != modelName) {
                continue;
            }
            isParamsInFile = true;

            c.value().at("lyapunov_exponent_num").get_to(lyapunov_exponent_num);
            c.value().at("epsilon").get_to(epsilon);
            c.value().at("integration_step").get_to(integration_step);
            c.value().at("skip_time_attractor").get_to(skip_time_attractor);
            c.value().at("skip_time_slave_trajectory").get_to(skip_time_slave_trajectory);
            c.value().at("skip_time_normalization").get_to(skip_time_normalization);
            c.value().at("total_time").get_to(total_time);
            c.value().at("initial_state").get_to(initial_state);
            c.value().at("initial_model_params").get_to(initial_model_params);
            c.value().at("first_sweep_param").get_to(first_sweep_param);
            c.value().at("second_sweep_param").get_to(second_sweep_param);
            break;
        }
    }
    catch (std::exception &e) {
        printf("error: model: %s config exception: %s\n", modelName.c_str(), e.what());
        throw std::runtime_error("config error");
    }
    if (!isParamsInFile) {
        printf("error: model: %s config not found\n", modelName.c_str());
        throw std::runtime_error("config error");
    }

    // Check that size of initial_model_params and second_sweep_param is equal 4
    if (first_sweep_param.size() != 4 || second_sweep_param.size() != 4) {
        printf("error: model: %s config sweep parameters size error\n", modelName.c_str());
        throw std::runtime_error("config error");
    }
}

json KneadingsParameters::paramsToJson() const {
    json params;

    params["is_dopri"] = is_dopri;
    params["sequence_len"] = sequence_len;
    params["integration_step"] = integration_step;
    params["symbols_to_skip"] = symbols_to_skip;
    params["indent_from_init_point"] = indent_from_init_point;
	params["min_dopri_step_kneading"] = min_dopri_step_kneading;
    params["max_dopri_step"] = max_dopri_step;
    params["dopri_tolLocErr"] = dopri_tolLocErr;
    params["dopri_tolGlobErr"] = dopri_tolGlobErr;
    params["initial_state"] = initial_state;
    params["initial_model_params"] = initial_model_params;
    params["first_sweep_param"] = first_sweep_param;
    params["second_sweep_param"] = second_sweep_param;
    params["num_GPUs"] = num_GPUs;

    return params;
}

void KneadingsParameters::Print() const {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Kneadings parameters: " << std::endl;
    std::cout << "Sequence length: " << sequence_len << std::endl;
    std::cout << "Number of symbols to skip: " << symbols_to_skip << std::endl;
    std::cout << "Step of integration: " << integration_step << std::endl;
    std::cout << "Initial state: ";
    int count = 0;
    for (const auto &element : initial_state) {
        std::cout << " [" << count << "] = " << element << " ";
        count++;
    }
    std::cout << std::endl;

    std::cout << "Initial parameters of model: ";
    count = 0;
    for (const auto &element : initial_model_params) {
        std::cout << " [" << count << "] = " << element << " ";
        count++;
    }
    std::cout << std::endl;

    std::cout << "First sweep parameter: ";
    count = 0;
    for (const auto &element : first_sweep_param) {
        std::cout << " [" << count << "] = " << element << " ";
        count++;
    }
    std::cout << std::endl;

    std::cout << "Second sweep parameter: ";
    count = 0;
    for (const auto &element : second_sweep_param) {
        std::cout << " [" << count << "] = " << element << " ";
        count++;
    }
    std::cout << std::endl;
    std::cout << "Number of GPUs: " << num_GPUs << std::endl;
    std::cout << "Using QR: " << std::boolalpha << is_dopri << std::endl;
    if (is_dopri) {
        std::cout << "Max step: " << max_dopri_step << std::endl;
		std::cout << "Min_dopri_step_kneading: " << min_dopri_step_kneading << std::endl;
        std::cout << "TolGlobErr: " << dopri_tolGlobErr << std::endl;
        std::cout << "TolLocErr: " << dopri_tolLocErr << std::endl;
    }
    std::cout << "************************************************************************" << std::endl << std::endl;
}

KneadingsParameters::KneadingsParameters(const json &jsonConfig, const std::string &modelName) {
    bool isModelFound = false;
    try {
        jsonConfig.at("hardware").at("num_GPUs").get_to(num_GPUs);

        auto conf = jsonConfig.at("systems");
        for (const auto &element : conf.items()) {
            if (element.value().at("model_name") != modelName) {
                continue;
            }
            isModelFound = true;
			element.value().at("general").at("integration_step").get_to(integration_step);
			element.value().at("general").at("initial_state").get_to(initial_state);
			element.value().at("general").at("initial_model_params").get_to(initial_model_params);
			element.value().at("general").at("first_sweep_param").get_to(first_sweep_param);
			element.value().at("general").at("second_sweep_param").get_to(second_sweep_param);

			element.value().at("kneading").at("is_dopri").get_to(is_dopri);
			element.value().at("kneading").at("sequence_len").get_to(sequence_len);
			element.value().at("kneading").at("symbols_to_skip").get_to(symbols_to_skip);
			element.value().at("kneading").at("indent_from_init_point").get_to(indent_from_init_point);
			element.value().at("kneading").at("min_dopri_step_kneading").get_to(min_dopri_step_kneading);
			element.value().at("kneading").at("max_dopri_step").get_to(max_dopri_step);
			element.value().at("kneading").at("dopri_tolLocErr").get_to(dopri_tolLocErr);
			element.value().at("kneading").at("dopri_tolGlobErr").get_to(dopri_tolGlobErr);

            break;
        }
    }
    catch (std::exception &e) {
        printf("error: model: %s config exception: %s\n", modelName.c_str(), e.what());
        throw std::runtime_error("config error");
    }
    if (!isModelFound) {
        printf("error: model: %s config not found\n", modelName.c_str());
        exit(-1);
    }

    // Check that size of initial_model_params and second_sweep_param is equal 4
    if (first_sweep_param.size() != 4 || second_sweep_param.size() != 4) {
        printf("error: model: %s config sweep parameters size error\n", modelName.c_str());
        throw std::runtime_error("config error");
    }
}

std::shared_ptr<Parameters> ConfigFactory::paramsFromConfigFile(ToolID toolId, const std::string &modelName) {
    json jsonConfig;
    std::string configPath;

    switch (toolId) {
        case ToolID::LYAPUNOV:
        case ToolID::ANGLE:
        case ToolID::KNEADING:
            configPath = "config.json";
            break;
        default:
            return nullptr;
    }
    std::ifstream config(configPath);
    if (!config.good()) {
        printf("error: config file: %s not found\n", configPath.c_str());
        return nullptr;
    }
    try {
        config >> jsonConfig;
    }
    catch (...) {
        printf("error: config file: %s parse failure\n", configPath.c_str());
        return nullptr;
    }

    switch (toolId) {
        case ToolID::LYAPUNOV:
        case ToolID::ANGLE:
            return std::make_shared<LyapunovParameters>(jsonConfig, modelName);
        case ToolID::KNEADING:
            return std::make_shared<KneadingsParameters>(jsonConfig, modelName);
        default:
            return nullptr;
    }
}

std::shared_ptr<Parameters>
ConfigFactory::paramsFromJson(ToolID toolId, const std::string &modelName, const json &jsonConfig) {
    switch (toolId) {
        case ToolID::LYAPUNOV:
        case ToolID::ANGLE:
            return std::make_shared<LyapunovParameters>(jsonConfig, modelName);
        case ToolID::KNEADING:
            return std::make_shared<KneadingsParameters>(jsonConfig, modelName);
        default:
            return nullptr;
    }
}
