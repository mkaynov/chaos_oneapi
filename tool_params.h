#pragma once

#include <string>

#include "include/json/json.hpp"
#include <memory>

using json = nlohmann::json;

enum ToolID {
    UNDEFINED = -1,
    LYAPUNOV = 0,
    ANGLE = 2,
    KNEADING = 1,
};

class Parameters {
public:
	virtual void Print() const {};
	virtual json paramsToJson() const = 0;
};

class LyapunovParameters : public Parameters{
public:
	LyapunovParameters(const json& jsonConfig, const std::string& modelName);
    void Print() const override;
	json paramsToJson() const override;

    bool is_qr = true;
    bool is_var = true;
    int32_t lyapunov_exponent_num;
    int32_t num_GPUs;
    int32_t samples_num;
    double epsilon;
    double integration_step;
    double skip_time_attractor; // time to reach attractor
    double skip_time_slave_trajectory; // time to make ortonormalized vectors get correct direction
    double skip_time_normalization; // time to accumulate lyapunov vectors before make ortonormalization
    double total_time;
    std::vector<double> initial_state;
    std::vector<double> initial_model_params;
    std::vector<double> first_sweep_param;
    std::vector<double> second_sweep_param;
};

class KneadingsParameters: public Parameters {
public:
	KneadingsParameters(const json& jsonConfig, const std::string& modelName);
	void Print() const override;
	json paramsToJson() const override;

	bool is_dopri;
	int32_t sequence_len;
	int32_t symbols_to_skip;
	int32_t num_GPUs;
	double indent_from_init_point;
	double min_dopri_step_kneading;
	double integration_step;
	double max_dopri_step;
	double dopri_tolLocErr;
	double dopri_tolGlobErr;
	std::vector<double> initial_state;
	std::vector<double> initial_model_params;
	std::vector<double> first_sweep_param;
	std::vector<double> second_sweep_param;
};

struct ConfigFactory {
    static std::shared_ptr<Parameters> paramsFromConfigFile(ToolID toolId, const std::string& modelName);
    static std::shared_ptr<Parameters> paramsFromJson(ToolID toolId, const std::string& modelName, const json& jsonConfig);
};