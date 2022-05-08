
#include"fileHandler.h"
#include "model_factory.h"

std::string GetToolName(int tool_id) {
	std::string tool_name("Unknown");

	switch (tool_id)
	{
	case 0:
	{
		tool_name = "Lyapunov";
		break;
	}
	case 1:
	{
		tool_name = "Kneadings";
		break;
	}
	case 2:
	{
		tool_name = "Angle";
		break;
	}
	case 3:
	{
		tool_name = "minDistance";
		break;
	}
	}

	return tool_name;
}

int32_t
saveCalculatedData(ToolID tool_id, const std::string& modelName, const Parameters *params, const Task* task) {
    if (!task->mResultsStr.empty()) {
        auto a = task->mResultsStr.size();
        auto b = task->mResults.size();
        assert(task->mResultsStr.size() == task->mResults.size());
    }
    assert(params != nullptr);
    json outputData;
    outputData["Parameters"] = params->paramsToJson();
	std::string tool_name = GetToolName(tool_id);

    outputData["ID_Tool"] = tool_name;

    std::string data_str;
    data_str.reserve(task->mResults.size() * task->mResults.at(0).size() * 9);
    size_t row_cnt = 0;
    for (auto &row : task->mResults) {
        for (auto &val : row) {
            data_str.append(std::to_string(val));
            data_str.append(" ");
        }
        if (!task->mResultsStr.empty()) {
            data_str.append(task->mResultsStr.at(row_cnt));
			data_str.append(" ");
        }
        data_str.at(data_str.size() - 1) = '\n';
        row_cnt += 1;
    }
    outputData["_Data"] = data_str;

    ModelFactoryRegistry::Storage & storage = Singleton<ModelFactoryRegistry::Storage>::Instance();
    json modelData;
    const auto& current_model = storage._data.at(modelName);

    modelData["Name"] = current_model._model_name;
    modelData["Dimension"] = current_model._dimension;
    modelData["Number of parameters"] = current_model._param_number;
    modelData["Name of parameters"] = current_model._model_param;
    modelData["Name of phase"] = current_model._model_phase;
    modelData["Equations"] = current_model._model_equat;
    modelData["systemType"] = current_model._type;

    outputData["Model"] = modelData;

    try {
        std::ofstream jsonOStream("Data/" + tool_name + " " + current_model._model_name + ".json");
        jsonOStream << std::setw(4) << outputData << std::endl;
    }
    catch (const std::exception & e) {
        std::cout << "Unable to write the file: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}

int32_t save(std::string sysName, int32_t dimension, double* initVals, int32_t paramsNum, double* params, int32_t numOfIterations, double* trajectory, std::string fName) {
	json trajectoryJSON;

	trajectoryJSON["System"] = sysName;
	trajectoryJSON["Dimension"] = dimension;
	trajectoryJSON["Params"] = std::vector<double>(params, params + paramsNum);
	trajectoryJSON["Initial value"] = std::vector<double>(initVals, initVals + dimension);
	trajectoryJSON["Trajectory"] = std::vector<double>(trajectory, trajectory + dimension * numOfIterations);

	try {
		std::ofstream jsonOStream(fName + ".json");
		jsonOStream << std::setw(4) << trajectoryJSON << std::endl;
		return 0;
	}
	catch (const std::exception & e) {
		std::cout << "Unable to write the file: " << e.what() << std::endl;
		return -1;
	}
}

int32_t save(std::map < std::vector<double>, std::string > kneadingDiagram, std::string fName) {
	json kneadingDiagramJSON;
	kneadingDiagramJSON = kneadingDiagram;
	try {
		std::ofstream jsonOStream(fName + ".json");
		jsonOStream << std::setw(4) << kneadingDiagramJSON << std::endl;
		return 0;
	}
	catch (const std::exception & e) {
		std::cout << "Unable to write the file: " << e.what() << std::endl;
		return -1;
	}
}

int32_t save(std::map < std::vector<double>, std::vector<double> > hamNFCodes, std::string fName) {
	json hamNFCodesJSON;
	hamNFCodesJSON = hamNFCodes;
	try {
		std::ofstream jsonOStream(fName + ".json");
		jsonOStream << std::setw(4) << hamNFCodesJSON << std::endl;
		return 0;
	}
	catch (const std::exception & e) {
		std::cout << "Unable to write the file: " << e.what() << std::endl;
		return -1;
	}
}

int32_t save(std::map < std::vector<double>, double > codes, std::string fName) {
	json codesJSON;
	codesJSON = codes;
	try {
		std::ofstream jsonOStream(fName + ".json");
		jsonOStream << std::setw(4) << codesJSON << std::endl;
		return 0;
	}
	catch (const std::exception & e) {
		std::cout << "Unable to write the file: " << e.what() << std::endl;
		return -1;
	}
}


int32_t save(std::string sysName,  std::map < std::vector<double>, double > codes, std::string fName) {
	json codesJSON;
	codesJSON["SysName"] = sysName;
	codesJSON["Kneadings"] = codes;
	try {
		std::ofstream jsonOStream(fName + ".json");
		jsonOStream << std::setw(4) << codesJSON << std::endl;
		return 0;
	}
	catch (const std::exception & e) {
		std::cout << "Unable to write the file: " << e.what() << std::endl;
		return -1;
	}
}
