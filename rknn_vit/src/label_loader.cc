#include "label_loader.h"
#include <fstream>
#include <sstream>
#include "rapidjson/istreamwrapper.h"

LabelLoader::LabelLoader(const std::string& json_path) {
    std::ifstream ifs(json_path);
    rapidjson::IStreamWrapper isw(ifs);
    rapidjson::Document doc;
    doc.ParseStream(isw);
    
    for (auto& m : doc.GetObject()) {
        int id = std::stoi(m.name.GetString());
        id2label_[id] = m.value.GetString();
    }
}

std::string LabelLoader::get_label(int id) const {
    auto it = id2label_.find(id);
    return it != id2label_.end() ? it->second : "";
}
