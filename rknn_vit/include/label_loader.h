#include <string>
#include <unordered_map>
#include "rapidjson/document.h"

class LabelLoader {
public:
    LabelLoader(const std::string& json_path);
    std::string get_label(int id) const;
    
private:
    std::unordered_map<int, std::string> id2label_;
    void parse_json(const std::string& json_str);
};
