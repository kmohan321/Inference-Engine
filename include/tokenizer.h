#include <string>
#include <vector>
#include <unordered_map>

struct PairHash {
    std::size_t operator()(const std::pair<std::string, std::string>& p) const {
        std::size_t h1 = std::hash<std::string>{}(p.first);
        std::size_t h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};


class Tokenizer {
private:

    std::unordered_map<std::string, int> vocab_to_id;
    std::unordered_map<int, std::string> id_to_vocab;
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merges;
    std::unordered_map<std::string, int> special_tokens;

    std::unordered_map<uint8_t, std::string> byte_to_unicode_map;
    std::unordered_map<std::string, uint8_t> unicode_to_byte_map;

    void init_byte_to_unicode();

    std::vector<std::string> bpe_merge(std::vector<std::string> symbols);

public:
    Tokenizer();

    bool load(const std::string& json_path);
    std::vector<int> encode(const std::string& text);
    std::string decode(int token_id);
    std::string decode(const std::vector<int>& ids);
};