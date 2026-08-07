#include <fstream>
#include <iostream>
#include <sstream>
#include "cJSON.h"
#include "tokenizer.h"


bool Tokenizer::load(const std::string& json_path) {
    // 1. Read the JSON file into a C++ string
    std::ifstream file(json_path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open tokenizer JSON file: " << json_path << std::endl;
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json_data = buffer.str();
    file.close();

    // 2. Parse the JSON using cJSON
    cJSON* root = cJSON_Parse(json_data.c_str());
    if (!root) {
        const char* error_ptr = cJSON_GetErrorPtr();
        if (error_ptr) {
            std::cerr << "cJSON Parse Error before: " << error_ptr << std::endl;
        }
        return false;
    }

    // =========================================================================
    // STEP A: Parse "added_tokens" (Special Tokens)
    // =========================================================================
    cJSON* added_tokens_arr = cJSON_GetObjectItemCaseSensitive(root, "added_tokens");
    if (cJSON_IsArray(added_tokens_arr)) {
        cJSON* token_item = NULL;
        cJSON_ArrayForEach(token_item, added_tokens_arr) {
            cJSON* id_obj = cJSON_GetObjectItemCaseSensitive(token_item, "id");
            cJSON* content_obj = cJSON_GetObjectItemCaseSensitive(token_item, "content");

            if (cJSON_IsNumber(id_obj) && cJSON_IsString(content_obj)) {
                int id = id_obj->valueint;
                std::string content = content_obj->valuestring;

                special_tokens[content] = id;
                vocab_to_id[content] = id;
                id_to_vocab[id] = content;
            }
        }
    }

    // =========================================================================
    // STEP B: Parse "model" -> "vocab" and "merges"
    // =========================================================================
    cJSON* model_obj = cJSON_GetObjectItemCaseSensitive(root, "model");
    if (cJSON_IsObject(model_obj)) {
        
        // --- B1: Parse Vocab ---
        cJSON* vocab_obj = cJSON_GetObjectItemCaseSensitive(model_obj, "vocab");
        if (cJSON_IsObject(vocab_obj)) {
            cJSON* vocab_item = NULL;
            cJSON_ArrayForEach(vocab_item, vocab_obj) {
                // In a JSON object, 'vocab_item->string' is the Key (token string)
                // 'vocab_item->valueint' is the Value (token ID)
                if (vocab_item->string && cJSON_IsNumber(vocab_item)) {
                    std::string token_str = vocab_item->string;
                    int id = vocab_item->valueint;

                    vocab_to_id[token_str] = id;
                    id_to_vocab[id] = token_str;
                }
            }
        }

        // --- B2: Parse Merges ---
        cJSON* merges_arr = cJSON_GetObjectItemCaseSensitive(model_obj, "merges");
        if (cJSON_IsArray(merges_arr)) {
            int rank = 0;
            cJSON* pair_item = NULL;

            cJSON_ArrayForEach(pair_item, merges_arr) {
                // Case 1: Merges is an array of 2-element arrays: [ ["Ġ", "Ġ"], ["ĠĠ", "ĠĠ"] ]
                if (cJSON_IsArray(pair_item) && cJSON_GetArraySize(pair_item) == 2) {
                    cJSON* part1 = cJSON_GetArrayItem(pair_item, 0);
                    cJSON* part2 = cJSON_GetArrayItem(pair_item, 1);

                    if (cJSON_IsString(part1) && cJSON_IsString(part2)) {
                        merges[{part1->valuestring, part2->valuestring}] = rank++;
                    }
                }
                // Case 2: Merges is an array of space-separated strings: [ "Ġ Ġ", "ĠĠ ĠĠ" ]
                else if (cJSON_IsString(pair_item)) {
                    std::string pair_str = pair_item->valuestring;
                    size_t space_pos = pair_str.find(' ');
                    if (space_pos != std::string::npos) {
                        std::string p1 = pair_str.substr(0, space_pos);
                        std::string p2 = pair_str.substr(space_pos + 1);
                        merges[{p1, p2}] = rank++;
                    }
                }
            }
        }
    }

    // 3. Free cJSON memory allocated tree
    cJSON_Delete(root);

    std::cout << "Successfully loaded Tokenizer:" << std::endl;
    std::cout << "  - Vocab size: " << vocab_to_id.size() << " tokens" << std::endl;
    std::cout << "  - Merges count: " << merges.size() << " pairs" << std::endl;
    std::cout << "  - Special tokens: " << special_tokens.size() << " tokens" << std::endl;

    return true;
}
