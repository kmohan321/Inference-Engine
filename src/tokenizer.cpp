#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <utility>
#include <functional>
#include <regex>
#include "tokenizer.h"

#include <iostream>

void Tokenizer::init_byte_to_unicode() {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        // Ranges: '!' to '~' (33-126), '¡' to '¬' (161-172), '®' to 'ÿ' (174-255)
        bool is_printable = (b >= 33 && b <= 126) || 
                            (b >= 161 && b <= 172) || 
                            (b >= 174 && b <= 255);
        
        int unicode_code_point;
        
        if (is_printable) {
            unicode_code_point = b;
        } else {
            unicode_code_point = 256 + n;
            n++;
        }
        
        std::string utf8_str = "";
        
        if (unicode_code_point <= 127) {
            utf8_str += static_cast<char>(unicode_code_point);
        } else {
            utf8_str += static_cast<char>(0xC0 | (unicode_code_point >> 6));
            utf8_str += static_cast<char>(0x80 | (unicode_code_point & 0x3F));
        }
        
        byte_to_unicode_map[static_cast<uint8_t>(b)] = utf8_str;
        unicode_to_byte_map[utf8_str] = static_cast<uint8_t>(b);
    }
}

std::vector<std::string> Tokenizer::bpe_merge(std::vector<std::string> symbols) {

  if (symbols.size() < 2) {
        return symbols;
    }

  while (true) {
      int best_rank = 1e9; 
      int best_idx = -1;

      for (size_t i = 0; i < symbols.size() - 1; i++) {
          std::pair<std::string, std::string> current_pair = {symbols[i], symbols[i + 1]};
          auto it = merges.find(current_pair);
          
          if (it != merges.end()) {
              if (it->second < best_rank) {
                  best_rank = it->second;
                  best_idx = i;
              }
          }
      }

      if (best_idx == -1) {
          break; 
      }

      symbols[best_idx] = symbols[best_idx] + symbols[best_idx + 1];
      symbols.erase(symbols.begin() + best_idx + 1);
  }
  return symbols;

}

std::string Tokenizer::decode(int token_id) {
    auto it = id_to_vocab.find(token_id);
    if (it == id_to_vocab.end()) {
        return "";
    }

    std::string token_str = it->second;

    if (special_tokens.find(token_str) != special_tokens.end()) {
        return token_str;
    }

    std::string raw_bytes = "";
    size_t i = 0;
    while (i < token_str.length()) {
        unsigned char c = static_cast<unsigned char>(token_str[i]);
        size_t char_len = 1;

        if ((c & 0x80) == 0)        char_len = 1; // 1-byte ASCII
        else if ((c & 0xE0) == 0xC0) char_len = 2; // 2-byte UTF-8
        else if ((c & 0xF0) == 0xE0) char_len = 3; // 3-byte UTF-8
        else if ((c & 0xF8) == 0xF0) char_len = 4; // 4-byte UTF-8

        std::string utf8_char = token_str.substr(i, char_len);
        i += char_len;

        auto byte_it = unicode_to_byte_map.find(utf8_char);
        if (byte_it != unicode_to_byte_map.end()) {
            raw_bytes += static_cast<char>(byte_it->second);
        } else {
            raw_bytes += utf8_char;
        }
    }

    return raw_bytes;
}

std::string Tokenizer::decode(const std::vector<int>& ids) {
    std::string full_text = "";
    for (int id : ids) {
        full_text += decode(id);
    }
    return full_text;
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> token_ids;
    if (text.empty()) return token_ids;

    struct Segment {
        std::string content;
        bool is_special;
    };
    std::vector<Segment> segments;

    size_t pos = 0;
    while (pos < text.length()) {
        size_t earliest_pos = std::string::npos;
        std::string matched_special = "";

        for (const auto& kv : special_tokens) {
            size_t found = text.find(kv.first, pos);
            if (found != std::string::npos && found < earliest_pos) {
                earliest_pos = found;
                matched_special = kv.first;
            }
        }

        if (earliest_pos == std::string::npos) {
            segments.push_back({text.substr(pos), false});
            break;
        } else {
            if (earliest_pos > pos) {
                segments.push_back({text.substr(pos, earliest_pos - pos), false});
            }
            segments.push_back({matched_special, true});
            pos = earliest_pos + matched_special.length();
        }
    }

    auto process_chunk = [&](const std::string& chunk) {
        if (chunk.empty()) return;

        std::vector<std::string> symbols;
        for (char c : chunk) {
            uint8_t byte_val = static_cast<uint8_t>(c);
            symbols.push_back(byte_to_unicode_map[byte_val]);
        }

        std::vector<std::string> merged_symbols = bpe_merge(symbols);

        for (const std::string& sym : merged_symbols) {
            auto v_it = vocab_to_id.find(sym);
            if (v_it != vocab_to_id.end()) {
                token_ids.push_back(v_it->second);
            }
        }
    };

    std::regex pre_token_regex("(?:'s|'t|'re|'ve|'m|'ll|'d|'S|'T|'RE|'VE|'M|'LL|'D)"
        "|[a-zA-Z]+"
        "|[0-9]+"
        "| ?[^\\s a-zA-Z0-9]+[\\r\\n]*"
        "|\\s*[\\r\\n]+"
        "|\\s+");

    for (const auto& seg : segments) {
        if (seg.is_special) {
            token_ids.push_back(special_tokens[seg.content]);
            continue;
        }

        auto words_begin = std::sregex_iterator(seg.content.begin(), seg.content.end(), pre_token_regex);
        auto words_end = std::sregex_iterator();

        size_t last_pos = 0;
        for (std::sregex_iterator it = words_begin; it != words_end; ++it) {
            std::smatch match = *it;
            size_t match_pos = match.position();
            size_t match_len = match.length();

            if (match_pos > last_pos) {
                process_chunk(seg.content.substr(last_pos, match_pos - last_pos));
            }

            process_chunk(match.str());
            last_pos = match_pos + match_len;
        }

        if (last_pos < seg.content.length()) {
            process_chunk(seg.content.substr(last_pos));
        }
    }

    return token_ids;
}

Tokenizer::Tokenizer() {
    init_byte_to_unicode();
}

