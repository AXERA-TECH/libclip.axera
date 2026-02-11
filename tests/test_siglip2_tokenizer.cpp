#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include "tokenizer/tokenizer.hpp"

int main(int argc, char *argv[])
{
    std::string tokenizer_path = "../tests/tokenizer/siglip2_tokenizer.txt";
    
    std::cout << "Loading SigLIP2 tokenizer from: " << tokenizer_path << std::endl;
    
    std::ifstream fs(tokenizer_path);
    if (!fs.good())
    {
        std::cerr << "Failed to open tokenizer file: " << tokenizer_path << std::endl;
        return -1;
    }
    fs.close();

    std::unique_ptr<MNN::Transformer::Tokenizer> tokenizer;
    tokenizer.reset(MNN::Transformer::Tokenizer::createTokenizer(tokenizer_path));
    
    if (!tokenizer)
    {
        std::cerr << "Failed to create tokenizer" << std::endl;
        return -1;
    }
    
    std::cout << "Tokenizer loaded successfully!" << std::endl;

    // Test special tokens
    std::cout << "\nTesting special tokens:" << std::endl;
    
    std::vector<std::string> special_tokens = {"<pad>", "<eos>", "<bos>", "<unk>"};
    for (const auto &token : special_tokens)
    {
        auto ids = tokenizer->encode(token);
        std::cout << "  " << token << " -> ";
        for (auto id : ids)
        {
            std::cout << id << " ";
        }
        std::cout << "(is_special: " << (tokenizer->is_special(ids[0]) ? "yes" : "no") << ")" << std::endl;
    }

    // Test text encoding
    std::cout << "\nTesting text encoding:" << std::endl;
    std::vector<std::string> texts = {
        "a photo of 2 cats",
        "a photo of 2 dogs",
        "hello world"
    };
    
    for (const auto &text : texts)
    {
        auto ids = tokenizer->encode(text);
        std::cout << "  \"" << text << "\" -> ";
        std::cout << "[";
        for (size_t i = 0; i < ids.size() && i < 10; i++)
        {
            if (i > 0) std::cout << ", ";
            std::cout << ids[i];
        }
        if (ids.size() > 10) std::cout << "...";
        std::cout << "] (length: " << ids.size() << ")" << std::endl;
    }

    // Test decode
    std::cout << "\nTesting decode:" << std::endl;
    std::vector<int> test_ids = {1, 2, 3, 4, 5}; // Some token ids
    for (auto id : test_ids)
    {
        std::string decoded = tokenizer->decode(id);
        std::cout << "  " << id << " -> \"" << decoded << "\"" << std::endl;
    }

    // Verify SigLIP2 detection logic
    std::cout << "\nVerifying SigLIP2 detection logic:" << std::endl;
    
    // Check empty string (bos for SigLIP2 is "")
    auto empty_ids = tokenizer->encode("");
    std::cout << "  Empty string encode: ";
    for (auto id : empty_ids) std::cout << id << " ";
    std::cout << std::endl;
    
    // Check if <eos> is special
    auto eos_ids = tokenizer->encode("<eos>");
    bool eos_is_special = tokenizer->is_special(eos_ids[0]);
    std::cout << "  <eos> is_special: " << (eos_is_special ? "yes" : "no") << std::endl;

    std::cout << "\nSigLIP2 tokenizer test completed!" << std::endl;
    return 0;
}
