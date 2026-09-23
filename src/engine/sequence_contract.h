#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace pulse {
class SequenceCursor {
    int size_, position_=0;
public:
    explicit SequenceCursor(int capacity):size_(capacity) {
        if(capacity<1) throw std::runtime_error("Invalid context capacity");
    }
    int position() const { return position_; }
    void check(int token,int vocabulary) const {
        if(position_>=size_) throw std::runtime_error("Sequence exceeds context capacity");
        if(token<0 || token>=vocabulary) throw std::runtime_error("Token is outside the vocabulary");
    }
    void advance() {
        if(position_>=size_) throw std::runtime_error("Sequence exceeds context capacity");
        ++position_;
    }
    void reset() { position_=0; }
};
inline int greedy_token(const std::vector<float>& logits) {
    if(logits.empty()) throw std::runtime_error("Empty output logits");
    for(float value:logits) if(!std::isfinite(value)) throw std::runtime_error("Nonfinite output logit");
    return int(std::max_element(logits.begin(),logits.end())-logits.begin());
}
}
