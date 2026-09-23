#include "sequence_contract.h"
#include "normalization.h"
#include <cassert>
#include <limits>
#include <iostream>
template<class F> void rejects(F f) { bool rejected=false; try { f(); } catch(const std::runtime_error&) { rejected=true; } assert(rejected); }
int main() {
    const float epsilon=1e-6f;
    assert(pulse::l2_squared_norm_floor(0,epsilon)==epsilon*epsilon);
    assert(pulse::l2_squared_norm_floor(1e-20f,epsilon)==epsilon*epsilon);
    assert(pulse::l2_squared_norm_floor(1e-8f,epsilon)==1e-8f);
    // A vector with norm 1e-10 scales to norm 1e-4, not one.
    const float scaled=1e-10f/std::sqrt(pulse::l2_squared_norm_floor(1e-20f,epsilon));
    assert(std::abs(scaled-1e-4f)<1e-10f);
    pulse::SequenceCursor first(2),second(2);
    first.check(9,10); first.advance();
    assert(first.position()==1 && second.position()==0);
    first.check(0,10); first.advance();
    rejects([&]{first.check(0,10);}); rejects([&]{first.advance();});
    first.reset(); assert(first.position()==0); first.check(0,10);
    rejects([&]{first.check(-1,10);}); rejects([&]{first.check(10,10);});
    assert(pulse::greedy_token({1,3,3,2})==1);
    rejects([]{pulse::greedy_token({});});
    rejects([]{pulse::greedy_token({std::numeric_limits<float>::quiet_NaN(),1});});
    std::cout << "sequence contract tests passed\n";
}
