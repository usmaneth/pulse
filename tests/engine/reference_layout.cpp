#include "reference_layout.h"
#include <cassert>
#include <iostream>
int main() {
    // Two tokens share a Q/K parent. Each key view skips the next query.
    std::vector<float> parent(16);
    for (size_t i=0; i<parent.size(); ++i) parent[i]=float(i);
    int64_t ne[4]={2,2,2,1};
    size_t nb[4]={4,8,32,64};
    auto p=pulse::pack_reference(reinterpret_cast<uint8_t*>(parent.data()+4),
                                12*4,ne,nb,4);
    float values[8]; std::memcpy(values,p.data(),sizeof values);
    for(int i=0;i<4;++i) { assert(values[i]==i+4); assert(values[i+4]==i+12); }
    // Transposition and singleton strides also preserve logical order.
    int64_t nt[4]={2,2,1,1}; size_t bt[4]={8,4,99,999};
    p=pulse::pack_reference(reinterpret_cast<uint8_t*>(parent.data()),64,nt,bt,4);
    std::memcpy(values,p.data(),16);
    assert(values[0]==0 && values[1]==2 && values[2]==1 && values[3]==3);
    bool rejected=false;
    try { pulse::pack_reference(reinterpret_cast<uint8_t*>(parent.data()),4,ne,nb,4); }
    catch(const std::runtime_error&) { rejected=true; }
    assert(rejected);
    std::cout << "reference layout tests passed\n";
}
