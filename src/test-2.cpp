#include <cassert>
#include <omp.h>
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#ifdef TEST_USM
    #pragma omp requires unified_shared_memory
#endif

#ifdef TEST_UA
    #pragma omp requires unified_address
#endif


int main(){
    constexpr int items = 100'000'000;
    std::vector<double> v0(items), v1(items), v2(items);

    for(int i = 0 ; i < items ; i++){
        v0[i]=(pow(i,2.0)/(((i>>i)%57181)+1));
        v1[i]=(pow(i,3.0)/(((i>>i)%169071)+1));
    }

    ankerl::nanobench::Bench().run("on cpu", [&] {
        double *v2d = v2.data(), *v1d = v1.data(), *v0d = v0.data();
        #pragma omp teams distribute parallel for
        for(int i = 0 ; i < items ; i++){
            v2d[i] = v0d[i] + v1d[i]; 
        }
        //ankerl::nanobench::doNotOptimizeAway(d);
    });

    assert(abs(v2[500]-(v0[500] + v1[500]))<0.001);

    ankerl::nanobench::Bench().run("on gpu", [&] {
        double *v2d = v2.data(), *v1d = v1.data(), *v0d = v0.data();
        #pragma omp target teams distribute parallel for
        for(int i = 0 ; i < items ; i++){
            v2d[i] = v0d[i] + v1d[i]; 
        }
        //ankerl::nanobench::doNotOptimizeAway(d);
    });

    assert(abs(v2[500]-(v0[500] + v1[500]))<0.001);

    ankerl::nanobench::Bench().run("on gpu", [&] {
        double *v2d = v2.data(), *v1d = v1.data(), *v0d = v0.data();
        #pragma omp target teams distribute parallel for
        for(int i = 0 ; i < items ; i++){
            v2d[i] = v0d[i] + v1d[i] + 1; 
        }
        //ankerl::nanobench::doNotOptimizeAway(d);
    });


    assert(abs(v2[500]-(v0[500] + v1[500] + 1))<0.001);

    return 0;
}