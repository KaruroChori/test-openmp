#include <cassert>
#include <omp.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iostream>

#ifdef TEST_USM
    #pragma omp requires unified_shared_memory
#endif

#ifdef TEST_UA
    #pragma omp requires unified_address
#endif

double speed_measurement (void (*operation)(size_t,std::vector<double>&, std::vector<double>&, std::vector<double>&), size_t elements,std::vector<double>& v1, std::vector<double>& v2, std::vector<double>&v3 )
{
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    double k[10];
    #pragma omp target teams distribute
    for(size_t i=0;i<10;i++)
    {
        k[i]=i+1;
    }

    auto t1 = high_resolution_clock::now();
    operation(elements,v1,v2,v3);
    auto t2 = high_resolution_clock::now();

    duration<double, std::milli> ms_double = t2 - t1;

    return ms_double.count();
}

double speed_measurement2 (void (*operation)(size_t,double*, double*, double*), size_t elements,std::vector<double>& v1, std::vector<double>& v2, std::vector<double>&v3 )
{
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    double* v1d=v1.data(),*v2d=v2.data(),*v3d=v3.data();
    #pragma omp target enter data map (to: v1d[0:elements], v2d[0:elements])
    #pragma omp target enter data map (alloc: v3d[0:elements])
    auto t1 = high_resolution_clock::now();
    operation(elements,v1d,v2d,v3d);
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    return ms_double.count();
    #pragma omp target exit data map (from: v3d[0:elements], v2d[0:elements], v1d[0:elements] )
}

void operation_gpu_without_mapping_with_pointers(size_t elements, double* v1, double* v2, double *v3)
{
    #pragma omp target teams distribute parallel for simd
    for(size_t i=1;i<elements;i++)
    {
       v3[i]=v1[i]+v2[i];
    }

    for(size_t i=1;i<elements;i++)
    {
       assert(v3[i]==v1[i]+v2[i]);
    }
}

void operation_gpu_without_mapping_with_pointers(size_t elements, std::vector<double>& v1, std::vector<double>& v2, std::vector<double>&v3)
{
    double* v1d=v1.data(),*v2d=v2.data(),*v3d=v3.data();
    #pragma omp target teams distribute parallel for simd
    for(size_t i=1;i<elements;i++)
    {
        v3d[i]=v1d[i]+v2d[i];
    }

    for(size_t i=1;i<elements;i++)
    {
       assert(v3[i]==v1[i]+v2[i]);
    }
}



void operation_gpu_with_mapping_with_pointers(size_t elements, std::vector<double>& v1, std::vector<double>& v2, std::vector<double>&v3)
{
    double* v1d=v1.data(),*v2d=v2.data(),*v3d=v3.data();

    #pragma omp target enter data map (to: v1d[0:elements]) nowait
    #pragma omp target enter data map (to: v2d[0:elements]) nowait
    #pragma omp target enter data map (alloc: v3d[0:elements] ) nowait

    #pragma omp taskwait

    #pragma omp target teams distribute parallel for simd
    for(size_t i=1;i<elements;i++)
    {
       v3d[i]=v1d[i]+v2d[i];
    }

    #pragma omp target exit data map (from: v3d[0:elements] )

    for(size_t i=1;i<elements;i++)
    {
       assert(v3[i]==v1[i]+v2[i]);
    }
}

void operation_gpu_without_mapping_with_vectors(size_t elements, std::vector<double>& v1, std::vector<double>& v2, std::vector<double>&v3)
{
    #pragma omp target teams distribute parallel for simd
    for(size_t i=1;i<elements;i++)
    {
       v3[i]=v1[i]+v2[i];
    }

    for(size_t i=1;i<elements;i++)
    {
       assert(v3[i]==v1[i]+v2[i]);
    }
}


void operation_cpu_with_pointers(size_t elements, std::vector<double>& v1, std::vector<double>& v2, std::vector<double>&v3)
{
    double* v1d=v1.data(),*v2d=v2.data(),*v3d=v3.data();

    #pragma omp  parallel for simd
    for(size_t i=1;i<elements;i++)
    {
       v3d[i]=v1d[i]+v2d[i];
    }

    for(size_t i=1;i<elements;i++)
    {
       assert(v3[i]==v1[i]+v2[i]);
    }
}

void operation_cpu_with_vectors(size_t elements, std::vector<double>& v1, std::vector<double>& v2, std::vector<double>&v3)
{
    #pragma omp parallel for simd
    for(size_t i=1;i<elements;i++)
    {
       v3[i]=v1[i]+v2[i];
    }

    for(size_t i=1;i<elements;i++)
    {
       assert(v3[i]==v1[i]+v2[i]);
    }
}

void filldata_on_cpu(size_t elements,std::vector<double>&v1,std::vector<double>&v2,std::vector<double>&v3)
{
    #pragma omp parallel for simd
    for(size_t i=1;i<elements;i++)
    {
        v1[i]=rand()%10;
        v2[i]=rand()%10;
        v3[i]=0;
    }

}

int main()
{
    size_t elements=10000000;

    std::vector<double> v1a(elements),v2a(elements), v3a(elements);
    std::vector<double> v1b(elements),v2b(elements), v3b(elements);
    std::vector<double> v1c(elements),v2c(elements), v3c(elements);
    std::vector<double> v1d(elements),v2d(elements), v3d(elements);
    std::vector<double> v1e(elements),v2e(elements), v3e(elements);
    std::vector<double> v1f(elements),v2f(elements), v3f(elements);

    filldata_on_cpu(elements,v1a,v2a,v3a);
    filldata_on_cpu(elements,v1b,v2b,v3b);
    filldata_on_cpu(elements,v1c,v2c,v3c);
    filldata_on_cpu(elements,v1d,v2d,v3d);
    filldata_on_cpu(elements,v1e,v2e,v3e);
    filldata_on_cpu(elements,v1f,v2f,v3f);

    double dur;
    
    dur=speed_measurement(operation_gpu_with_mapping_with_pointers, elements,v1a,v2a,v3a);
    std::cout<<"on GPU with mapping with pointers:"<<dur<<"\n";

    dur=speed_measurement(operation_gpu_without_mapping_with_vectors, elements,v1b,v2b,v3b);
    std::cout<<"on GPU without mapping with vectors:"<<dur<<"\n";

    dur=speed_measurement(operation_gpu_without_mapping_with_pointers, elements,v1c,v2c,v3c);
    std::cout<<"on GPU without mapping with pointers:"<<dur<<"\n";

    dur=speed_measurement2(operation_gpu_without_mapping_with_pointers, elements,v1f,v2f,v3f);
    std::cout<<"on GPU already mapped vector with pointers:"<<dur<<"\n";

    dur=speed_measurement(operation_cpu_with_pointers,elements,v1d,v2d,v3d);
    std::cout<<"on CPU with pointers:"<<dur<<"\n";

    dur=speed_measurement(operation_cpu_with_vectors,elements,v1e,v2e,v3e);
    std::cout<<"on CPU with vectors:"<<dur;

}