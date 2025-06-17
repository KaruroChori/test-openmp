project(
    'myproject',
    'cpp',
    default_options: ['cpp_std=c++23'],
    meson_version: '>= 1.3',
)

nanobench_dep = dependency( 'nanobench',  required: true)

if get_option('usm')
    add_project_arguments('-DUSE_USM', language: 'cpp')
endif

if get_option('ua')
    add_project_arguments('-DUSE_UA', language: 'cpp')
endif

cxx = meson.get_compiler('cpp')
openmp_compile_args = []
openmp_link_args = []

if get_option('offload') != ''
    add_project_arguments('-DOFFLOAD_TARGET=${quoted}', language: 'cpp')
    if get_option('offload') == 'nvptx'
        if  cxx.get_id() == 'clang'
            openmp_compile_args = [ '-fopenmp', '-g', '-fopenmp-targets=nvptx64', '-march=native', '-stdlib=libc++', '-Xoffload-linker', '-lc++', '-Xoffload-linker', '-lc++abi']
        elif cxx.get_id() == 'gcc'
            openmp_compile_args =[ '-fopenmp', '-g', '-foffload=nvptx-none', '-march=native']
        endif
    elif get_option('offload') == 'nvptx-cuda'
        if  cxx.get_id() == 'clang'
            openmp_compile_args = [ '-fopenmp', '-g', '-fopenmp-targets=nvptx64-nvidia-cuda', '-fopenmp-cuda-mode', '-march=native', '-stdlib=libc++', '-Xoffload-linker', '-lc++', '-Xoffload-linker', '-lc++abi']
        elif cxx.get_id() == 'gcc'
            error('Not supported nvptx-cuda with gcc')
        endif
    endif

endif

openmp_link_args = openmp_compile_args

executable(
    'test-1',
    './src/test-1.cpp',
    dependencies: [ nanobench_dep ],
    cpp_args: [openmp_compile_args],
    link_args: [openmp_link_args],
)


 executable(
    'test-2',
    './src/test-2.cpp',
    dependencies: [ nanobench_dep ],
    cpp_args: [openmp_compile_args],
    link_args: [openmp_link_args],
)

