rm -rf ./build-*

meson setup build-0 -Doptimization=3 -Dbuildtype=release -Dusm=true -Dua=true -Doffload=nvptx --native-file=./platforms/clang-21.ini
meson compile -C build-0

meson setup build-1 -Doptimization=3 -Dbuildtype=release -Dusm=true -Dua=true -Doffload=nvptx-cuda --native-file=./platforms/clang-21.ini
meson compile -C build-1

meson setup build-2 -Doptimization=3 -Dbuildtype=release -Dusm=true -Dua=true -Doffload=nvptx --native-file=./platforms/gcc-15.ini
meson compile -C build-2

./build-0/test-1

./build-1/test-1

./build-2/test-1