#/bin/bash

rm -rf ./build-*


LOGDIR="./logs/$(date +%Y-%m-%d)"
mkdir -p $LOGDIR

#TOOO: maybe save results of meson in there as well.

meson setup build-0 -Doptimization=3 -Dbuildtype=release -Dusm=true -Dua=true -Doffload=nvptx --native-file=./platforms/clang-21.ini
meson compile -C build-0

meson setup build-1 -Doptimization=3 -Dbuildtype=release -Dusm=true -Dua=true -Doffload=nvptx-cuda --native-file=./platforms/clang-21.ini
meson compile -C build-1

meson setup build-2 -Doptimization=3 -Dbuildtype=release -Dusm=true -Dua=true -Doffload=nvptx --native-file=./platforms/gcc-15.ini
meson compile -C build-2


nvidia-smi > "$LOGDIR/nvidia.info"
cat /proc/cpuinfo > "$LOGDIR/cpu.info"
./build-0/test-1 > "$LOGDIR/clang-21.nvptx.log"
./build-1/test-1 > "$LOGDIR/clang-21.nvptx-cuda.log"
./build-2/test-1 > "$LOGDIR/clang-15.nvptx.log"