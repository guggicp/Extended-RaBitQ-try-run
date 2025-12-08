rm -rf build
rm -rf bin

export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

# 再次确认一下版本
$CXX --version
echo "----------------------"
$CC --version
echo "----------------------"
mkdir build bin
cd ./build
cmake ..
bear -- make -j$(nproc)
mv compile_commands.json ../
