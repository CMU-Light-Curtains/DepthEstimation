sudo apt-get install libpng++-dev
cd deval_lib
rm -rf build; mkdir build; cd build;
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
cp pyevaluatedepth_lib.cpython-35m-x86_64-linux-gnu.so ../pyevaluatedepth_lib.so
cd ..; cd ..
cd perception_lib
rm -rf build; mkdir build; cd build;
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
cp pyperception_lib.cpython-35m-x86_64-linux-gnu.so ../pyperception_lib.so
cp libperception_lib.so ../
cd ..; cd ..
cd utils_lib
rm -rf build; mkdir build; cd build;
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
cp utils_lib.cpython-35m-x86_64-linux-gnu.so ../utils_lib.so
cd ..; cd ..
cd lcsim
git submodule init
git submodule update
rm -rf build; mkdir build; cd build;
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
cd ..; cd ..;

