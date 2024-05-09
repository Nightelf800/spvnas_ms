#! /bin/bash
cd convolution
rm -rf build
rm -rf convolution_cuda.so
python3 setup.py convolution_cuda.cu convolution_cuda.so

cd ..
cd devoxelize
rm -rf build
rm -rf devoxelize_cuda.so
python3 setup.py devoxelize_cuda.cu devoxelize_cuda.so

cd ..
cd 'hash'
rm -rf build
rm -rf hash_cuda.so
python3 setup.py hash_cuda.cu hash_cuda.so

cd ..
cd hashmap
rm -rf build
rm -rf hashmap_cuda.so
python3 setup.py hashmap_cuda.cu hashmap_cuda.so

cd ../others
rm -rf build
rm -rf count_cuda.so
rm -rf query_cuda.so
python3 setup.py count_cuda.cu count_cuda.so
python3 setup.py query_cuda.cu query_cuda.so

cd ../voxelize
rm -rf build
rm -rf voxelize_cuda.so
python3 setup.py voxelize_cuda.cu voxelize_cuda.so
