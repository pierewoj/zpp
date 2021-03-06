sudo apt-get install build-essential
sudo apt-get install linux-headers-`uname -r`
sudo apt-get install curl

if ! [ -f  cuda.run ]; then
curl -O "http://developer.download.nvidia.com/compute/cuda/8.0/secure/prod/local_installers/cuda_8.0.44_linux.run?autho=1479921175_65d5eb50fd6a1d774261bd67ae733942&file=cuda_8.0.44_linux.run"
mv *cuda_8.0.44_linux.run cuda.run
chmod +x cuda.run
fi

sudo ./cuda.run --kernel-source-path=/usr/src/linux-headers-`uname -r`i /
echo 'export PATH=/usr/local/cuda-8.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/usr/local/lib' >> ~/.bashrc
source ~/.bashrc
