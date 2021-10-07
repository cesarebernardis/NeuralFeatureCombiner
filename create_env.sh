conda create -y -n recsys-nfc python=3.8 --file requirements.txt -c conda-forge -c anaconda

source activate recsys-nfc

pip install similaripy
pip install tensorflow-gpu==2.5
pip install -e ./RecSysFramework/Utils/deepexplain/

unzip data.zip

cd RecSysFramework/Utils
git clone https://github.com/aksnzhy/xlearn.git
cd xlearn
rm -rf ./.git
mkdir build
cd build
cmake ../
make
cd ../../../..
