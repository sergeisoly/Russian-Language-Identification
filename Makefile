.PHONY: train data requirements

requirements: requirements.txt
	pip install -r requirements.txt

data: 
	mkdir ./data
	mkdir ./data/lenta

	wget https://zenodo.org/record/841984/files/wili-2018.zip?download=1
	unzip ./wili-2018.zip?download=1 -d ./data
	rm ./wili-2018.zip?download=1

	gdown https://drive.google.com/uc?id=1NN9ttpm5bBfN1B8Bop0fWR2M9Hgo4Ah4
	unrar e ./Lenta.rar ./data/lenta/

split_data: data
	python3 src/split_datasets.py data/lenta data

