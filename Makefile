.PHONY: train data requirements

requirements: requirements.txt
	pip install -r requirements.txt

data: 
	mkdir ./data
	mkdir ./data/lenta

	wget https://zenodo.org/record/841984/files/wili-2018.zip?download=1
	unzip ./wili-2018.zip?download=1 -d ./data
	rm ./wili-2018.zip?download=1

	unrar e /content/drive/My\ Drive/ABBYY_MIPT/Lenta.rar ./data/lenta/

split: data
	python3 src/split_dataset.py data/lenta data

