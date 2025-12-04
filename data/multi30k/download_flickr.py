import sys
import subprocess


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def import_or_install(package):
    try:
        return __import__(package)
    except ImportError:
        install(package)
        return __import__(package)


# install or import opendatasets and gdown
od = import_or_install('opendatasets')
gdown = import_or_install('gdown')

# Assign the Kaggle data set URL into variable
dataset = 'https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset'
# Using opendatasets let's download the data sets
od.download(dataset)

# Google Drive file ID of flickr 2017 dataset
file_id = '1mHCPUvu3anva-m0IzOLMvUYGd21y56T2'
# Output file name
output = 'test_2017-flickr-images.tar.gz'
# Google Drive download link
url = f'https://drive.google.com/uc?id={file_id}'
gdown.download(url, output, quiet=False)
