# Clone the repository with submodules
git clone --recursive https://github.com/multi30k/dataset.git multi30k-dataset

# Run download script
python download_flickr.py

# Process test_2017-flickr-images.tar.gz
mkdir -p ./images
tar -xvzf test_2017-flickr-images.tar.gz
mv task1 test_2017_flickr
mv test_2017_flickr ./images
rm -rf test_2017-flickr-images.tar.gz

# Process flickr30k-images.tar.gz and split into train, val, and test
python create_test_val_flickr.py
rm -rf flickr-image-dataset

# Process the captions
mv multi30k-dataset/* .
rm -rf multi30k-dataset
mkdir text
mv data text
mv scripts text

# Navigate to the raw data folder
cd text/data/task1/raw

# Unzip all gz files
gunzip *.gz

# Rename files as needed and change their permissions
for file in val.*; do
  mv "$file" "test_2016_val.${file##*.}"
  chmod 644 "test_2016_val.${file##*.}" # Set read and write for owner, read for others
done

cd ../image_splits
mv val.txt test_2016_val.txt
chmod 644 test_2016_val.txt # Set read and write for owner, read for others
