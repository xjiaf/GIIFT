import shutil
from pathlib import Path


def create_directories(base_path):
    image_dir = base_path / 'images'
    # text_dir = base_path / 'text'
    # text_train_dir = text_dir / 'text_train'
    # text_val_dir = text_dir / 'text_val'
    # text_test_dir = text_dir / 'text_test'
    image_train_dir = image_dir / 'train'
    image_val_dir = image_dir / 'test_2016_val'
    image_test_dir = image_dir / 'test_2016_flickr'

    # dirs = [processed_dir, multi30k_dir, text_train_dir, text_val_dir, text_test_dir, image_train_dir, image_val_dir, image_test_dir]
    dirs = [image_dir, image_train_dir, image_val_dir, image_test_dir]
    for dir in dirs:
        dir.mkdir(parents=True, exist_ok=True)
    # return text_train_dir, text_val_dir, text_test_dir, image_train_dir, image_val_dir, image_test_dir
    return image_train_dir, image_val_dir, image_test_dir


def move_text_files(task1_tok_dir, text_train_dir, text_val_dir,
                    text_test_dir):
    for file in task1_tok_dir.iterdir():
        if file.name.startswith('train'):
            shutil.copy2(str(file), str(text_train_dir / file.name))
        elif file.name.startswith('val'):
            shutil.copy2(str(file), str(text_val_dir / file.name))
        elif file.name.startswith('test_2016'):
            shutil.copy2(str(file), str(text_test_dir / file.name))


def move_image_files(image_splits_dir, flickr_images_dir, image_train_dir,
                     image_val_dir, image_test_dir):
    splits = {
        'train.txt': image_train_dir,
        'val.txt': image_val_dir,
        'test_2016_flickr.txt': image_test_dir
    }

    for split_file, target_dir in splits.items():
        with open(image_splits_dir / split_file, 'r') as f:
            for line in f:
                image_file = flickr_images_dir / line.strip()
                if image_file.exists():
                    shutil.copy2(str(image_file),
                                 str(target_dir / image_file.name))


def main():
    base_path = Path('.')
    data_dir = base_path
    # task1_tok_dir = data_dir / 'multi30k-dataset' / 'data' / 'task1' / 'tok'
    image_splits_dir = data_dir / 'multi30k-dataset' / 'data' / 'task1' / 'image_splits'
    flickr_images_dir = data_dir / 'flickr-image-dataset' / 'flickr30k_images' / 'flickr30k_images'

    image_train_dir, image_val_dir, image_test_dir = create_directories(
        base_path)
    # move_text_files(task1_tok_dir, text_train_dir, text_val_dir, text_test_dir)
    move_image_files(image_splits_dir, flickr_images_dir, image_train_dir,
                     image_val_dir, image_test_dir)


if __name__ == "__main__":
    main()
