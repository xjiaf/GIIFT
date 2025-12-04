# GIIFT: Graph-guided Inductive Image-free Multimodal Machine Translation  [[Paper](https://aclanthology.org/2025.wmt-1.6/)][[arXiv](https://arxiv.org/abs/2507.18562)]

Setup the repository with the following commands:
```bash
git clone https://github.com/xjiaf/GIIFT.git
cd GIIFT
conda env create --file environment.yml
conda activate giift
```

# Setup Data

All data should be organised in the ```data/``` directory.

Download the images for [Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) and upzip it in the ```data/multi30k``` folder.

Execute ```data/multi30k/setup_multi30k.sh``` to download the text data and organize the folders from kaggle and google drive.

# GIIFT Training script
## 1. Quick Start
This repository contains a Bash launcher for GIIFT. Simply run:

```bash
bash run.sh
```
The default configuration uses:

* **Dataset:** `multi30k`
* **Backbone:** `mbart`
* **Task:** `en` -> `de`, `fr`

This executes **Stage 1 → Stage 2 → Testing** for each target language.

---

## 2. Default Commands (Fully Expanded)

Below are the *actual commands* generated internally by the script (with default values).
This helps users understand what the launcher builds.

### **Stage 1**

Example for German (`de`) and backbone `mbart`:

```bash
python src/main.py --num_gpus 1 \
  --mn multi30k_mbart \
  --prefix_length 1 \
  --bs 64 \
  --update_count 4 \
  --lr 2e-5 \
  --epochs 50 \
  --test_ds "2016 val" \
  --stage caption \
  --tgt_lang de \
  --num_heads 8 \
  --num_layers 9 \
  --mapping_network gatl \
  --mask_prob 0 \
  --backbone mbart \
  --use_gate --use_mbart_encoder --use_fusion \
  --gpu_id 0
```

### **Stage 2**

```bash
python src/main.py --num_gpus 1 \
  --mn multi30k_mbart \
  --prefix_length 1 \
  --bs 64 \
  --update_count 4 \
  --lr 1e-5 \
  --epochs 50 \
  --test_ds "2016 val" \
  --stage translate \
  --tgt_lang de \
  --lm model_pretrained.pth \
  --num_heads 8 \
  --num_layers 9 \
  --mapping_network gatl \
  --backbone mbart \
  --use_gate --use_mbart_encoder --use_fusion \
  --gpu_id 0
```

### **Testing**

Runs on:

* 2016:flickr
* 2017:flickr
* 2017:mscoco

```bash
python src/main.py --num_gpus 1 \
  --mn multi30k_mbart \
  --src_lang en \
  --tgt_lang de \
  --prefix_length 1 \
  --bs 64 \
  --test_ds 2016 flickr \
  --stage translate \
  --test \
  --lm model_best_test.pth \
  --num_heads 8 \
  --num_layers 9 \
  --mapping_network gatl \
  --backbone mbart \
  --use_gate --use_mbart_encoder --use_fusion \
  --gpu_id 0
```

---

## 3. How to Change Key Settings

### **A. Select Dataset**

In the script:

```bash
dataset="multi30k"
```

Switch to WMT:

```bash
dataset="wmt"
```

---

### **B. Set Translation Languages**

Modify:

```bash
languages=("de" "fr")
```

Examples:

**Only German**

```bash
languages=("de")
```

**German + Czech + Ukrainian**

```bash
languages=("de" "cs" "uk")
```

**Important:**
These values directly become `--tgt_lang` in both stages.

---

## 4. Other setting example

### **Run stage 1 only**

```bash
run_stage1=true
run_stage2=false
```

### **Run stage 2 only (must have pretrained Stage 1 model)**

```bash
run_stage1=false
run_stage2=true
```

### **Run WMT-German only, MBART backbone**

```bash
dataset="wmt"
languages=("de")
backbones=("mbart")
```


---
# Citation
If the code and/or method was useful to your work, please consider citing us!

```code
@inproceedings{xiong-zhao-2025-giift,
    title = "{GIIFT}: Graph-guided Inductive Image-free Multimodal Machine Translation",
    author = "Xiong, Jiafeng  and
      Zhao, Yuting",
    booktitle = "Proceedings of the Tenth Conference on Machine Translation",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    doi = "10.18653/v1/2025.wmt-1.6",
    pages = "98--112",
    ISBN = "979-8-89176-341-8"
}
```
