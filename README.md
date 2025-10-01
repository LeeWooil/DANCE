# DANCE: Disentangled Concepts Speak Louder Than Words – Explainable Video Action Recognition [[NeurIPS 2025 Spotlight]]()

[![Conference](https://img.shields.io/badge/NeurIPS-2025-red)]() [![Project Page](https://img.shields.io/badge/Project-Page-green)](https://jong980812.github.io/DANCE/) [![Lab](https://img.shields.io/badge/Vision%20and%20Learning-Lab-orange)](https://vll.khu.ac.kr/)  [![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)]() ![visitor badge](https://visitor-badge.laobi.icu/badge?page_id=your_github_username.DANCE)

---
<p align="center">
  <img src="assets/architecture-gif.gif" alt="DANCE Architecture Overview" width="800"/>
</p>


## ✨ Highlights

- 🔍 Proposes **DANCE**, a Concept Bottleneck Model framework for **explainable video action recognition**.
- 🔧 Uses **disentangled concepts** from motion dynamics, objects, and scenes.
- 🔄 Includes **concept-level interventions**, **concept swapping**, and **concept ablations**.
- 📈 Demonstrates **strong interpretability** and competitive performance on **Penn Action**, **UCF101**, etc.

---

## 🛠️ Installation

We provide two ways to set up the environment:

### Option 1. Using conda with environment.yml (recommended for reproducibility)
```bash
# Create and activate environment
conda env create -f environment.yml
conda activate dance
```
### Option 2. Manual setup with requirements.txt
```bash
# Create and activate environment
conda create -n dance python=3.10 -y
conda activate dance

# Install PyTorch (modify CUDA version if needed)
conda install pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install additional dependencies
pip install -r requirements.txt
```


---

## 📁 Project Structure

```bash
DANCE/
│── CBM_training/
│   ├── train_video_cbm.py
│   ├── Feature_extraction/
│   ├── model/
│   └── ...
│── Concept_extraction/
│   ├── keyframe_selection/
│   ├── Motion_discovery/
│   └── ...
│── Dataset/
│── Experiments/
│   ├── Evaluation.ipynb
│   └── Intervention.ipynb
│── result/
│── requirements.txt
│──
│── README.md
```

---

## 🧠 Dataset and Features
<!-- 
- `Penn Action` is the main benchmark. Videos and annotations should be organized as:

```bash
/Penn_Action/videos/
... Penn_Action_motion_label/
... Penn_Action_feature/
```

- Precomputed features:
  - `--backbone_features`: extracted from VideoMAE
  - `--vlm_features`: extracted from InternVideo-200M
  - `--pose_label`: cluster-based motion labels -->

---

## 🚀 Training

```bash
python CBM_training/train_video_cbm.py \
    --data_set penn-action \
    --nb_classes 15 \
    --spatial_concept_set PATH_TO_OBJECT_CONCEPTS.txt \
    --place_concept_set PATH_TO_SCENE_CONCEPTS.txt \
    --temporal_concept_set PATH_TO_TEMPORAL_CONCEPTS.txt \
    --batch_size 64 \
    --finetune PATH_TO_BACKBONE.pt \
    --dual_encoder internvid_200m \
    --activation_dir ./result/Penn_Action_result \
    --save_dir ./result/Penn_Action_result \
    --n_iters 30000 \
    --interpretability_cutoff 0.3 \
    --clip_cutoff 0.2 \
    --backbone vmae_vit_base_patch16_224 \
    --proj_steps 3000 \
    --train_mode pose spatial place \
    --data_path ./data/videos \
    --backbone_features ./features/backbone.pt \
    --vlm_features ./features/vlm.pt \
    --pose_label ./motion_labels \
    --proj_batch_size 50000 \
    --saga_batch_size 128 \
    --loss_mode concept \
    --use_mlp
```

---

## 📊 Evaluation & Visualization

Evaluation scripts and concept intervention scripts are included in:

- `Experiments/Evaluation.ipynb`
- `Experiments/Intervention.ipynb`
<p align="center">
  <img src="assets/Demo_no_animation.gif" alt="Prediction Demo" width="700"/>
</p>


## 📎 Citation

```bibtex
@inproceedings{,
  title={DANCE: Disentangled Concepts Speak Louder Than Words – Explainable Video Action Recognition},
  author={},
  booktitle={NeurIPS},
  year={2025}
}
```

---

## 📜 License

This project is licensed under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## 🙏 Acknowledgements

This work builds upon:  
[Trustworthy-ML-Lab/Label-free-CBM](https://github.com/Trustworthy-ML-Lab/Label-free-CBM/tree/main)