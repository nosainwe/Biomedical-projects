# Knee Osteoarthritis Severity Classifier

DenseNet201 fine-tuned to grade knee osteoarthritis severity from X-ray images using the Kellgren-Lawrence (KL) scale.

## Classes

| Grade | Description |
|-------|-------------|
| 0 | Normal |
| 1 | Doubtful |
| 2 | Minimal |
| 3 | Moderate |
| 4 | Severe |

## Dataset

[Knee Osteoarthritis Dataset with Severity](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity) - download and place under:

```
knee-osteoarthritis-dataset-with-severity/
├── train/
│   ├── 0/  1/  2/  3/  4/
├── val/
│   ├── 0/  1/  2/  3/  4/
└── test/
    ├── 0/  1/  2/  3/  4/
```

## Architecture

- **Backbone:** DenseNet201 pretrained on ImageNet (all layers unfrozen)
- **Head:** GlobalAveragePooling → Dropout(0.4) → Dense(5, softmax) with L1+L2 regularization
- **Optimizer:** Adam (lr=1e-5)
- **Loss:** Categorical cross-entropy

## Augmentation

Training uses heavy augmentation to compensate for limited medical imaging data:

- Horizontal/vertical flips, rotation up to 40°
- Shear, zoom, width/height shifts
- **Random Erasing** - randomly zeros a rectangular patch to prevent over-reliance on local texture regions

No augmentation is applied at validation or test time.

## Training

```bash
pip install tensorflow tensorflow-addons scikit-learn opencv-python matplotlib numpy
python knee_oa_classifier.py
```

Trains for up to 120 epochs with early stopping (patience=20 on val_accuracy). Best checkpoint saved to `Best_DenseNet201.h5`.

## Outputs

- `Best_DenseNet201.h5` - best model by validation accuracy
- Accuracy and loss curves plotted inline
- Confusion matrix and per-class classification report on test set

## Notes

- `steps_per_epoch` and `validation_steps` are hardcoded to the dataset split sizes (5778 train / 826 val / 1656 test) - update these if using a different split
- `predict_generator` is deprecated in TF2.x but kept for compatibility with older environments
