# ViT5 Baseline for Legal Question Answering (VLQA)

This project fine-tunes the [ViT5](https://arxiv.org/abs/2204.07410) model for the Legal Question Answering (LQA) task.

## 📁 Project Structure

```
T5_BASELINE/
├── infer/                   # Inference scripts
│   ├── infer_main.py
│   ├── infer_metrics.py
│   └── infer_utils.py
├── output/                  # Output folder (checkpoints, results, etc.)
│── VLQA_data/               # Sample data files for training/testing
│   ├── law_data_example.json
│   ├── test_data_example.json
│   └── train_data_example.json
├── checkpoint_type.py
├── data_loader.py
├── dataset.py
├── main.py                  # Fine-tuning script
├── metrics.py
├── model.py
├── model_config.py
├── tokenizer.py
├── vlqa_data.py
├── README.md
└── requirement.txt
```

## ⚙️ Installation

```bash
pip install -r requirement.txt
```

## 🏋️ Fine-tuning

Run the following command to fine-tune the model:

```bash
python main.py
```

## 🔍 Inference

To perform inference with a trained model:

```bash
python -m infer.infer_main
```

## 📊 Data

The `VLQA_data` folder contains sample data files for demonstration purposes only, with a small number of items. Replace them with your own dataset for real training/inference.

---

Feel free to contribute or customize the project for your own legal QA needs.