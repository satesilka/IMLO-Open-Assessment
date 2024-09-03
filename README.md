# IMLO-Open-Assessment
**Intelligent Systems: Machine Learning and Optimisation Individual Open Assessment**

## Overview

This project involves creating a neural network model to classify images of flowers into specific species using the **Oxford 102 Category Flower Dataset**. The model was designed, trained, and evaluated using **PyTorch**. This README provides instructions on how to set up the environment, run the provided scripts, and evaluate the pre-trained model.

## Files Included

- [`flowers102_classifier.py`](flowers102_classifier.py): Python script used to train and evaluate the neural network model.
- [`flowers102_classifier.ipynb`](flowers102_classifier.ipynb): Jupyter notebook where the model was initially developed and evaluated.
- [`test_model.py`](test_model.py): Python script to load and evaluate the pre-trained model without retraining.
- [`flowers102_classifier_model.pth`](flowers102_classifier_model.pth): Pre-trained model file.
- [`Flowers102Report.pdf`](Flowers102Report.pdf): Final report detailing the methodology, results, and conclusions.

## Setting Up the Environment

1. **Create a Virtual Environment**:

   ```bash
   python -m venv flowers102_env
2. **Activate the Virtual Environment:**
   
   *On Windows:*

   ```bash
   flowers102_env\Scripts\activate
   ```

   *On macOS/Linux:*

   ```bash
   source flowers102_env/bin/activate
   ```
3. **Install Required Packages:**
```bash
pip install torch torchvision
```
## Running the Code

4. **To Train and Evaluate the Model**
- Activate the virtual environment ([`flowers102_env`](flowers102_env)).
- Run the [`flowers102_classifier.py`](flowers102_classifier.py) script to train the model and evaluate its performance:
```bash
python flowers102_classifier.py
```
5. **To Load and Evaluate the Pre-trained Model**
Ensure [`flowers102_classifier_model.pth`](flowers102_classifier_model.pth)h is in the same directory as test_model.py.
Run the [`test_model.py`](test_model.py) script to load and evaluate the pre-trained model:
```bash
python test_model.py
```
