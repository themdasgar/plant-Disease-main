# 🌿 Plant Leaf Disease Detection Project

## About this project

This project is built to **identify plant leaves and detect plant diseases from images, videos, or a live camera** in a simple and understandable way.

The main idea is very practical:

* If the input image **contains a plant leaf**, the system tries to **identify the plant and its disease**.
* If the input image **does not contain a leaf** (for example: a person, bottle, animal, or any other object), the system clearly says **“This is not a leaf”** and tells **what object it is**.

The goal is to avoid wrong disease predictions and make the system reliable and easy to explain, even to non-technical users.

---

## Why this project is useful

In real life, farmers or users may upload **any image**, not always a clean leaf photo. If a system blindly predicts disease on every image, the result becomes confusing and incorrect.

This project solves that problem by:

* First checking **whether the image looks like a leaf**
* Then predicting disease **only if it is really a leaf**

So the system behaves more like a human decision process.

---

## Dataset information

* The plant disease dataset used for training the model is taken from **Kaggle**.
* The dataset contains images of different plant leaves with various diseases and healthy conditions.
* Each image belongs to a specific **plant type** and **disease category**.

The dataset was already well-structured and widely used for learning and academic projects.

---

## How the system works (simple explanation)

You can think of this project as a **three-step pipeline**:

### 1️⃣ Leaf check (Color-based detection)

First, the system looks at the **colors in the image**.

* Green leaves
* Yellow leaves (dry or unhealthy)
* Brown leaves (old or infected)

If these colors cover a large part of the image, the system assumes:

> “Yes, this looks like a leaf.”

If not, it assumes:

> “This is probably not a leaf.”

This step is very fast and avoids many mistakes.

---

### 2️⃣ Object detection (for non-leaf images)

If the image **does not look like a leaf**, the system uses a pre-trained object detection model to recognize common objects such as:

* Person
* Bottle
* Animal
* Vehicle

Then it shows a clear message like:

> ❌ Not a leaf – Detected object: Person

So users immediately understand what the image actually contains.

---

### 3️⃣ Disease prediction (only for leaf images)

If the image **is confirmed as a leaf**, it is passed to a trained deep learning model that:

* Identifies the **plant name**
* Identifies the **disease** (or healthy condition)
* Shows a **confidence score**

Example output:

> 🌱 Plant: Tomato
> 🦠 Disease: Late Blight
> 📊 Confidence: 96%

---

## Model training (brief and clear)

* The disease prediction model is trained using a **Convolutional Neural Network (CNN)**.
* Images are resized to **224 × 224** for training and prediction.
* Data augmentation (rotation, zoom, flipping, etc.) is used to make the model stronger.
* The model is trained for **multiple plant diseases and healthy classes**.

The trained model is saved as:

```
best_model.h5
```

---

## Project features

* Works with **image upload**
* Works with **video files**
* Works with **live camera**
* Avoids false disease predictions on non-leaf images
* Easy-to-use web interface
* Clear and understandable output messages

---

## Project structure (simple view)

```
plant_disease_project/
│
├── app.py                # Main application file
├── best_model.h5        # Trained disease prediction model
├── class_names.txt      # List of plant and disease labels
├── yolov8n.pt           # Object detection model
├── plant-disease-classification-using-cnn.ipynb  # Model training reference
└── README.md            # Project explanation
```

---

## How to run the project

1. Install required libraries
2. Place all model files in the same folder
3. Run the application

```bash
streamlit run app.py
```

Then open the browser link shown in the terminal.

---

## Limitations (honest points)

* Very green backgrounds (like grass or trees) may sometimes be detected as a leaf.
* Very dark or blurred images can reduce accuracy.
* This project is best suited for **learning, demo, and academic use**.

---

## Future improvements

* Training a dedicated leaf detection model
* More accurate leaf area extraction
* Adding treatment suggestions for diseases
* Deploying the project online

---

## Final note

This project is designed to be **simple to understand**, **safe to demonstrate**, and **practical for real-world learning**. It focuses not only on prediction accuracy but also on **making correct decisions before prediction**, which is an important concept in applied AI.

---

## One-line summary

> A smart plant disease detection system that first checks whether an image is a leaf and then predicts plant disease only when it makes sense to do so.
