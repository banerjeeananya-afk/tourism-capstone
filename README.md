# Preserving Heritage: Enhancing Tourism with AI 

This project is an end‑to‑end AI/ML capstone that combines **deep learning** and **recommender systems** to support cultural heritage preservation and enhance tourism experiences.

- **Part 1:** Historical structure image classification using transfer learning (EfficientNetB0, TensorFlow/Keras)
- **Part 2:** Tourism data analysis and an item‑based collaborative filtering recommender system

---

## 1. Project overview

### Part 1 — Historical structure classification (Deep Learning)

Goal: Classify images of historical structures (e.g., churches, bell towers, monasteries) into multiple architectural categories using transfer learning.

Key points:

- Dataset: **10,236 training images** and **1,479 test images**
- Model: **EfficientNetB0** pretrained on ImageNet
- Two variants:
  - **Model A:** Baseline (no augmentation)
  - **Model B:** With data augmentation (flip, rotation, zoom)
- Input pipeline: `image_dataset_from_directory`, caching, shuffling, prefetching
- Early stopping based on validation accuracy

Result:  
**Model A (No Augmentation)** achieved slightly better validation accuracy and lower validation loss than the augmented model, likely due to the large, diverse dataset and the sensitivity of architectural features to geometric distortions.

---

### Part 2 — Tourism data analysis & recommender system

Goal: Analyze tourism patterns and build a recommender system to suggest similar tourist destinations.

Data sources:

- `user.csv` — user demographics
- `tourism_rating.csv` — user ratings for places
- `tourism_with_id.xlsx` — metadata for tourist spots (city, category, etc.)

Key steps:

- Data cleaning: remove invalid ages/ratings, handle duplicates
- Exploratory analysis:
  - Age distribution of tourists
  - Top origin cities
  - Most common categories and city–category relationships
- Insights:
  - Strong domestic tourism from major cities
  - Nature and cultural attractions are highly rated
- Recommender:
  - Built an **item‑based collaborative filtering** model
  - User–item rating matrix + cosine similarity between places
  - Given a place (e.g., *Pulau Pari*), recommend similar destinations

---

## 2. Tech stack

- **Languages:** Python
- **Deep Learning:** TensorFlow, Keras, EfficientNetB0
- **Data Science:** Pandas, NumPy, Matplotlib, Seaborn
- **ML / Recommender:** Scikit‑learn (cosine similarity, collaborative filtering)
- **Environment:** Google Colab

---

## 3. How to run the project

### 3.1. Clone the repository

```bash
git clone https://github.com/<your-username>/tourism-heritage-aiml-capstone.git
cd tourism-heritage-aiml-capstone
