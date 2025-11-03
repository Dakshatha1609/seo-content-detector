# SEO Content Quality & Duplicate Detector

This project analyzes web content for SEO quality and detects duplicate pages using NLP and ML.

## 📘 Overview
The system:
- Extracts and cleans text from raw HTML or live URLs.
- Calculates SEO metrics (word count, sentence count, Flesch reading ease).
- Identifies duplicate or near-duplicate content using cosine similarity on TF-IDF embeddings.
- Predicts content quality labels (Low / Medium / High) using a trained RandomForest model.
- Provides a real-time `analyze_url(url)` function for quick testing on new web pages.

---

## ⚙️ Setup
1. Clone or download this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your dataset inside the `data/` folder as `data.csv`.
   - Must have at least `url` and `html_content` columns.

---

## 🚀 How to Run
1. Open **`seo_pipeline.ipynb`** in Jupyter Notebook or Google Colab.
2. Run the notebook **cell by cell** from top to bottom.
3. The notebook will automatically:
   - Parse HTML and extract titles & body text.
   - Generate features and TF-IDF vectors.
   - Detect duplicate pages.
   - Train the SEO quality classifier.
   - Provide a live analysis cell for new URLs.

---

## 📊 Outputs
After running, you’ll get:

| File | Description |
|------|--------------|
| `data/extracted_content.csv` | Extracted titles and body text |
| `data/features.csv` | SEO features and quality scores |
| `data/duplicates.csv` | Duplicate page pairs with similarity values |
| `models/quality_model.pkl` | Trained RandomForest model |

---

## 🧠 Key Components
- **BeautifulSoup + lxml** – HTML parsing  
- **TF-IDF Vectorizer** – keyword extraction and embeddings  
- **Cosine Similarity** – duplicate detection  
- **RandomForestClassifier** – SEO quality classification  
- **textstat** – readability scoring (Flesch Reading Ease)

---

## ✅ Checklist Alignment
- Repository is public on GitHub  
- `requirements.txt` pinned versions  
- Notebook runs end-to-end  
- Real-time URL analysis works  
- `.gitignore` excludes venv/pycache  
- No API keys committed  
- All data files in CSV format  

---

## 👩‍💻 Author
**Dakshatha Urs M S**  
MSc Data Science | Christ University  
🔗 [GitHub](https://github.com/yourusername)

---

## 📄 License
This project is for educational purposes and open for academic reference.
