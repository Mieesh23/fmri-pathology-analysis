## 🔍 Key Findings & Insights from the Thesis

This thesis evaluated multiple machine-learning models (XGB, Random Forest, SVC) and feature approaches (functional connectivity, graph-based metrics, activated features) for detecting depression while explicitly examining the impact of data leakage on model performance.

### **Main findings**
- Models **appeared to perform better when data leakage was present**, highlighting how leakage can artificially inflate accuracy and must be strictly avoided.
- **XGBoost** consistently achieved the highest performance across feature types, followed by Random Forest.
- The **graph-based approach** performed weakest overall, likely due to challenges in defining optimal thresholding strategies.
- The activated-feature (AF) approach showed **high variability**, depending on node selection and leakage presence.

### **Overfitting analysis**
Although XGB-AF with five selected nodes performed extremely well before validation,  
**performance dropped sharply on the validation set (ACC, SPE, AUC)** — indicating:
- overfitting,
- over-optimistic classification behavior,
- limited generalization to unseen subjects.

### **Application to healthy controls**
Applying the best model to a healthy dataset identified **34.8%** as “potentially depressed.”  
Two alternative methods estimated a hidden prevalence (“dark figure”) of **20.9–26.5%**,  
both higher than in comparable studies, underscoring the need for model refinement and stricter leakage control.

---

## Future Directions & Methodological Outlook

Promising directions for improving robustness and generalization include:

### **1. Deep Learning for Feature Engineering**
Use of **Autoencoders** to:
- automatically extract informative latent biomarkers,  
- reduce dimensionality,  
- suppress noise and redundancy,  
- generate more stable feature representations.

### **2. Probabilistic Edge Selection in Graphs**
Instead of manual thresholding, evaluate:
- which edges show highest probability of relevance across subjects,  
- prioritizing these during random sampling from all 6,670 possible connections.

### **3. Fusion of Questionnaire-Based and fMRI-Derived Graphs**
Building **multi-modal graph networks** combining:
- psychological questionnaire data,  
- rs-fMRI connectivity metrics,  
to create more comprehensive and discriminative feature sets.

---

## 🧠 Overall Conclusion

This work highlights both the potential and the challenges of using ML models to identify depression from brain connectivity data.  
Careful control of **data leakage**, **overfitting**, and **feature selection** remains essential.  
Progress in this area will depend on close collaboration between **psychology**, **data science**, and **machine learning**, especially when exploring multi-modal and graph-theoretical approaches.

