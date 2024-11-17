# FitVista: Sculpting Wellness, Shaping Lives
FitVista harnesses machine learning to empower fitness professionals and healthcare providers with a robust obesity classification model. By analyzing lifestyle and behavioral data, our model categorizes individuals into various obesity risk levels, offering actionable insights for tailored interventions.

## Key Highlights:
Models Used: Random Forest (best performer), XGBoost (variable performance), Logistic Regression (less effective for complex patterns).
Beneficiaries: Fitness experts, healthcare providers, and individuals seeking personalized wellness strategies.
## Future Vision:
Expand data sources for increased accuracy.
Integrate personalized health plans and mobile apps for real-time wellness guidance.
FitVista redefines fitness by making data-driven, personalized health a reality.

## Project Setup
Clone the Repository:
``` bash
Copy code
git clone https://github.com/swapna-9/FitVista_Multi_Class_Classification.git
``` `
Install Dependencies:
``` bash
Copy code
pip install -r requirements.txt
``` `
Run the Model: Dive into the src folder to explore the code and see the model in action.
```
## Project Structure

Copy code
├── data
│   ├── Obesity_DataSet.csv
│   └── __init__.py
├── outputs
│   ├── Comparison of Models over various obesity classes.png
│   ├── Correlation mapping.png
│   ├── Effect of High Caloric Food on Obesity Levels.png
│   ├── F1_Scores_for_multi_class.png
│   ├── Family_History_of_Overweight Vs Obesity_Levels.png
│   ├── Logistic_Regression_Confusion_matrix.png
│   ├── Pairplot.png
│   ├── Random_Forest_Confusion_matrix.png
│   └── XGBoost_Confusion_matrix.png
├── readme.md
├── requirements.txt
├── slides
│   ├── FitVista_multi_class_classification.pdf
│   └── FitVista_multi_class_classification.pptx
└── src
    ├── __init__.py
    ├── obesity_predict.ipynb
    └── obesity_predict.py

## Description of Key Folders:

data/: Contains the raw dataset and initialization files.
outputs/: Stores visualizations and results, including model comparisons and performance metrics.
slides/: Presentation files (PDF and PowerPoint) showcasing the project overview and findings.
src/: Source code for data processing, model training, and prediction scripts.