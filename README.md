
# Slightly Less Basic Sentiment Analysis

This project expands upon the basics of sentiment analysis by incorporating multiple machine learning algorithms to classify app reviews into positive or negative sentiments. The models are trained to predict app ratings and to determine which developer companies have the best-performing apps.

## Project Overview

The analysis includes:
1. **Data Loading and Preprocessing**: Uses a custom function to read the review dataset, then applies text feature extraction techniques.
2. **Bag of Words Representation**: Converts the text data into a numerical format using the Bag of Words model.
3. **Machine Learning Models**: Implements several classifiers including Decision Tree, K-Nearest Neighbors (KNN), Linear and Non-linear Support Vector Machine (SVM), Random Forest, and a Voting Ensemble.
4. **Model Evaluation**: Evaluates each model's performance using precision, recall, and F1-score metrics.
5. **Developer Ranking**: Aggregates results to identify which developer companies produce the best-rated apps.

## Libraries Used

- **scikit-learn**: For machine learning models and evaluation metrics.
- **CountVectorizer**: For converting text reviews into numerical features.
- **Classification Algorithms**: Decision Tree, KNN, SVM, Random Forest, and Voting Classifier.

## Models and Algorithms

The project includes the following classifiers:
1. **Decision Tree Classifier**
2. **K-Nearest Neighbors (KNN)**: Evaluated for different values of K (1, 3, 15, 20).
3. **Support Vector Machine (SVM)**: Includes both linear and non-linear kernels.
4. **Random Forest Classifier**
5. **Voting Classifier**: Combines the results of multiple models to make a final prediction.

### Model Performance

Here are some key performance metrics observed for each model:
- **Decision Tree**: Achieved an accuracy of 70% with a high recall for positive reviews.
- **KNN (various values of K)**: Showed varying levels of accuracy, with K=15 and K=20 achieving the best performance.
- **SVM**: Linear SVM showed convergence issues but performed well overall, while the non-linear SVM achieved an accuracy of 79%.
- **Random Forest**: Demonstrated robust performance with a balanced precision and recall.
- **Voting Classifier**: Combined the strengths of multiple classifiers to reach a 76% accuracy.

## How to Run

1. **Requirements**:
   - Ensure you have the required Python libraries installed:
     ```bash
     pip install scikit-learn
     ```
2. **Data Preparation**:
   - Place the training and test datasets in the same directory as the notebook.
3. **Execution**:
   - Run each cell in sequence to train the models and evaluate their performance.

## Results

The results from the various classifiers were aggregated to determine which developer companies had the best apps based on positive sentiment reviews. The performance of developer companies was analyzed across different models to identify which company consistently produced well-rated apps.

### Developer Rankings
The notebook outputs the number of positively-rated apps for each developer across the different models, giving a comprehensive view of which developer had the best performance.

## Future Improvements

Potential improvements include:
- Implementing advanced text processing techniques like TF-IDF.
- Experimenting with other ensemble methods for better accuracy.
- Optimizing the SVM parameters to avoid convergence warnings.



### Contact
For any questions or feedback, please connect with me on [LinkedIn](https://www.linkedin.com/in/jayceeuzo/).