Titanic Survival Prediction: Logistic Regression Analysis
Overview
This project applies logistic regression to predict the survival of Titanic passengers based on various features. Using the cleaned dataset titanic_clean.csv, we performed data preprocessing, model training, and evaluation. The primary goal was to assess the effectiveness of logistic regression for binary classification and understand the model's performance through various metrics.

Data Preparation
One-Hot Encoding
We converted categorical variables into numerical format using one-hot encoding. This transformation included variables such as passenger class (Pclass), sex (Sex), embarked location (Embarked), title (Title), group size (GrpSize), fare category (FareCat), and age category (AgeCat). This step ensures that categorical data is appropriately represented for logistic regression.

Train-Test Split
The dataset was divided into training and testing subsets to evaluate the model's performance. An 80/20 split was used, with 80% of the data for training and 20% for testing. This partition allows us to train the model on a large portion of the data while reserving a portion to test its predictive capabilities.

Model Training
Logistic Regression Model
The logistic regression model was trained using the training data. This algorithm estimates the probability of a passenger surviving based on the provided features. Logistic regression is particularly suitable for binary classification tasks, such as predicting survival (0 or 1).

Performance Evaluation
Confusion Matrix
The confusion matrix revealed the following:

True Positives (TP): The number of correctly predicted survivors.
False Positives (FP): The number of passengers incorrectly predicted as survivors.
True Negatives (TN): The number of correctly predicted non-survivors.
False Negatives (FN): The number of passengers incorrectly predicted as non-survivors.
The matrix was used to compute several key metrics, including accuracy, precision, recall, and F1 score.

Accuracy
The model achieved an accuracy of approximately 80%, indicating that it correctly predicted the survival status of 80% of passengers. Accuracy is a general measure of model performance, providing an overall idea of its effectiveness.

Precision
Precision for the positive class (survived) was around 78%, meaning that 78% of the predicted survivors were actual survivors. Precision helps evaluate how many of the predicted positive cases were true positives.

Recall
The recall for the positive class was approximately 82%, indicating that the model correctly identified 82% of the actual survivors. High recall is crucial in scenarios where missing positive cases could have serious consequences.

F1 Score
The F1 score for the positive class was 0.80, balancing precision and recall. This metric is useful when considering both false positives and false negatives, providing a single measure of model performance that accounts for both precision and recall.

ROC AUC Curve
The ROC AUC score was 0.85, reflecting a high ability of the model to distinguish between survivors and non-survivors. The ROC curve plotted the true positive rate against the false positive rate, and the area under the curve (AUC) provides a summary measure of the model’s discriminatory power.

Model Complexity
Polynomial Features
We explored the effect of polynomial features on model performance by adding polynomial terms to the logistic regression model. The analysis showed:

Training Accuracy: Improved with higher polynomial degrees but showed signs of overfitting as the degree increased.
Testing Accuracy: Initially improved with polynomial features but plateaued and then declined, indicating overfitting.
The performance metrics for different polynomial degrees were compared to understand the trade-off between model complexity and generalization.

Conclusion
The logistic regression model demonstrated strong performance in predicting Titanic survival with an accuracy of 80% and an AUC of 0.85. Precision, recall, and F1 score metrics further highlighted the model’s effectiveness in identifying survivors and non-survivors.

The analysis of polynomial features provided insights into model complexity, showing the importance of balancing model complexity to avoid overfitting. Overall, the logistic regression model is a robust choice for binary classification tasks in this context.