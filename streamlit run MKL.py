{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "a5d2116a-b035-4f1c-be69-3a3931e546fd",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "4c9d8241-35e2-49da-9676-03f04e410d28",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "355b963e-9d84-4304-9992-7acc837d6a1b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the dataset\n",
    "file_path = r'C:\\Users\\PERFECT COMPUTERS\\Downloads\\t20.csv'\n",
    "t20_data = pd.read_csv(file_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "a692291a-e81b-4db1-969d-671dd369f323",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Data Preprocessing\n",
    "# Convert Date to datetime\n",
    "t20_data['Date'] = pd.to_datetime(t20_data['Date'], errors='coerce')\n",
    "\n",
    "# Encoding categorical variables\n",
    "le = LabelEncoder()\n",
    "t20_data['Venue'] = le.fit_transform(t20_data['Venue'])\n",
    "t20_data['Bat First'] = le.fit_transform(t20_data['Bat First'])\n",
    "t20_data['Bat Second'] = le.fit_transform(t20_data['Bat Second'])\n",
    "t20_data['Winner'] = le.fit_transform(t20_data['Winner'])\n",
    "\n",
    "# Features and target variable\n",
    "X = t20_data.drop(columns=['Winner', 'Date'])\n",
    "y = t20_data['Winner']\n",
    "\n",
    "# Splitting the dataset\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "6e7871de-dc34-4295-a5cf-5e7459ee21f6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic Regression Accuracy: 0.11653116531165311\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\PERFECT COMPUTERS\\anaconda\\Lib\\site-packages\\sklearn\\linear_model\\_logistic.py:469: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    }
   ],
   "source": [
    "# Logistic Regression\n",
    "log_reg = LogisticRegression(max_iter=200)\n",
    "log_reg.fit(X_train, y_train)\n",
    "y_pred = log_reg.predict(X_test)\n",
    "\n",
    "# Evaluation\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(\"Logistic Regression Accuracy:\", accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "8166b77e-2263-4fd7-ac9c-0197d464b686",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic Regression Accuracy: 0.11653116531165311\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\PERFECT COMPUTERS\\anaconda\\Lib\\site-packages\\sklearn\\linear_model\\_logistic.py:469: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "# Logistic Regression with L2 regularization\n",
    "log_reg = LogisticRegression(max_iter=200, C=1.0)  # You can adjust C to control regularization strength\n",
    "log_reg.fit(X_train, y_train)\n",
    "y_pred = log_reg.predict(X_test)\n",
    "\n",
    "# Evaluation\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(\"Logistic Regression Accuracy:\", accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "fa6585fa-0a8b-463d-8b33-ee494b9671cf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Decision Tree Accuracy: 0.43089430894308944\n"
     ]
    }
   ],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "# Decision Tree\n",
    "tree_clf = DecisionTreeClassifier()\n",
    "tree_clf.fit(X_train, y_train)\n",
    "y_pred = tree_clf.predict(X_test)\n",
    "\n",
    "# Evaluation\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(\"Decision Tree Accuracy:\", accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "79ff980a-9aef-468e-b5d3-5e0582735307",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Accuracy: 0.4634146341463415\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "# Random Forest\n",
    "rf_clf = RandomForestClassifier()\n",
    "rf_clf.fit(X_train, y_train)\n",
    "y_pred = rf_clf.predict(X_test)\n",
    "\n",
    "# Evaluation\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(\"Random Forest Accuracy:\", accuracy)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "83940f18-a950-4f72-bea2-a7790ff763e6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Accuracy with Regularization: 0.4037940379403794\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Load and preprocess the dataset (assuming X and y are already defined)\n",
    "# Splitting the dataset into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Random Forest with regularization\n",
    "rf_clf = RandomForestClassifier(\n",
    "    n_estimators=100,  # Number of trees in the forest\n",
    "    max_depth=10,      # Maximum depth of each tree\n",
    "    min_samples_split=5,  # Minimum number of samples required to split an internal node\n",
    "    min_samples_leaf=4,   # Minimum number of samples required to be at a leaf node\n",
    "    max_features='sqrt',  # Number of features to consider at each split (can be 'auto', 'sqrt', 'log2', or an integer)\n",
    "    random_state=42       # For reproducibility\n",
    ")\n",
    "\n",
    "# Training the model\n",
    "rf_clf.fit(X_train, y_train)\n",
    "\n",
    "# Making predictions\n",
    "y_pred = rf_clf.predict(X_test)\n",
    "\n",
    "# Evaluation\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(\"Random Forest Accuracy with Regularization:\", accuracy)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "bfbf8fcb-e337-4ef0-b1d9-011499cc564a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SVM Accuracy: 0.15447154471544716\n"
     ]
    }
   ],
   "source": [
    "from sklearn.svm import SVC\n",
    "\n",
    "# SVM\n",
    "svm_clf = SVC()\n",
    "svm_clf.fit(X_train, y_train)\n",
    "y_pred = svm_clf.predict(X_test)\n",
    "\n",
    "# Evaluation\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(\"SVM Accuracy:\", accuracy)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "61e821e1-8845-4c73-8bbb-6be7b6ca8c14",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "KNN Accuracy: 0.24390243902439024\n"
     ]
    }
   ],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "# KNN\n",
    "knn_clf = KNeighborsClassifier()\n",
    "knn_clf.fit(X_train, y_train)\n",
    "y_pred = knn_clf.predict(X_test)\n",
    "\n",
    "# Evaluation\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(\"KNN Accuracy:\", accuracy)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e1ab0b6d-42fc-4b8b-9cf1-370d1b81b64e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
