# Fake News Detection

This project aims to build a machine learning model to detect whether a given news article is real or fake. The model is trained on a dataset of news articles labeled as "REAL" or "FAKE" using a linear Support Vector Machine (SVM) algorithm and the TF-IDF (Term Frequency-Inverse Document Frequency) feature extraction technique.

## Dataset

The dataset used in this project is `fake_or_real_news.csv`, which contains the following columns:

- `id`: A unique identifier for each news article
- `title`: The title of the news article
- `text`: The content of the news article
- `label`: The label indicating whether the news article is "REAL" or "FAKE"

## Requirements

To run this project, you need to have the following libraries installed:

- NumPy
- Pandas
- Scikit-learn

You can install these libraries using pip:
```bash
pip install numpy pandas scikit-learn jupyter
```

## Usage

1. Clone the repository or download the project files.
2. Make sure you have the `fake_or_real_news.csv` dataset file in the same directory as the Jupyter Notebook.
3. Open the `Fake News Detection.ipynb` notebook in Jupyter Notebook or JupyterLab.
4. Run the notebook cells in order to train the model and make predictions.

## Workflow

1. Import the necessary libraries: NumPy, Pandas, and Scikit-learn.
2. Load the dataset from the CSV file using Pandas.
3. Preprocess the data by converting the "REAL" and "FAKE" labels to 0 and 1, respectively.
4. Split the data into training and test sets using Scikit-learn's `train_test_split` function.
5. Convert the text data into numerical features using TF-IDF vectorization with Scikit-learn's `TfidfVectorizer`.
6. Train a linear SVM classifier using Scikit-learn's `LinearSVC` on the training data.
7. Evaluate the model's performance on the test data using the `score` method.
8. Make predictions on new articles by vectorizing the text and using the trained model's `predict` method.

## Example

The notebook includes an example where a random article from the test set is selected, and the trained model is used to predict whether it is real or fake news. The prediction is then compared with the actual label to verify the model's accuracy.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
