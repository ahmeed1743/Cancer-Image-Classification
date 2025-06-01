# Cancer Image Classification Using EfficientNetB0

## Project Overview
This project implements a deep learning pipeline to classify cancer images into multiple categories using transfer learning with EfficientNetB0. It includes data preprocessing, model training, evaluation, and visualization of results.

## Dataset
- Images are organized in directories by class in separate `train` and `test` folders.
- The code generates dataframes of file paths and labels for loading images.

## Environment and Libraries
- Python 3.x
- TensorFlow and Keras for deep learning
- OpenCV, NumPy, Pandas for data handling
- Matplotlib and Seaborn for visualization
- Scikit-learn for evaluation metrics

## Pipeline
1. **Data Preparation**
   - Load image file paths and labels into Pandas DataFrames.
   - Use `ImageDataGenerator` to generate batches of images and labels with optional augmentation.

2. **Model Architecture**
   - Base model: EfficientNetB0 pretrained on ImageNet, excluding top layers.
   - Added Batch Normalization, Dense layers with L1 and L2 regularization, Dropout, and a final softmax classification layer.

3. **Training**
   - Compiled with Adamax optimizer and categorical cross-entropy loss.
   - Trained for 10 epochs using training and validation generators.

4. **Evaluation**
   - Plot training and validation accuracy and loss.
   - Calculate and display confusion matrix for test data.
   - Print classification report for detailed performance metrics.

## Usage
1. Organize your dataset into `train` and `test` folders with subfolders for each class.
2. Update the `train_data_dir` and `test_data_dir` paths in the script.
3. Run the script to train the model and view evaluation results.

## Results
- Training and validation accuracy and loss are plotted for each epoch.
- Confusion matrix visualization highlights model performance across classes.
- Final evaluation metrics include loss and accuracy on both training and test datasets.

## Notes
- The model uses transfer learning which can be fine-tuned by freezing or unfreezing the base model layers.
- Hyperparameters such as learning rate, dropout rate, and regularization strength can be adjusted for improved performance.

## References
- [TensorFlow EfficientNet Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet)
- [Keras ImageDataGenerator](https://keras.io/api/preprocessing/image/)
- [Confusion Matrix Explanation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

## Author
Ahmed Elsayed

## License
MIT License
