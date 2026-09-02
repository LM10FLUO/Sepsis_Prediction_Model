# Sepsis Prediction using Machine Learning Approaches

## Project Description

**IDEA:**

Develop an AI-powered system that uses machine learning (ML) to continuously analyse ICU patient data and predict the likelihood of sepsis before it is clinically recognised.
The system would be trained on historical ICU datasets containing time-series patient data, enabling it to learn the patterns and trends that typically precede the onset of sepsis.
Once trained, the model would process real-time patient data, continuously updating a dynamic sepsis risk score. If the predicted probability of sepsis exceeds a predefined threshold, the system would automatically alert clinicians, allowing for earlier intervention and potentially improving patient outcomes.

**WHAT THE APPLICATION DOES**

Given a patient's current vital logs, the model outputs a prediction of the likelihood (as a probability) of the patient developing sepsis in the next 12 hours. We define 1.0 as the patient will definitely develop sepsis in the next 12 hours (positive cases) and 0.0 as a patient will not develop sepsis at any time in the next 12 hours.

**WHY WE HAVE USED THIS APPROACH**

Here, we have used the physionet training data from 2019 in order to training an LSTM (Long Short Term Memory) neural network model. We have chosen this model as we will be receiving a series of logs for each patient during their admission to the ICU - we want to see how the vitals of the patient changes over time to foresee whether they are likely to develop sepsis 12 hours prior to possible onset of the disease.

**CHALLENGES FACED DURING DEVELOPMENT**

As mispredicting the onset of sepsis is much more serious than the misprediction of a patient not developing sepsis (serious risk to the patient's life vs directing unnecessary attention to particular patients), a careful balance needed to be struck between the two. Initially, during training, there was a very high rate of false positives. Given what has been said previously, this is extremely dangerous, especially in such life or death situations. As you will see through the confusion matrices ("cm_threshold03.png" and "cm_threshold05.png") produced by the final model, it was impossible to completely eradicate false positives and false negatives from the test set. Additionally, the threshold by which a doctor may classify a patient as "likely" to develop sepsis is subjective; hence, this model should be used as a supplement rather than a clear cut probability for developing the disease. 

Another problem was the model overfitting too much to the training set and not generalising well with the cross-validation (cv) and test sets, as you can see through the figures "cv_loss_new_model.png" and "training_loss_new_model.png". This was dealt with by early stopping mechanisms in place during training, balancing model accuracy with the training data and the test data for two different hospitals, allowing the model to generalise this pattern. However, as discussed and for this reason, the model is not 100% accurate though, as seen by the confusion matrices, reasonably high accuracy.

**FEATURES FOR THE FUTURE**

With future development, it would be useful to explore different machine learning approaches, exploring different neural network architectures to achieve a more accurate and reliable model. Additionally, implementation of a GUI which allows doctors to seemlessly feed in new real-time patient data would allow the model to be used in hospitals and ICU units.

## How to Install and Run the Project

To run the project, please ensure that the following libraries are installed:
- NumPy (downloading the datasets will require versions before 2.x, however, training the model and testing will require version 2.5.2, though it may run a warning in the terminal)
- PyTorch
- Matplotlib (to generate graphs)
- Scikit-learn

Additionally, please download the training sets from physionet and save this in your home directory. This can be done by running the following command(s) in your terminal:
- (MacOS) with homebrew installed, run "brew install wget" followed by "wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/" in the terminal
- (Windows) run "wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/" in the terminal

To run the project and train and test the model yourself, please do the following in order:
1. Run the "downloads.py" script to clean and preprocess the downloaded training data to use to train the model
2. Run "normalisation.py" to calculate mean and std for use in training the model
3. Run "training.py"
4. Run "testing.py" which will produce a confusion matrix visualising the model's effectiveness on the test set

# How to Use the Project

To use the trained neural network, please ensure that any data used is in the .psv format used by the physionet training data and that this data is preprocessed using the mechanisms used in "datasets.py" - this can be done by instantiating the DataPreprocessor class and running the process_single_file method, providing the file arguments as a tuple (input_dir, output_dir and overwrite=True). Then provide the absolute file path to this cleaned file to "prediction.py" when run to obtain a probability for the likelihood of the patient to develop sepsis in the next 12h.

# Credits
Felix Luo :)
Github: https://github.com/LM10FLUO




