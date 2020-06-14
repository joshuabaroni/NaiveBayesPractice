# NaiveBayesPractice
Basic Naive Bayes ML model written in Java

## Purpose of This Project
- Naive Bayes is a stepping stone algorithm to understand how to effectively utilize machine learning. It takes on a simple approach: Predict the class of a new input response instance based off of similar instances whose classes are known.
### Bayes Theorem
- P(A|C) = [P(C|A)*P(A)]/[P(C)]
  - Posterior: P(A | C)
  - Likelihood: P(C | A)
  - Prior: P(A)
  - Evidence: P(C)
- False Positive - If a dog barks for no good reason, you will not act on it
- P(A) Prior Probability - Describes the degree to which we believe the model accurately describes reality based on all prior information
- P(C|A) Likelihood - Describes how well the model predicts the data
- P(C) Normalizing Constant - The constant that makes the posterior density integrate to one
- P(A|C) Posterior Probability - Represents the degree to which we believe a given model accurately describes the situation given the available data and all of our prior information

## Project Concept
### Overview
- Using probability as its primary tool, Naive Bayes compares previous responses and their correlation to a class output to predict what the final response of a new class will be.
- Any new output successfully predicted will serve as reinforcement learning to improve the accuracy of the algorithm's future predictions.
- The current model can be trained with reasonable accuracy (60-80% on qualitative and 80-95% on quantitative data) on any .arff dataset

### Labor Negotiation Poll
- This dataset trains the model to recognize acceptable compensations for labor.
- Accuracy with this dataset: ~80% correct

### Iris
- The Iris flower data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems.
- It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species.
- The data set consists of samples from each of three species of Iris (Iris Setosa, Iris virginica, and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
- Accuracy with this dataset: 95% correct

### Voting Data
- The current model can be trained to recognize a voter's political party preference, based on their answers to a 16-question poll.
- This build features a user console interface that not only presents its prediction based on the questionnaire but also asks the user to verify the accuracy of its outcome. The algorithm will then reweight the model according to the user's feedback.
  - The voterPollUi() method is a public static class callable within the BasicNaiveBayes class
- Training data used is from the [1984 voter census](https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records)
- Accuracy with this dataset: ~85% correct

### Other datasets are included in the file
- Model handles most typical NB classification examples well, with at least 70% accuracy

## Technologies
- Based off of analysis of the [Weka Naive Bayes model](https://weka.sourceforge.io/doc.dev/weka/classifiers/bayes/NaiveBayes.html)
- Taken from the [UCI machine learning repository](https://archive.ics.uci.edu/ml/index.php)
