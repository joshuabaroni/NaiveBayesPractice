# NaiveBayesPractice
Basic Naive Bayes AI approach written in Java

## Purpose of This Project
- Naive Bayes is one of the most fundamental AI algorithms in existence. It takes on a simple approach: Predict the class of a new input response instance based off of similar instances whose classes are known.

## Project Concept
### Overview
- Using probability as its primary tool, Naive Bayes compares previous responses and their correlation to a class output to predict what the final response of a new class will be.
- Any new output successfully predicted will serve as reinforcement learning to improve the accuracy of the algorithm's future predictions.

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

### Voting Model
- The current model build is trained to recognize a voter's political party preference, based on their answers to a 16-question poll.
- This build features a user console interface that not only presents its prediction based on the questionnaire but also asks the user to verify the accuracy of its outcome. The algorithm will then reweight the model according to the user's feedback.
- As the model's base algorithm is improved, it should more widely accept data from any context formatted into .arff files.

## Technologies
- Based off of analysis of the Weka Naive Bayes model: https://weka.sourceforge.io/doc.dev/weka/classifiers/bayes/NaiveBayes.html
