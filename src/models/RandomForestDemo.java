package models;

import java.io.File;
import java.io.FileInputStream;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.IOException;
import weka.classifiers.trees.RandomForest;

import utilities.Utils;

/**
 * @author Gowtham Girithar Srirangasamy
 *
 */
/**
 * @author Gowtham Girithar Srirangasamy
 *
 */
public class RandomForestDemo {

    /** file names are defined */
    public static final String TRAINING_DATA_SET_FILENAME = Utils.FILESPACE + "weight_height_train.arff";
    public static final String TESTING_DATA_SET_FILENAME = Utils.FILESPACE + "weight_height_test.arff";

    /**
     * This method is to load the data set.
     * 
     * @param fileName
     * @return
     * @throws IOException
     */
    public static Instances getDataSet(String fileName) throws IOException {
        /**
         * we can set the file i.e., loader.setFile("filename") to load the data
         */
        int classIdx = 1;
        /** the arffloader to load the arff file */
        ArffLoader loader = new ArffLoader();
        /** load the traing data */
//        loader.setSource(RandomForestDemo.class.getResourceAsStream(fileName));
        loader.setSource(new FileInputStream(new File(fileName)));
        /**
         * we can also set the file like loader3.setFile(new
         * File("test-confused.arff"));
         */
        // loader.setFile(new File(fileName));
        Instances dataSet = loader.getDataSet();
        /** set the index based on the data given in the arff files */
        dataSet.setClassIndex(classIdx);
        return dataSet;
    }

    /**
     * This method is used to process the input and return the statistics.
     * 
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {

        Instances trainingDataSet = getDataSet(TRAINING_DATA_SET_FILENAME);
        Instances testingDataSet = getDataSet(TESTING_DATA_SET_FILENAME);

        RandomForest forest = new RandomForest();
        forest.setNumFeatures(10);

        /** */
        forest.buildClassifier(trainingDataSet);
        /**
         * train the alogorithm with the training data and evaluate the algorithm with
         * testing data
         */
        Evaluation eval = new Evaluation(trainingDataSet);
        eval.evaluateModel(forest, testingDataSet);

        /** Print the algorithm summary */
        System.out.println("** Decision Tress Evaluation with Datasets **");
        System.out.println(eval.toSummaryString());
        System.out.print(" the expression for the input data as per alogorithm is ");
        System.out.println(forest);
        System.out.println(eval.toMatrixString());
        System.out.println(eval.toClassDetailsString());

    }
}
