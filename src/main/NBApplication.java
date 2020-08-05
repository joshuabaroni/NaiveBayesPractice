package main;

import java.io.File;
import java.io.IOException;

import models.BasicNaiveBayes;

public class NBApplication {
    /**
     * Main method; inits and provides arguments.
     * @param args first argument = destination of train file, second argument = destination of test file
     */
    public static void deprecated() {
        String[] args = new String[2];
        args[0] = utilities.Utils.FILESPACE + "/sick/sick.arff"; // train
        args[1] = utilities.Utils.FILESPACE + "/sick/sick.arff"; // test
        File testDataFile = new File(args[0]);
        BasicNaiveBayes bnb = null;
        try {
            bnb = BasicNaiveBayes.naiveBayesBuilder(testDataFile).train();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Done training model: " + bnb.toString());

        System.out.println(BasicNaiveBayes.predictClassFile(bnb, new File(args[1])));
    }
}
