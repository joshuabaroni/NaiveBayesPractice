package models;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/**
 * Handles basic yes/no inputs
 * 
 * @author Josh Baroni
 *
 */
public class BasicNaiveBayes {

//    private int NUM_DATA_INST_PER_LAYER = 10; // TODO less datapoints : more extreme weighting
    private String[] inputKeys; // keys correspond to attribute
    private Map<String, Double[]> values; // keys are paired with their respective values. R = 1.0, D = 0.0
    public Map<String, Double[]> model; // value[0] = likelihood of 'y' if class = true, 'n' if class = false
    public int numInstances = 0;
//--------------------------------Constructors--------------------------------    

    public BasicNaiveBayes() {
        values = new HashMap<>();
        model = new HashMap<>();
    }

//------------------------------Instance Methods------------------------------

    // --------------------private---------------------

    private void reweight(double classification, int i) throws UntrainedModelException { // TODO reweights model after every layer
        for (String key : inputKeys) {
            Double[] temp = model.get(key);
            double see_value = values.get(key)[i];
            if (!Double.isNaN(see_value)) {
                if (classification == 1.0) {
                    temp[0] = (see_value + temp[0] * (numInstances - 1)) / numInstances;
                    temp[1] = (Math.abs(1 - see_value) + temp[1] * (numInstances - 1)) / numInstances;
                } else if (classification == 0.0) {
                    temp[1] = (see_value + temp[1] * (numInstances - 1)) / numInstances;
                    temp[0] = (Math.abs(1 - see_value) + temp[0] * (numInstances - 1)) / numInstances;
                }
            }
            model.put(key, temp);
        }
    }

    // --------------------public----------------------

    public Map<String, Double[]> getModel() {
        return model;
    }

    public BasicNaiveBayes buildMap(String[] inputs, double[][] dataPts) {
        this.inputKeys = inputs;
        values = new HashMap<>();
        for (int i = 0; i < dataPts[0].length; i++) {
            Double[] temp = new Double[dataPts.length];
            for (int j = 0; j < dataPts.length; j++) {
                temp[j] = dataPts[j][i];
            }
            values.put(inputKeys[i], temp);
        }
        return this;
    }

    public Double[] predictClass(String[] keyChain, Double[] responses) {
        // need keyChain because keys will not be in the same order as this.inputKeys
        Double[] finalProb = {0.5, 0.5};
        for (int i = 0; i < responses.length; i++) {
            double model_0 = model.get(keyChain[i])[0];
            double model_1 = model.get(keyChain[i])[1];
            if (responses[i] == 1.0) {
                finalProb[0] = (finalProb[0] + model_0) / 2;
                finalProb[1] = (finalProb[1] + model_1) / 2;
            } else if (responses[i] == 0.0) {
                finalProb[0] = (finalProb[0] + (1 - model_0)) / 2;
                finalProb[1] = (finalProb[1] + (1 - model_1)) / 2;
            } else {
                continue;
            }
        }
        return finalProb;
    }

    public BasicNaiveBayes train() {
        // width first (instance iterator; covers all keywords in a model instance)
        for (int i = 0; i < values.get("Class").length; i++) {
            numInstances++;
            // depth second (keyword iterator; steps through each keyword in an instance
            Double[] thisInstance = new Double[inputKeys.length - 1];
            for (int j0 = 0; j0 < inputKeys.length; j0++) {
                if (inputKeys[j0].equalsIgnoreCase("class")) {
                    continue;
                } else {
                    thisInstance[j0] = values.get(inputKeys[j0])[i];
                }
            }
            Double classification = values.get(inputKeys[inputKeys.length - 1])[i];
            if (i == 0) { // cannot reweight effectively till numInstances > 2
                for (String j0 : inputKeys) {
                    double firstValue = values.get(j0)[i]; // need to check for NaN on the first instance
                    if (Double.isNaN(firstValue)) {
                        firstValue = 0.5;
                    }
                    if (classification == 1.0) // isFirstClass
                        model.put(j0, new Double[] { firstValue, Math.abs(1 - firstValue) });
                    else if (classification == 0.0) // !isFirstClass
                        model.put(j0, new Double[] { Math.abs(1 - firstValue), firstValue });
                }
            } else { // reweight after every instance
                reweight(classification, i);
            }
            if (/*i == 2 || i == 28 || i == 256 ||*/ i == 434) { // to check at random intervals that the model algo is correct
                System.out.println("instanceCheckpoint");
            }
        }
        return this;
    }

//-------------------------------Static Methods-------------------------------    

    // ----------------private------------------

    private static Instances loadArff(File file) throws FileNotFoundException, IOException {
        ArffLoader.ArffReader reader = new ArffLoader.ArffReader(new BufferedReader(new FileReader(file)));
        Instances data = reader.getData();
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    // ----------------public-------------------

    /**
     * Runs testfile alongside trained model
     * 
     * @param bnb
     * @param file
     * @return
     * @throws FileNotFoundException
     * @throws IOException
     */
    public static void predictClassFile(BasicNaiveBayes bnb, File file) {
        Instances data;
        try {
            data = loadArff(file);

            List<Attribute> attr = Collections.list(data.enumerateAttributes());

            String[] attrArray = new String[attr.size() + 1]; // last is for class
            Double[] testDataPtsArray = new Double[attr.size()];

            double countCorrect = 0.0, countIncorrect = 0.0;
            for (int outer = 0; outer < data.size(); outer++) {
                double[] temp = data.get(outer).toDoubleArray();

                for (int i = 0; i < temp.length - 1; i++) // deep copy double[] to Double[]
                    testDataPtsArray[i] = temp[i];

                for (int i = 0; i < attr.size(); i++) {
                    attrArray[i] = attr.get(i).name();
                }

                attrArray[attrArray.length - 1] = "Class";
                Double[] finalProb = bnb.predictClass(attrArray, testDataPtsArray);
                if (finalProb[0] > finalProb[1] && data.get(outer).classValue() == 1.0
                        || finalProb[0] < finalProb[1] && data.get(outer).classValue() == 0.0) {
                    countCorrect++;
                } else {
                    countIncorrect++;
                }
            }
            System.out.println("Count correct: " + countCorrect + "\nCount incorrect: " + countIncorrect
            + "\nPercent Accuracy for model: " + (countCorrect / countCorrect + countIncorrect) + "%");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static Double[] predictClassUi(BasicNaiveBayes bnb, Scanner ui) {
        System.out.println("Respond to the following prompts by typing 'y' or 'n' and hitting the <ENTER> key.");
        String[] keyChain = bnb.model.keySet().toArray(new String[bnb.model.keySet().size()]);
        Double[] responses = new Double[keyChain.length - 1];
        for (int i = 0; i < responses.length; i++) {
            if (keyChain[i].equals("Class"))
                responses[i] = Double.NaN;
            else {
                System.out.print(keyChain[i] + ": ");
                String response = ui.nextLine();
                if (response.equalsIgnoreCase("y")) {
                    responses[i] = 1.0;
                } else if (response.equalsIgnoreCase("n")) {
                    responses[i] = 0.0;
                } else {
                    responses[i] = 0.5;
                }
            }
        }
        return bnb.predictClass(keyChain, responses);
    }

    public static BasicNaiveBayes naiveBayesBuilder(File file) throws IOException {
        Instances data = loadArff(file);

        BasicNaiveBayes bnb = new BasicNaiveBayes();
        List<Attribute> attr = Collections.list(data.enumerateAttributes());

        // changes A<S> attr to S[]
        // changes A<S[]> to S[][]
        String[] attrArray = new String[attr.size() + 1]; // last is for class
        double[][] dataPtsArray = new double[data.size()][attr.size()];

        int index = 0;
        while (index < data.size()) {
            dataPtsArray[index] = data.get(index).toDoubleArray();
            index++;
        }

        for (int i = 0; i < attr.size(); i++) {
            attrArray[i] = attr.get(i).name();
        }

        attrArray[attrArray.length - 1] = "Class";
        bnb.buildMap(attrArray, dataPtsArray);
        return bnb;
    }

    public static void main(String[] args) {
        System.out.println(System.getProperty("user.dir"));
        args = new String[2];
        args[0] = "src/testdata/voting_train.arff";
        args[1] = "src/testdata/voting_test.arff";
        File testDataFile = new File(args[0]);
        BasicNaiveBayes bnb = null;
        try {
            bnb = naiveBayesBuilder(testDataFile).train();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Done training model: " + bnb.toString());
        Scanner ui = new Scanner(System.in);

        Pair<String, Double> result;
        predictClassFile(bnb, new File(args[1]));
//            Double[] finalProb = predictClassUi(bnb, ui);
//            result = (finalProb[1] < finalProb[0]) ? new Pair<>("Republican", finalProb[0] / (finalProb[0] + finalProb[1]))
//                    : new Pair<>("Democrat", finalProb[1] / (finalProb[0] + finalProb[1]));
//            System.out.println("Based on your responses, I am " + String.format("%.2f", result.val2 * 100)
//                    + "% sure you identify as a " + result.val1);

        ui.close();

    }
}
