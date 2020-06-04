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

    private void reweight(boolean isCorrect) throws UntrainedModelException { // TODO reweights model after every layer
        if (model == null) {
            throw new UntrainedModelException();
        }
        for (String i : inputKeys) {
            if (i.equalsIgnoreCase("class")) {
                continue;
            } else {
                Double[] temp = model.get(i);
                if (i.equalsIgnoreCase("crime")) {
                    System.out.println("reweightChkpt");
                }
                double p0 = 1.0 + (model.get(i)[0] / model.get(i).length);
                double p1 = 1.0 + (model.get(i)[1] / model.get(i).length);
                if ((temp[0] > temp[1] && isCorrect) || (temp[0] < temp[1] && !isCorrect)) {
                    temp[0] *= (1.0 + (model.get(i)[0] / model.get(i).length));
                    temp[1] /= (1.0 + (model.get(i)[1] / model.get(i).length));
                } else if ((temp[0] < temp[1] && isCorrect) || (temp[0] > temp[1] && !isCorrect)) {
                    temp[1] *= (1.0 + (model.get(i)[1] / model.get(i).length));
                    temp[0] /= (1.0 + (model.get(i)[0] / model.get(i).length));
                }
                model.put(i, temp);
            }
        }
        // re-weight after layer of size "NUM_DATA_INST_PER_LAYER"
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
        System.out.println(finalProb);
        return finalProb;
    }

    public BasicNaiveBayes train() throws InvalidDataValueException {
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
                for (String j1 : inputKeys) {
//                    if (j1.equalsIgnoreCase("synfuels-corporation-cutback")) {
//                        System.out.println("trainChkpt");
//                    }
                    Double[] temp = model.get(j1);
                    double see_value = values.get(j1)[i];
                    if (!Double.isNaN(see_value)) {
                        if (classification == 1.0) {
                            temp[0] = (see_value + temp[0] * (numInstances - 1)) / numInstances;
                            temp[1] = (Math.abs(1 - see_value) + temp[1] * (numInstances - 1)) / numInstances;
                        } else if (classification == 0.0) {
                            temp[1] = (see_value + temp[1] * (numInstances - 1)) / numInstances;
                            temp[0] = (Math.abs(1 - see_value) + temp[0] * (numInstances - 1)) / numInstances;
                        }
                    }
                    model.put(j1, temp);
                }
//                Double[] temp = this.predictClass(inputKeys, thisInstance); TODO reweight() refactor?
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
     * designed to test only the first line in a <test>.arff
     * 
     * @param bnb
     * @param file
     * @return
     * @throws FileNotFoundException
     * @throws IOException
     */
    public static /*Pair<String, Double>*/ void predictClassFile(BasicNaiveBayes bnb, File file)
            throws FileNotFoundException, IOException {
        Instances data = loadArff(file);
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
        // get results
//        Pair<String, Double> result;
//        result = (finalProb[1] < finalProb[0]) ? new Pair<>("Republican", finalProb[0] / (finalProb[0] + finalProb[1]))
//                : new Pair<>("Democrat", finalProb[1] / (finalProb[0] + finalProb[1]));
//        return result;
    }

    public static Pair<String, Double> predictClassUi(BasicNaiveBayes bnb, Scanner ui) {
        System.out.println("Respond to the following prompts by typing 'y' or 'n' and hitting the <ENTER> key.");
        String[] keyChain = bnb.model.keySet().toArray(new String[bnb.model.keySet().size()]);
        Double[] responses = new Double[keyChain.length - 1];
        for (int i = 0; i < responses.length; i++) {
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
        Double[] finalProb = bnb.predictClass(keyChain, responses);
        // get results
        Pair<String, Double> result;
        result = (finalProb[1] < finalProb[0]) ? new Pair<>("Republican", finalProb[0] / (finalProb[0] + finalProb[1]))
                : new Pair<>("Democrat", finalProb[1] / (finalProb[0] + finalProb[1]));
        return result;
    }

    public static BasicNaiveBayes naiveBayesBuilder(File file) throws IOException {
        Instances data = loadArff(file);

        BasicNaiveBayes bnb = new BasicNaiveBayes();
        List<Attribute> attr = Collections.list(data.enumerateAttributes());
//        ArrayList<String[]> dataPts = new ArrayList<>();

        // changes A<S> attr to S[]
        // changes A<S[]> to S[][]
        String[] attrArray = new String[attr.size() + 1]; // last is for class
        double[][] dataPtsArray = new double[data.size()][attr.size()];

        int index = 0;
        while (index < data.size()) {
            dataPtsArray[index] = data.get(index).toDoubleArray();
//            System.out.println("Current line #: " + reader.getLineNo());
            index++;
        }

        for (int i = 0; i < attr.size(); i++) {
            attrArray[i] = attr.get(i).name();
//            dataPtsArray[i] = dataPts.get(i);
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
        } catch (InvalidDataValueException | IOException e) {
            e.printStackTrace();
        }
        System.out.println("Done training model: " + bnb.toString());
        Scanner ui = new Scanner(System.in);
        // TODO load from arff

        Pair<String, Double> result;
        try {
            /*result = */predictClassFile(bnb, new File(args[1]));
//            result = predictClassUi(bnb, ui);
//            System.out.println("Based on your responses, I am " + String.format("%.2f", result.val2 * 100)
//                    + "% sure you identify as a " + result.val1);
        } catch (IOException e) {
            e.printStackTrace();
        }
//        boolean flag = false;
//        while (!flag) {
//            System.out.print("Was this output correct? ");
//            String next = ui.next();
//            if (next.equals("y")) {
//                bnb.reweight(true);
//                flag = true;
//            } else if (next.equals("n")) {
//                bnb.reweight(false);
//                flag = true;
//            } else {
//                System.out.println("Please enter a valid response <y> or <n>.");
//            }
//        }
//        System.out.println("Thank you for your feedback. Good luck in the polls!");

        ui.close();

    }
}
