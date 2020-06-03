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

//--------------------------------Constructors--------------------------------    

    /**
     * Builds empty key map to prep for train data insertion
     * 
     * @param inputKeys
     */

    public BasicNaiveBayes() {
        values = new HashMap<>();
        model = new HashMap<>();
    }

//------------------------------Instance Methods------------------------------

    // --------------------private---------------------

    private void reweight(boolean isCorrect) throws UntrainedModelException { // reweights this set after every layer
        if (model == null) {
            throw new UntrainedModelException();
        }
        for (String i : inputKeys) {
            if (i.equalsIgnoreCase("class")) {
                continue;
            } else {
                Double[] temp = model.get(i);
                if ((temp[0] > temp[1] && isCorrect) || (temp[0] < temp[1] && !isCorrect)) {
                    temp[0] *= 1.0 + (1.0 / model.get(i).length);
                    temp[1] /= 1.0 + (1.0 / model.get(i).length);
                } else if ((temp[0] < temp[1] && isCorrect) || (temp[0] > temp[1] && !isCorrect)) {
                    temp[1] *= 1.0 + (1.0 / model.get(i).length);
                    temp[0] /= 1.0 + (1.0 / model.get(i).length);
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
        Double[] finalProb = { 0.5, 0.5 };
        double yes = 0;
        double no = 0;
        for (int i = 0; i < responses.length; i++) {
            if (Double.isNaN(responses[i])) {
                continue;
            } else if (responses[i] == 1.0) { // factor in 'yes' responses
//                if (this.model.get(keyChain[i])[0] < this.model.get(keyChain[i])[1]) {
                double temp_1 = this.model.get(keyChain[i])[1];
                double temp_0 = this.model.get(keyChain[i])[0];
                if (Double.isNaN(temp_1) || Double.isNaN(temp_0)) {
                    continue;
                } else {
                    finalProb[1] += temp_1;
                    finalProb[0] += temp_0;
                    yes++;
                }
            } else if (responses[i] == 0.0) {
//                } else if (this.model.get(keyChain[i])[0] > this.model.get(keyChain[i])[1]) {
                double temp_1 = 1.0 - this.model.get(keyChain[i])[1];
                double temp_0 = 1.0 - this.model.get(keyChain[i])[0];
                if (Double.isNaN(temp_1) || Double.isNaN(temp_0)) {
                    continue;
                } else {
                    finalProb[0] += temp_1;
                    finalProb[1] += temp_0;
                    no++;
                }
            } else {
                continue;
            }
        }
        finalProb[0] *= (yes / (yes + no)); // P(C|A)
        finalProb[1] *= (no / (yes + no));
        return finalProb;
    }

    public BasicNaiveBayes train() throws InvalidDataValueException {
        // width first (instance iterator; covers all keywords in a model instance)
        for (int i = 0; i < values.get("Class").length; i++) {
            // depth second (keyword iterator; steps through each keyword in an instance
            Double[] thisInstance = new Double[inputKeys.length - 1];
            for (int j0 = 0; j0 < inputKeys.length; j0++) {
                if (inputKeys[j0].equalsIgnoreCase("class")) {
                    continue;
                } else {
                    thisInstance[j0] = values.get(inputKeys[j0])[i];
                }
            }
            // reweight after every instance
            Double classification = values.get(inputKeys[inputKeys.length - 1])[i];
            if (i > 0) { // cannot reweight on first instance
                Double[] temp = this.predictClass(inputKeys, thisInstance);
                double p1 = (temp[1] / (temp[0] + temp[1]));
                double p0 = (temp[0] / (temp[0] + temp[1]));
                if (classification == 1.0 && p0 > p1) {
                    this.reweight(true);
                } else {//if (classification == 0.0 && p0 > p1) {
                    this.reweight(false);
                }
//                else {
//                    throw new InvalidDataValueException(); // TODO throw if high inaccuracy
//                }
            } else {
                for (String j1 : inputKeys) {
                    if (classification == 1.0) // isFirstClass
                        model.put(j1, new Double[] { values.get(j1)[i], Math.abs(1 - values.get(j1)[i]) });
                    else if (classification == 0.0) // !isFirstClass
                        model.put(j1, new Double[] { Math.abs(1 - values.get(j1)[i]), values.get(j1)[i] });
                }
            }
            if (i == 28 || i == 256 || i == 432) {
                System.out.println("checkpoint");
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
    public static Pair<String, Double> predictClassFile(BasicNaiveBayes bnb, File file)
            throws FileNotFoundException, IOException {
        Instances data = loadArff(file);
        List<Attribute> attr = Collections.list(data.enumerateAttributes());

        String[] attrArray = new String[attr.size() + 1]; // last is for class
        Double[] testDataPtsArray = new Double[attr.size()];

        double[] temp = data.get(0).toDoubleArray();

        for (int i = 0; i < temp.length - 1; i++) // deep copy double[] to Double[]
            testDataPtsArray[i] = temp[i];

        for (int i = 0; i < attr.size(); i++) {
            attrArray[i] = attr.get(i).name();
        }

        attrArray[attrArray.length - 1] = "Class";
        Double[] finalProb = bnb.predictClass(attrArray, testDataPtsArray);
        // get results
        Pair<String, Double> result;
        result = (finalProb[1] < finalProb[0]) ? new Pair<>("Republican", finalProb[0] / (finalProb[0] + finalProb[1]))
                : new Pair<>("Democrat", finalProb[1] / (finalProb[0] + finalProb[1]));
        return result;
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
            result = predictClassFile(bnb, new File(args[1]));
//            result = predictClassUi(bnb, ui);
            System.out.println("Based on your responses, I am " + String.format("%.2f", result.val2 * 100)
                    + "% sure you identify as a " + result.val1);
        } catch (IOException e) {
            e.printStackTrace();
        }
        boolean flag = false;
        while (!flag) {
            System.out.print("Was this output correct? ");
            String next = ui.next();
            if (next.equals("y")) {
                bnb.reweight(true);
                flag = true;
            } else if (next.equals("n")) {
                bnb.reweight(false);
                flag = true;
            } else {
                System.out.println("Please enter a valid response <y> or <n>.");
            }
        }
        System.out.println("Thank you for your feedback. Good luck in the polls!");

        ui.close();

    }
}
