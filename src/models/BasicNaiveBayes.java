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
//    private String[] classes = {republican, democrat}; // TODO
    private String[] inputKeys; // keys correspond to attribute
    private Map<String, Double[]> values; // keys are paired with their respective values. R = 1.0, D = 0.0
    public int n;
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
            Double[] temp = model.get(i);
            if ((temp[0] > temp[1] && isCorrect) || (temp[0] < temp[1] && !isCorrect)) {
                temp[0] *= 1 + (1 / model.size());
                temp[1] /= 1 + (1 / model.size());
            } else if ((temp[0] < temp[1] && isCorrect) || (temp[0] > temp[1] && !isCorrect)) {
                temp[1] *= 1 + (1 / model.size());
                temp[0] /= 1 + (1 / model.size());
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

    public Pair<String, Double> predictClass(String[] keyChain, Double[] responses) {
        // need keyChain because keys will not be in the same order as this.inputKeys
        Double[] finalProb = { 0.5, 0.5 };
        double yes = 0;
        double no = 0;
        for (int i = 0; i < responses.length; i++) {
            if (responses[i] == 1.0) { // factor in 'yes' responses
//                if (this.model.get(keyChain[i])[0] < this.model.get(keyChain[i])[1]) {
                double tempDem = this.model.get(keyChain[i])[1];
                double tempRep = this.model.get(keyChain[i])[0];
                finalProb[1] += tempDem;
                finalProb[0] += tempRep;
                yes++;
            } else if (responses[i] == 0.0) {
//                } else if (this.model.get(keyChain[i])[0] > this.model.get(keyChain[i])[1]) {
                double tempDem = 1.0 - this.model.get(keyChain[i])[1];
                double tempRep = 1.0 - this.model.get(keyChain[i])[0];
                finalProb[0] += tempDem;
                finalProb[1] += tempRep;
                no++;
            } else {
                continue;
            }
        }
        finalProb[0] *= (yes / (yes + no)); // P(C|A)
        finalProb[1] *= (no / (yes + no));
        // get results
        Pair<String, Double> result;
        // TODO generalize category_0, category_1
        result = (finalProb[1] < finalProb[0]) ? new Pair<>("Republican", finalProb[0] / (finalProb[0] + finalProb[1]))
                : new Pair<>("Democrat", finalProb[1] / (finalProb[0] + finalProb[1]));
        return result;
    }

    public BasicNaiveBayes train() throws InvalidDataValueException {
        double sumValueClass, sumValueNotClass; // 0th index if final = true, 1st if final = false
        int numClass = 0, numNotClass = 0;
        for (int i = 0; i < inputKeys.length - 1; i++) { // test instance iterator
            sumValueClass = 0.0;
            sumValueNotClass = 0.0;

            for (int j1 = 0; j1 < values.get(inputKeys[i]).length; j1++) { // data points in this test instance
                String str = inputKeys[inputKeys.length - 1];
                Double classification = values.get(str)[j1];
                if (classification == 1.0) {
                    if (values.get(inputKeys[i])[j1] == 1.0) { // if true
                        sumValueClass += values.get(inputKeys[i])[j1];
                        numClass++;
                    } else if (values.get(inputKeys[i])[j1] == 0.0) { // if false
//                        sumValueClass += values.get(inputKeys[i])[j1];
                        numClass++;
                    } else {
                        if (Double.isNaN(values.get(inputKeys[i])[j1])) {
//                            sumValueClass += 0.5;
                        } else {
                            throw new InvalidDataValueException();
                        }
                    }
                } else if (classification == 0.0) {
                    if (values.get(inputKeys[i])[j1] == 1.0) { // if true
                        sumValueNotClass += values.get(inputKeys[i])[j1];
                        numNotClass++;
                    } else if (values.get(inputKeys[i])[j1] == 0.0) { // if false
//                        sumValueNotClass += values.get(inputKeys[i])[j1];
                        numNotClass++;
                    } else {
                        if (Double.isNaN(values.get(inputKeys[i])[j1])) {
//                            sumValueNotClass += 0.5;
                        } else {
                            throw new InvalidDataValueException();
                        }
                    }
                } else {
                    throw new InvalidDataValueException();
                }
            }
//            System.out.println("");
            Double[] thisOut = { sumValueClass / numClass, sumValueNotClass / numNotClass };
            model.put(inputKeys[i], thisOut); // thisOut[0] = p(Yes|R), thisOut[1] = p(No|R)
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
        return bnb.predictClass(attrArray, testDataPtsArray);
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
        return bnb.predictClass(keyChain, responses);
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
//        Pair<String, Double> result = predictClassUi(bnb, ui);
            System.out.println("Based on your responses, I am " + String.format("%.2f", result.val2 * 100)
                    + "% sure you identify as a " + result.val1);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        boolean flag = false;
        while (!flag) {
            System.out.print("Was this output correct? ");
            if (ui.next().equals("y")) {
                bnb.reweight(true);
                flag = true;
            } else if (ui.next().equals("n")) {
                bnb.reweight(false);
                flag = true;
            } else {
                System.out.println("Please enter a valid response <y> or <n>.");
            }
        }

        ui.close();

    }
}
