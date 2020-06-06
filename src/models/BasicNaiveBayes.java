package models;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import Exceptions.UnsupportedFiletypeException;
import Exceptions.UntrainedModelException;
import org.apache.commons.io.FilenameUtils;
import utilities.Pair;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;

/**
 * NB Model I wrote to help me understand better how the NB algorithm works.
 * 
 * @author Josh Baroni
 *
 */
public class BasicNaiveBayes {

//    private int NUM_DATA_INST_PER_LAYER = 10; // TODO less datapoints : more extreme weighting
    private String[] inputKeys; // keys correspond to attribute
    private Double[] possibleClasses;
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
            Double[] rangeCategories;
            Double[] temp = model.get(key);
            double see_value = values.get(key)[i];

            // check for quantitative data
            if (values.get(key)[i] > 1.0) {
                rangeCategories = quanToQual(key);

                for (int j = 1; j < rangeCategories.length; j++) {
                    if (see_value > rangeCategories[j - 1] && see_value < rangeCategories[j]) {
                        if (j < rangeCategories.length / 2)
                            // find the halfway mark. median = 1.0, extremes = 0.0
                            see_value = 1.0 - (j / (rangeCategories.length / 2.0)); // ex: 3rd cat with size n=8: 1 - (1 - 3/4) = 3/4
                            // TODO find the frequencies of each category and weight membership in each category accordingly to account for mean
                        else
                            see_value = 1.0 - (Math.abs(1.0 - (j / (rangeCategories.length / 2.0)))); // ex: 5th cat with size n=8: 1 - (1 - 5/4) = 3/4
                    }
                }
            }

            if (!Double.isNaN(see_value)) {
                // TODO account for more than 2 values
                for (int j = 0; j < possibleClasses.length; j++) {
                    if (classification == possibleClasses[j]) { // check classification == i for all String i : class
                        temp[j] = (see_value + temp[j] * (numInstances - 1)) / numInstances;
                    }
                }
            }
            model.put(key, temp);
        }
    }

    private Double[] quanToQual(String key) {
        Pair<Double, Double> dimensions;
        Double min = Double.MAX_VALUE;
        Double max = Double.MIN_VALUE;
        Double[] vals = values.get(key);
        for (Double i : vals) {
            if (min > i)
                min = i;
            if (max < i)
                max = i;
        }
        dimensions = new Pair(max - min, 1 + (Math.log(vals.length) / Math.log(2)));
        double classWidth = dimensions.val1 / dimensions.val2;
        Double[] rangeCategories = new Double[dimensions.val2.intValue()];
        for (int k = 0; k < rangeCategories.length; k++) {
            // 1st cat to nth cat
            // if (within this rangeClass) then range value = (1/dimensions.val2) * which class
            rangeCategories[k] = (1.0 + k) / dimensions.val2; // lower bound == rangeCats[n-1]; upper bound == rangeCats[n]
        }
        return rangeCategories;
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

    /**
     * Returns double value of highest likelihood class
     * @param keyChain
     * @param responses
     * @return
     */
    public Double predictClass(String[] keyChain, Double[] responses) {
        // need keyChain because keys will not be in the same order as this.inputKeys
        int numClasses = possibleClasses.length;
        Double[] classProbs = new Double[numClasses];
        Arrays.fill(classProbs, 1.0 / numClasses); // fill all indices with equal probability
        for (int i = 0; i < responses.length; i++) { // TODO mod this to handle more than two inputs
            for (int j = 0; j < classProbs.length; j++) {// numClasses possible
                double model_j = model.get(keyChain[i])[j];
                if (responses[i] == 1.0) { // TODO if class i double val then finalProb[i] = (finalProb[i] + model_i) / 2
                    classProbs[j] = (classProbs[j] + model_j) / 2;
                } else if (responses[i] == 0.0) {
                    classProbs[j] = (classProbs[j] + (1 - model_j)) / 2;
                } else {
                    continue;
                }
            }
        }
        Double max = Double.MIN_VALUE;
        int maxIndex = 0;
        for (int i = 0; i < classProbs.length; i++)
            if (classProbs[i] > max) {
                maxIndex = i;
                max = classProbs[i];
            }
        return possibleClasses[maxIndex];
    }

    /**
     * Basic Naive Bayes approach works on true/false datasets
     * @return
     */
    public BasicNaiveBayes train() {
        // width first (instance iterator; covers all keywords in a model instance)
        for (int i = 0; i < values.get("Class").length; i++) {
            numInstances++;
            // depth second (keyword iterator; steps through each keyword in an instance
            Double[] thisInstance = new Double[inputKeys.length - 1];
            for (int j0 = 0; j0 < inputKeys.length; j0++) {
                if (inputKeys[j0].equalsIgnoreCase("class")) { // skip class attr
                    continue;
                } else {
                    thisInstance[j0] = values.get(inputKeys[j0])[i];
                }
            }
            Double classification = values.get(inputKeys[inputKeys.length - 1])[i];
            // translate quantitative data into qualitative data
            Double[] rangeCategories;
            if (model.size() == 0) { // cannot reweight effectively till numInstances > 2
                for (String j0 : inputKeys) {
                    double firstValue = values.get(j0)[i]; // need to check for NaN on the first instance
                    // check if value is qualitative. if quantitative, translate to qualitative (range-based classification)
                    if (Double.isNaN(firstValue))
                        firstValue = 1.0/possibleClasses.length;
                    else if (firstValue > 1.0) {
                        rangeCategories = quanToQual(j0);
                    }
                    for (Double k : possibleClasses) {
                        if (classification == k) {// isFirstClass
                            Double[] temp = new Double[possibleClasses.length];
                            Arrays.fill(temp, 1.0 / possibleClasses.length);
                            temp[0] = k; // TODO hardcoded 0 vs i? // first value is the first possible class
                            model.put(j0, temp);
                        }
                    }
                }
            } else { // reweight after every instance
                reweight(classification, i);
            }
        }
        return this;
    }

//-------------------------------Static Methods-------------------------------

    // ----------------private------------------

    private static Instances loadArff(File file) throws IOException {
        ArffLoader.ArffReader reader = new ArffLoader.ArffReader(new BufferedReader(new FileReader(file)));
        Instances data = reader.getData();
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    private static Instances loadCSV(File file) throws IOException {
        CSVLoader loader = new CSVLoader();
        loader.setSource(file);
        return loader.getDataSet();
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
            if (FilenameUtils.getExtension(file.getName()).equals("arff")) {
                data = loadArff(file);
            } else if (FilenameUtils.getExtension(file.getName()).equals("csv")) {
                data = loadCSV(file);
            } else {
                throw new UnsupportedFiletypeException();
            }

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
                Double finalProb = bnb.predictClass(attrArray, testDataPtsArray);
                // TODO alter to accomodate >2 categories
                if (data.get(outer).classValue() == finalProb) {
                    countCorrect++;
                } else {
                    countIncorrect++;
                }
            }
            System.out.println("Count correct: " + countCorrect + "\nCount incorrect: " + countIncorrect
            + "\nPercent Accuracy for model: " + String.format("%.1f", (countCorrect / (countCorrect + countIncorrect)) * 100.0) + "%");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Alter UI as needed per dataset
     * @param bnb
     * @param ui
     * @return
     */
    public static void voterPollUi(BasicNaiveBayes bnb, Scanner ui) {
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
        Double finalProb = bnb.predictClass(keyChain, responses);
//        Pair<String, Double> result = (finalProb[1] < finalProb[0]) ? new Pair<>("Republican", finalProb[0] / (finalProb[0] + finalProb[1]))
//                : new Pair<>("Democrat", finalProb[1] / (finalProb[0] + finalProb[1]));
//        System.out.println("Based on your responses, I am " + String.format("%.2f", result.val2 * 100)
//                + "% sure you identify as a " + result.val1);
    }

    public static BasicNaiveBayes naiveBayesBuilder(File file) throws IOException {
        Instances data;
        if (FilenameUtils.getExtension(file.getName()).equals("arff")) {
            data = loadArff(file);
        } else if (FilenameUtils.getExtension(file.getName()).equals("csv")) {
            data = loadCSV(file);
        } else {
            throw new UnsupportedFiletypeException();
        }

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

        Double[] temp = bnb.values.get("Class");
        ArrayList<Double> classes = new ArrayList<>();
        for (int i = 0; i < temp.length; i++) { // gets 1 of each variety of classes so we know what classes are possible
            boolean flag = false;
            for (int j = 0; j < i; j++) {
                if (temp[j].doubleValue() == temp[i].doubleValue()) {
                    flag = true;
                }
            }
            if (!flag) {
                classes.add(temp[i]);
            }
        }
        bnb.possibleClasses = new Double[classes.size()];
        classes.toArray(bnb.possibleClasses);
        return bnb;
    }

    public static void main(String[] args) {
        System.out.println(System.getProperty("user.dir"));
        args = new String[2];
        args[0] = utilities.Utils.FILESPACE + "labor_negotiations.arff"; // train
        args[1] = utilities.Utils.FILESPACE + "labor_negotiations.arff"; // test
        File testDataFile = new File(args[0]);
        BasicNaiveBayes bnb = null;
        try {
            bnb = naiveBayesBuilder(testDataFile).train();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Done training model: " + bnb.toString());
        Scanner ui = new Scanner(System.in);

        predictClassFile(bnb, new File(args[1]));

        ui.close();

    }
}
