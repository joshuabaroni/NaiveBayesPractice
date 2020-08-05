package models;

import Exceptions.UnsupportedFiletypeException;
import Exceptions.UntrainedModelException;
import org.apache.commons.io.FilenameUtils;
import utilities.Pair;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;

import java.io.*;
import java.util.*;

/**
 * NB Model I wrote to help me understand better how the NB algorithm works.
 * Ideas for improvement:
 *  - Run this alongside a KNN algo and pick the optimal output
 *  - Change from naive to comparison of input | input in addition to input | output
 * @author Josh Baroni
 */
public class BasicNaiveBayes {

    private String[] inputKeys; // keys correspond to attribute
    private Double[] possibleClasses;
    private Map<String, Double[]> rangeCategories; // [key/Attr][range] class not needed
    private Map<String, Double[][]> rangeFrequencies; // [key/Attr][class][frequencyOfEachRange]
    private Map<String, Double[]> values; // keys are paired with their respective values. R = 1.0, D = 0.0
    private Boolean[] isQuan; // if true, this attribute is quantitative, else this attribute is qualitative
    private String classKey;
    public int numInstances = 0;
//--------------------------------Constructors--------------------------------    

    public BasicNaiveBayes(String classKey) {
        values = new HashMap<>();
        this.classKey = classKey;
    }

//------------------------------Instance Methods------------------------------

    // --------------------private---------------------

    /**
     * Weights data points in model after every training data instance
     * @param classification
     * @param i
     * @throws UntrainedModelException if model has not been trained (train() not called)
     */
    private void reweight(double classification, int i) throws UntrainedModelException {
        for (int keyIndex = 0; keyIndex < inputKeys.length; keyIndex++) {
            if (!inputKeys[keyIndex].equalsIgnoreCase(classKey)) {
                int indexOfClass = 0;
                while (classification != possibleClasses[indexOfClass]) {
                    // check classification == i for all String i : class
                    indexOfClass++;
                }
                Double[] thisFrequency;
                double see_value = values.get(inputKeys[keyIndex])[i];

                Double[] thisRangeSet;
                if (i == 0) {
                    if (isQuan[keyIndex]) {
                        thisRangeSet = getQuanRangesDouble(inputKeys[keyIndex]);
                    } else {
                        thisRangeSet = getQualOptions(inputKeys[keyIndex]);
                        // qualitative cats: 0.0 to 1/numOptions, 1/numOptions to 2/numOptions, ... n-1/numOptions to n/numOptions
                    }
                    rangeCategories.put(inputKeys[keyIndex], thisRangeSet);
                } else {
                    thisRangeSet = rangeCategories.get(inputKeys[keyIndex]);
                }

                if (rangeFrequencies.get(inputKeys[keyIndex])[indexOfClass] == null) {
                    thisFrequency = new Double[thisRangeSet.length]; // thisFrequency
                    Arrays.fill(thisFrequency, 0.0);
                } else {
                    thisFrequency = rangeFrequencies.get(inputKeys[keyIndex])[indexOfClass];
                }

                if (isQuan[keyIndex]) {
                    for (int j = 0; j < thisRangeSet.length - 1; j++) {
                        if (see_value > thisRangeSet[thisRangeSet.length - 1]) {
                            thisFrequency[thisRangeSet.length - 1] = (1.0 + thisFrequency[thisRangeSet.length - 1] * (thisRangeSet.length - 1)) / (thisRangeSet.length);
                            break;
                        }
                        if (see_value > thisRangeSet[j] && see_value < thisRangeSet[j + 1]) {
                            thisFrequency[j] = (1.0 + thisFrequency[j] * (thisRangeSet.length - 1)) / (thisRangeSet.length);
                            break; // reweight by number of each attribute, not total instances
                        }
                    }
                } else {
                    for (int j = 0; j < thisRangeSet.length; j++) {
                        if (see_value == thisRangeSet[j]) {
                            thisFrequency[j] = (1.0 + thisFrequency[j] * (thisRangeSet.length - 1)) / (thisRangeSet.length);
                            break;
                        }
                    }
                }

                Double[][] rangeFreqAdjuster = rangeFrequencies.get(inputKeys[keyIndex]);
                rangeFreqAdjuster[indexOfClass] = thisFrequency;
                rangeFrequencies.put(inputKeys[keyIndex], rangeFreqAdjuster);
            }
        }
    }

    /**
     * Gets ranges for quantitative double data
     * @param key
     * @return
     */
    private Double[] getQuanRangesDouble(String key) {  // handle real data types
        Pair<Double, Double> dimensions;
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        Double[] vals = values.get(key);
        for (Double i : vals) {
            if (min > i)
                min = i;
            if (max < i)
                max = i;
        }
        dimensions = new Pair(max - min, 1 + (Math.log(vals.length) / Math.log(2)));
        double classWidth = dimensions.val1 / dimensions.val2;
        Double[] rangeCategorySet = new Double[dimensions.val2.intValue() + 1];
        for (int k = 0; k < rangeCategorySet.length; k++) {
            // 1st cat to nth cat
            // if (within this rangeClass) then range value = (1/dimensions.val2) * which class
            rangeCategorySet[k] = min + (classWidth * (k)); // lower bound == rangeCats[n-1]; upper bound == rangeCats[n]
        }
        return rangeCategorySet;
    }

    private Double[] getQualOptions(String key) { // handle enumerations, nominal classes, boolean data types
        ArrayList<Double> categorySet = new ArrayList<>();
        Double[] vals = values.get(key);
        for (int i = 0; i < vals.length; i++) {
            if (categorySet.size() == 0) { // default, add 1 to array
                int index = 0;
                while (Double.isNaN(vals[index])) {
                    index++;
                }
                categorySet.add(vals[index]);
                i = index;
            }
            for (int j = 0; j < categorySet.size(); j++) {
                if (vals[i].doubleValue() == categorySet.get(j).doubleValue() || Double.isNaN(vals[i].doubleValue())) { // look for repeats
                    break;
                }
                if (j >= categorySet.size() - 1) { // if no repeats
                    categorySet.add(vals[i]);
                }
            }
        }
        Double[] out = new Double[categorySet.size()];
        categorySet.toArray(out);
        Arrays.sort(out);
        return out;
    }

    // --------------------public----------------------

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
    
    public Map<String, List<Object>> getFullModel() {
    	// where list contains either double[] or double[][] for each key
    	Map<String, List<Object>> out = new HashMap<>();
    	for (String str : inputKeys) {
    		List<Object> temp = new ArrayList<>();
    		temp.add(rangeCategories.get(str));
    		temp.add(rangeFrequencies.get(str));
    		out.put(str, temp);
    	}
    	return out;
    }

    /**
     * Returns double value of highest likelihood class
     *
     * @param responses
     * @return
     */
    public Double predictClass(Double[] responses) {
        // need keyChain because keys will not be in the same order as this.inputKeys
        int numClasses = possibleClasses.length;
        Double[] classProbs = new Double[numClasses];
        Arrays.fill(classProbs, 1.0 / numClasses); // fill all indices with equal probability
        for (int i = 0; i < responses.length; i++) { // iterates through responses in this instance
            for (int j = 0; j < classProbs.length; j++) { // iterator for the probabilities of each class
                if (inputKeys[i].equalsIgnoreCase(classKey))
                    break;
                Double[] model_j = rangeFrequencies.get(inputKeys[i])[j];

                if (isQuan[i]) {
                    for (int k = 0; k < rangeFrequencies.get(inputKeys[i])[j].length - 1; k++) { // iterates through each frequency and finds a match
                        double thisAttr = rangeCategories.get(inputKeys[i])[k];
                        double nextAttr = rangeCategories.get(inputKeys[i])[k + 1];
                        if (responses[i] >= thisAttr && responses[i] < nextAttr) { // check range categories
                            classProbs[j] = (classProbs[j] + model_j[k] * (values.get(inputKeys[i]).length - 1)) / values.get(inputKeys[i]).length;
                            break;
                        }
                    }
                } else {
                    for (int k = 0; k < rangeFrequencies.get(inputKeys[i])[j].length; k++) {
                        double thisAttr = rangeCategories.get(inputKeys[i])[k];
                        if (responses[i].doubleValue() == thisAttr) { // check attribute values
                            classProbs[j] = (classProbs[j] + model_j[k] * (rangeCategories.get(inputKeys[i]).length - 1)) / (rangeCategories.get(inputKeys[i]).length);
                            break;
                        }
                    }
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
        return possibleClasses[maxIndex]; // returns the class with the highest likelihood score
    }

    /**
     * Basic Naive Bayes approach works on true/false datasets
     *
     * @return
     */
    public BasicNaiveBayes train() {
        // width first (instance iterator; covers all keywords in a model instance)
        for (int i = 0; i < values.get(classKey).length; i++) {
            numInstances++;
            // depth second (keyword iterator; steps through each keyword in an instance
            Double[] thisInstance = new Double[inputKeys.length - 1];
            for (int j0 = 0; j0 < inputKeys.length; j0++) {
                if (!inputKeys[j0].equalsIgnoreCase(classKey)) { // skip class attr
                    thisInstance[j0] = values.get(inputKeys[j0])[i];
                }
            }
            Double classification = values.get(inputKeys[inputKeys.length - 1])[i];
            reweight(classification, i);
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
     * Runs testfile alongside trained model and provides output
     *
     * @param bnb
     * @param file
     * @return
     * @throws FileNotFoundException
     * @throws IOException
     */
    public static String predictClassFile(BasicNaiveBayes bnb, File file) {
        String out = "";
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

            Double[] testDataPtsArray = new Double[attr.size()];

            double countCorrect = 0.0, countIncorrect = 0.0;
            for (int outer = 0; outer < data.size(); outer++) {
                double[] temp = data.get(outer).toDoubleArray();

                for (int i = 0; i < temp.length - 1; i++) // deep copy double[] to Double[]
                    testDataPtsArray[i] = temp[i];

                Double finalProb = bnb.predictClass(testDataPtsArray);
                Double classValue = data.get(outer).classValue();
                if (classValue.doubleValue() == finalProb.doubleValue()) {
                    countCorrect++;
                } else {
                    countIncorrect++;
                }
            }
            out += "Count correct: " + countCorrect + "\nCount incorrect: " + countIncorrect
                    + "\nPercent Accuracy for model: " + String.format("%.1f", (countCorrect / (countCorrect + countIncorrect)) * 100.0) + "%";
        } catch (IOException e) {
            e.printStackTrace();
        }
        return out;
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

        BasicNaiveBayes bnb = new BasicNaiveBayes(data.attribute(data.classIndex()).name());
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

        attrArray[attrArray.length - 1] = bnb.classKey;
        bnb.buildMap(attrArray, dataPtsArray);

        Double[] temp = bnb.values.get(bnb.classKey);
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

        Enumeration<Attribute> enumAttr = data.enumerateAttributes();
        if (bnb.isQuan == null) {
            bnb.isQuan = new Boolean[bnb.inputKeys.length];
            Arrays.fill(bnb.isQuan, false);
        }
        int indexOfAttr = 0;
        while (enumAttr.hasMoreElements()) {
            if (enumAttr.nextElement().isNumeric()){
                bnb.isQuan[indexOfAttr] = true;
            }
            indexOfAttr++;
        }

        bnb.possibleClasses = new Double[classes.size()];
        classes.toArray(bnb.possibleClasses);
        bnb.rangeFrequencies = new HashMap<>();
        bnb.rangeCategories = new HashMap<>();
        for (String key : bnb.inputKeys) {
            if (!key.equalsIgnoreCase(bnb.classKey)) {
                bnb.rangeCategories.put(key, new Double[bnb.possibleClasses.length]);    // [classes][quantitative categories]
                bnb.rangeFrequencies.put(key, new Double[bnb.possibleClasses.length][]); // [classes][quantitative categories][frequency of each category]
            }
        }
        return bnb;
    }
}
