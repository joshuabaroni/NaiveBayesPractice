package models;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;

import constants.Utils;

/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    IncrementalClassifier.java
 *    Copyright (C) 2009 University of Waikato, Hamilton, New Zealand
 *
 */
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 * This example trains NaiveBayes incrementally on data obtained from the
 * ArffLoader.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision$
 */
public class IncrementalClassifier {

    /**
     * Expects an ARFF file as first argument (class attribute is assumed to be the
     * last attribute).
     *
     * @param args the commandline arguments
     * @throws Exception if something goes wrong
     */
    public static void main(String[] args) throws Exception {

        // model
        IncrementalClassifier vote = new IncrementalClassifier();
        NaiveBayes naiveBayes = new NaiveBayes();
        Instances train = vote.train();
        Instances test = vote.setupTestInstance(train);
        naiveBayes.buildClassifier(train);
        System.out.println(naiveBayes);
        System.out.println("Test instance: " + test.instance(0) + "\n");

        test = vote.naiveBayes(naiveBayes, test);

        System.out.println("\nPredicted decision: " + test.instance(0).stringValue(test.numAttributes() - 1));
//        double[] instanceValue1 = new double[train.numAttributes()];
//        for (int i = 0; i < instanceValue1.length; i++) {
//            if ()
//        }
    }

    public Instances train() throws Exception {
        ConverterUtils.DataSource source1 = new ConverterUtils.DataSource(
                Utils.FILESPACE + "S_N_axis_compare_train.arff");
        Instances train = source1.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (train.classIndex() == -1)
            train.setClassIndex(train.numAttributes() - 1);
        return train;
    }

    public Instances setupTestInstance(Instances train) throws Exception {
        ConverterUtils.DataSource source2 = new ConverterUtils.DataSource(
                Utils.FILESPACE + "S_N_axis_compare_test.arff");
        Instances test = source2.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (test.classIndex() == -1)
            test.setClassIndex(train.numAttributes() - 1);
        return test;
    }

    public Instances naiveBayes(NaiveBayes naiveBayes, Instances test) throws Exception {
        double[] labels = naiveBayes.distributionForInstance(test.instance(0));
        Attribute attr = test.attribute(test.numAttributes() - 1);
        for (int i = 0; i < labels.length; i++) {
            System.out.println(
                    attr.toString() + " " + (i) + " likelihood: " + Math.round(labels[i] * 10000.0) / 100.0 + "%");
        }
        double label = naiveBayes.classifyInstance(test.instance(0));
        test.instance(0).setClassValue(label);
        return test;
    }
}