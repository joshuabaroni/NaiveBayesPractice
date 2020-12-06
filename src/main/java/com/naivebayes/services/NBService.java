package com.naivebayes.services;

import com.naivebayes.exceptions.InvalidDataValueException;
import com.naivebayes.models.BasicNaiveBayes;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

@Service
public class NBService {

    @Value("${data.test_data}")
    private String fileDir;

    private BasicNaiveBayes bnb = null;
    private File testDataFile;
    private File trainDataFile;
    private String customPath;
    private boolean useCustom = false;

    /**
     * setFilePath[0] = train data file
     * setFilePath[1] = test data file
     */
    private String[] setFilePath = new String[2];

    public Map<String, String> listAvailableFiles() {
        return buildFileList();
    }

//====================================== GET Methods =============================================

    public Map<String, Object> getModelRanges() {
        try {
            bnb = BasicNaiveBayes.naiveBayesBuilder(testDataFile).train();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return bnb.getModelRanges();
    }

    public Map<String, Object> getModelFrequencies() {
        try {
            bnb = BasicNaiveBayes.naiveBayesBuilder(testDataFile).train();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return bnb.getModelFrequencies();
    }

    public String getModelAccuracy() {
        try {
            bnb = BasicNaiveBayes.naiveBayesBuilder(testDataFile).train();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return /*("Done training model: " + bnb.toString() + "\n")
        		+ */BasicNaiveBayes.predictClassFile(bnb, trainDataFile);
    }

//====================================== POST Methods ===========================================

    public String fileSet(String fileKey) throws InvalidDataValueException {
        String out;
        useCustom = false;
        switch(fileKey) {
            case "hepatitis":
                setFilePath[0] = fileDir + "/hepatitis/hepatitis.arff"; // train
                setFilePath[1] = fileDir + "/hepatitis/hepatitis.arff"; // test
                out = "Train data set to: " + setFilePath[0] + "\nTest file set to: " + setFilePath[1];
                break;
            case "iris":
                setFilePath[0] = fileDir + "/iris/iris.arff"; // train
                setFilePath[1] = fileDir + "/iris/iris_test.arff"; // test
                out = "Train data set to: " + setFilePath[0]
                        + "\nTest file set to: " + setFilePath[1];
                break;
            case "labor_negotiation":
                setFilePath[0] = fileDir + "/labor_negotiation/labor_negotiations.arff"; // train
                setFilePath[1] = fileDir + "/labor_negotiation/labor_negotiations.arff"; // test
                out = "Train data set to: " + setFilePath[0]
                        + "\nTest file set to: " + setFilePath[1];
                break;
            case "mushrooms":
                setFilePath[0] = fileDir + "/mushrooms/mushroom.arff"; // train
                setFilePath[1] = fileDir + "/mushrooms/mushroom.arff"; // test
                out = "Train data set to: " + setFilePath[0]
                        + "\nTest file set to: " + setFilePath[1];
                break;
            case "soybeans": // TODO
                setFilePath[0] = fileDir + "/soybeans/soybeans.arff"; // train
                setFilePath[1] = fileDir + "/soybeans/soybeans.arff"; // test
                out = "Train data set to: " + setFilePath[0]
                        + "\nTest file set to: " + setFilePath[1];
                break;
            case "sponge":
                setFilePath[0] = fileDir + "/sponge/sponge.arff"; // train
                setFilePath[1] = fileDir + "/sponge/sponge.arff"; // test
                out = "Train data set to: " + setFilePath[0]
                        + "\nTest file set to: " + setFilePath[1];
                break;
            case "sick":
                setFilePath[0] = fileDir + "/sick/sick.arff"; // train
                setFilePath[1] = fileDir + "/sick/sick.arff"; // test
                out = "Train data set to: " + setFilePath[0]
                        + "\nTest file set to: " + setFilePath[1];
                break;
            case "weight_height":
                setFilePath[0] = fileDir + "/weight_height/weight_height.arff"; // train
                setFilePath[1] = fileDir + "/weight_height/weight_height.arff"; // test
                out = "Train data set to: " + setFilePath[0]
                        + "\nTest file set to: " + setFilePath[1];
                break;
            case "weather":
                setFilePath[0] = fileDir + "/weather/weather.arff"; // train
                setFilePath[1] = fileDir + "/weather/weather.arff"; // test
                out = "Train data set to: " + setFilePath[0]
                        + "\nTest file set to: " + setFilePath[1];
                break;
            case "voting":
                setFilePath[0] = fileDir + "/voting/voting.arff"; // train
                setFilePath[1] = fileDir + "/voting/voting.arff"; // test
                out = "Train data set to: " + setFilePath[0]
                        + "\nTest file set to: " + setFilePath[1];
                break;
            case "custom":
            case "":
                out = "Train data set to custom traindata file upload";
                useCustom = true;
                break;
            default:
                throw new InvalidDataValueException();
        }
        if (useCustom) {
            testDataFile = new File(customPath);
            trainDataFile = new File(customPath);
        }
        else if (setFilePath[0].length() == 0 || setFilePath[0] == null) {
            throw new InvalidDataValueException();
        }
        else {
            testDataFile = new File(setFilePath[0]);
            trainDataFile = new File(setFilePath[1]);
        }
        return out;
    }

    public String fileUpload(MultipartFile dataset) throws IOException {
        File dir = new File(fileDir + "/custom/");
        if (!dir.exists())
            dir.mkdir();
        Path filepath = Paths.get(dir.getAbsolutePath(), dataset.getOriginalFilename());

        try (OutputStream os = Files.newOutputStream(filepath)) {
            os.write(dataset.getBytes());
        }
        // TODO figure out which approach is optimal
//        dataset.transferTo(filepath);

        // sets the path of the uploaded file to the customPath global
        customPath = filepath.toString();
        return "Successfully uploaded new training data.";
    }

//----------------------private static builders----------------------

    /**
     * TODO for now hardcoded
     * @return List<Map> containing fileKeys of approved arff files
     * and a short description about each one
     */
    private static Map<String, String> buildFileList() {
        HashMap<String, String> map = new HashMap<>();
        map.put("hepatitis",
                "A collection of symptom data to classify the patient as having hepatitis||!hepatitis");
        map.put("iris",
                "The classic ML iris dataset");
        map.put("mushrooms",
                "Fungal Species classification based on featuout data");
        map.put("labor_negotiation",
                "Final settlements in labor negotitions in Canadian industry, Nov 1988");
        map.put("soybeans",
                "Soybean Species classification based on featuout data");
        map.put("sponge",
                "Classification of marine sponges based on featuout data");
        map.put("sick",
                "A collection of symptom data to classify the patient as sick||!sick");
        map.put("weight_height",
                "Classifies a person's gender based on their height/weight");
        map.put("weather",
                "Predicts whether or not one should go outside based on the weather forecast on a given day");
        map.put("voting",
                "Classifies whether a subject is democrat or republican based on their answers to a political poll");

        return map;
    }
}
