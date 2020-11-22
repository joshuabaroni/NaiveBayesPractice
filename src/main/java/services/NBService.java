package main.java.services;

import main.java.models.BasicNaiveBayes;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

@Service
public class NBService {

    @Value("${data.test_data}")
    private String filespace;

    private BasicNaiveBayes bnb = null;
    private File testDataFile;

    /**
     * setFilePath[0] = train data file
     * setFilePath[1] = test data file
     */
    private String[] setFilePath = new String[2];

    public Map<String, String> listAvailableFiles() {
        return buildFileList();
    }

    // Eventually will allow multipart file upload to Ai model
    public ResponseEntity<String> fileSet(String fileKey) {
        ResponseEntity<String> re = null;
        // TODO check map instead; improves scalability
        switch(fileKey) {
            case "hepatitis":
                setFilePath[0] = filespace + "/hepatitis/hepatitis.arff"; // train
                setFilePath[1] = filespace + "/hepatitis/hepatitis.arff"; // test
                re = new ResponseEntity<String>("Train data set to: " + setFilePath[0]
                        + "\nTest file set to: " + setFilePath[1], HttpStatus.ACCEPTED);
                break;
            case "iris":
                setFilePath[0] = filespace + "/iris/iris.arff"; // train
                setFilePath[1] = filespace + "/iris/iris_test.arff"; // test
                re = new ResponseEntity<String>("Train data set to: " + setFilePath[0]
                        + "\nTest file set to: " + setFilePath[1], HttpStatus.ACCEPTED);
                break;
            case "labor_negotiation":
                setFilePath[0] = filespace + "/labor_negotiation/labor_negotiations.arff"; // train
                setFilePath[1] = filespace + "/labor_negotiation/labor_negotiations.arff"; // test
                re = new ResponseEntity<String>("Train data set to: " + setFilePath[0]
                        + "\nTest file set to: " + setFilePath[1], HttpStatus.ACCEPTED);
                break;
            case "mushrooms":
                setFilePath[0] = filespace + "/mushrooms/mushroom.arff"; // train
                setFilePath[1] = filespace + "/mushrooms/mushroom.arff"; // test
                re = new ResponseEntity<String>("Train data set to: " + setFilePath[0]
                        + "\nTest file set to: " + setFilePath[1], HttpStatus.ACCEPTED);
                break;
            case "soybeans": // TODO
                setFilePath[0] = filespace + "/soybeans/soybeans.arff"; // train
                setFilePath[1] = filespace + "/soybeans/soybeans.arff"; // test
                re = new ResponseEntity<String>("Train data set to: " + setFilePath[0]
                        + "\nTest file set to: " + setFilePath[1], HttpStatus.ACCEPTED);
                break;
            case "sponge":
                setFilePath[0] = filespace + "/sponge/sponge.arff"; // train
                setFilePath[1] = filespace + "/sponge/sponge.arff"; // test
                re = new ResponseEntity<String>("Train data set to: " + setFilePath[0]
                        + "\nTest file set to: " + setFilePath[1], HttpStatus.ACCEPTED);
                break;
            case "sick":
                setFilePath[0] = filespace + "/sick/sick.arff"; // train
                setFilePath[1] = filespace + "/sick/sick.arff"; // test
                re = new ResponseEntity<String>("Train data set to: " + setFilePath[0]
                        + "\nTest file set to: " + setFilePath[1], HttpStatus.ACCEPTED);
                break;
            case "weight_height":
                setFilePath[0] = filespace + "/weight_height/weight_height.arff"; // train
                setFilePath[1] = filespace + "/weight_height/weight_height.arff"; // test
                re = new ResponseEntity<String>("Train data set to: " + setFilePath[0]
                        + "\nTest file set to: " + setFilePath[1], HttpStatus.ACCEPTED);
                break;
            case "weather":
                setFilePath[0] = filespace + "/weather/weather.arff"; // train
                setFilePath[1] = filespace + "/weather/weather.arff"; // test
                re = new ResponseEntity<String>("Train data set to: " + setFilePath[0]
                        + "\nTest file set to: " + setFilePath[1], HttpStatus.ACCEPTED);
                break;
            case "voting":
                setFilePath[0] = filespace + "/voting/voting.arff"; // train
                setFilePath[1] = filespace + "/voting/voting.arff"; // test
                re = new ResponseEntity<String>("Train data set to: " + setFilePath[0]
                        + "\nTest file set to: " + setFilePath[1], HttpStatus.ACCEPTED);
                break;
            default:
                re = new ResponseEntity<String>("To set a file: <host_url>/set_file?filename=<fileKey>", HttpStatus.BAD_REQUEST);
                break;
        }
        if (re.getStatusCode().value() < 300) {
            testDataFile = new File(setFilePath[0]);
        }
        return re;
    }

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

    public String getModelAccuracy() { // TODO return model
        if (bnb == null) {
            try {
                bnb = BasicNaiveBayes.naiveBayesBuilder(testDataFile).train();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        return /*("Done training model: " + bnb.toString() + "\n")
        		+ */BasicNaiveBayes.predictClassFile(bnb, new File(setFilePath[1]));
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
                "Fungal Species classification based on feature data");
        map.put("labor_negotiation",
                "Final settlements in labor negotitions in Canadian industry, Nov 1988");
        map.put("soybeans",
                "Soybean Species classification based on feature data");
        map.put("sponge",
                "Classification of marine sponges based on feature data");
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
