package main;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import models.BasicNaiveBayes;

@CrossOrigin(origins="http://localhost:3000, https://joshuabaroni.github.io")
@RestController
public class NBController {
	
	private BasicNaiveBayes bnb = null;
	private File testDataFile;
	
	/**
	 * setFilePath[0] = train data file
	 * setFilePath[1] = test data file
	 */
    private String[] setFilePath = new String[2];

    private Map<String, String> fileInfo = buildFileList();
    
    // TODO choose from list of included arff datasets
    @RequestMapping(value="/list_available_files", method=RequestMethod.GET)
    public Map<String, String> listAvailableFiles() {
    	return fileInfo;
    }
    
    // Eventually will allow multipart file upload to Ai model
	@RequestMapping(value="/set_file", method=RequestMethod.POST)
	public ResponseEntity<String> fileSet(@RequestParam(defaultValue="mushrooms") String fileKey) {
        ResponseEntity<String> re = null;
        // TODO check map instead; improves scalability
		switch(fileKey) {
        	case "hepatitis":
	        	setFilePath[0] = utilities.Utils.FILESPACE + "/hepatitis/hepatitis.arff"; // train
	            setFilePath[1] = utilities.Utils.FILESPACE + "/hepatitis/hepatitis.arff"; // test
	            re = new ResponseEntity<String>("Train data set to: " + setFilePath[0]
	            		+ "\nTest file set to: " + setFilePath[1], HttpStatus.ACCEPTED);
	            break;
        	case "iris":
	        	setFilePath[0] = utilities.Utils.FILESPACE + "/iris/iris.arff"; // train
	            setFilePath[1] = utilities.Utils.FILESPACE + "/iris/iris_test.arff"; // test
	            re = new ResponseEntity<String>("Train data set to: " + setFilePath[0]
	            		+ "\nTest file set to: " + setFilePath[1], HttpStatus.ACCEPTED);
	            break;
        	case "labor_negotiation":
	        	setFilePath[0] = utilities.Utils.FILESPACE + "/labor_negotiation/labor_negotiations.arff"; // train
	            setFilePath[1] = utilities.Utils.FILESPACE + "/labor_negotiation/labor_negotiations.arff"; // test
	            re = new ResponseEntity<String>("Train data set to: " + setFilePath[0]
	            		+ "\nTest file set to: " + setFilePath[1], HttpStatus.ACCEPTED);
	            break;
        	case "mushrooms":
	        	setFilePath[0] = utilities.Utils.FILESPACE + "/mushrooms/mushroom.arff"; // train
	            setFilePath[1] = utilities.Utils.FILESPACE + "/mushrooms/mushroom.arff"; // test
	            re = new ResponseEntity<String>("Train data set to: " + setFilePath[0]
	            		+ "\nTest file set to: " + setFilePath[1], HttpStatus.ACCEPTED);
	            break;
        	case "soybeans": // TODO
	        	setFilePath[0] = utilities.Utils.FILESPACE + "/soybeans/soybeans.arff"; // train
	            setFilePath[1] = utilities.Utils.FILESPACE + "/soybeans/soybeans.arff"; // test
	            re = new ResponseEntity<String>("Train data set to: " + setFilePath[0]
	            		+ "\nTest file set to: " + setFilePath[1], HttpStatus.ACCEPTED);
	            break;
        	case "sponge":
	        	setFilePath[0] = utilities.Utils.FILESPACE + "/sponge/sponge.arff"; // train
	            setFilePath[1] = utilities.Utils.FILESPACE + "/sponge/sponge.arff"; // test
	            re = new ResponseEntity<String>("Train data set to: " + setFilePath[0]
	            		+ "\nTest file set to: " + setFilePath[1], HttpStatus.ACCEPTED);
	            break;
        	case "sick":
	        	setFilePath[0] = utilities.Utils.FILESPACE + "/sick/sick.arff"; // train
	            setFilePath[1] = utilities.Utils.FILESPACE + "/sick/sick.arff"; // test
	            re = new ResponseEntity<String>("Train data set to: " + setFilePath[0]
	            		+ "\nTest file set to: " + setFilePath[1], HttpStatus.ACCEPTED);
	            break;
        	case "weight_height":
	        	setFilePath[0] = utilities.Utils.FILESPACE + "/weight_height/weight_height.arff"; // train
	            setFilePath[1] = utilities.Utils.FILESPACE + "/weight_height/weight_height.arff"; // test
	            re = new ResponseEntity<String>("Train data set to: " + setFilePath[0]
	            		+ "\nTest file set to: " + setFilePath[1], HttpStatus.ACCEPTED);
	            break;
        	case "weather":
	        	setFilePath[0] = utilities.Utils.FILESPACE + "/weather/weather.arff"; // train
	            setFilePath[1] = utilities.Utils.FILESPACE + "/weather/weather.arff"; // test
	            re = new ResponseEntity<String>("Train data set to: " + setFilePath[0]
	            		+ "\nTest file set to: " + setFilePath[1], HttpStatus.ACCEPTED);
	            break;
        	case "voting":
	        	setFilePath[0] = utilities.Utils.FILESPACE + "/voting/voting.arff"; // train
	            setFilePath[1] = utilities.Utils.FILESPACE + "/voting/voting.arff"; // test
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
	
	@RequestMapping(value="/get_model", method=RequestMethod.GET)
	public Map<String, List<Object>> getModel() {
        try {
            bnb = BasicNaiveBayes.naiveBayesBuilder(testDataFile).train();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return bnb.getFullModel();
	}
	
	@RequestMapping(value="/get_model_accuracy", method=RequestMethod.GET)
	/**
	 * Returns model containing categories, frequencies, and values
	 * 
	 * @return Map<ArrayList<Map>, String> where ArrayList<Map>[0] = rangeCategories,
	 * ArrayList<Map>[1] = rangeFrequencies, ArrayList<Map>[2] = rangeValues and
	 * String = model accuracy analysis
	 */
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
