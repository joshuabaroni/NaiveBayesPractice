package main.java.controllers;

import java.util.Map;

import main.java.services.NBService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

//@CrossOrigin(origins="http://localhost:3000")
@CrossOrigin(origins="https://joshuabaroni.github.io")
@RestController
public class NBController {

	@Autowired
	private NBService nbService;

	@RequestMapping(value="/list_available_files", method= RequestMethod.GET)
	public Map<String, String> listAvailableFiles() {
		return nbService.listAvailableFiles();
	}

	@RequestMapping(value="/get_model_ranges", method=RequestMethod.GET)
	public Map<String, Object> getModelRanges() {
		return nbService.getModelRanges();
	}

	@RequestMapping(value="/get_model_frequencies", method=RequestMethod.GET)
	public Map<String, Object> getModelFrequencies() {
		return nbService.getModelFrequencies();
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
		return nbService.getModelAccuracy();
	}

	@RequestMapping(value="/set_file", method=RequestMethod.POST)
	public ResponseEntity<String> fileSet(@RequestParam(defaultValue="mushrooms") String fileKey) {
		return nbService.fileSet(fileKey);
	}

	}
