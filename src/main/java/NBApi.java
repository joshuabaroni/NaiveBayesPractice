package main.java;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
/**
 * @SpringBootApplication tells spring that this is the
 * starting point for our spring application
 *
 */
public class NBApi {
	
	public static void main(String[] args) {
		new SpringApplication(NBApi.class).run(args);
	}
	
}

