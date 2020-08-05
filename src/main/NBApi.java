package main;

import java.util.Collections;

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
		SpringApplication app = new SpringApplication(NBApi.class);
		app.setDefaultProperties(Collections.singletonMap("server.port", (Object) 8080));
		app.run(args);
	}
	
}

