package com.naivebayes;

import org.apache.commons.io.FileUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.web.multipart.commons.CommonsMultipartResolver;

import java.io.File;
import java.io.IOException;

@SpringBootApplication
/**
 * @SpringBootApplication tells spring that this is the
 * starting point for our spring application
 *
 */
public class NBApi {

	@Value("${data.test_data}")
	private static String fileDir;

	public static void main(String[] args) {
		new SpringApplication(NBApi.class).run(args);

		// TODO figure out best way to implement shutdown hook to clear cache
		Runtime.getRuntime().addShutdownHook(new Thread()
		{
			public void run()
			{
				try {
					FileUtils.deleteDirectory(new File(fileDir + "/custom/"));
				} catch (IOException e) {
					System.err.println("Tempfiles directory already deleted; shutdown hook terminated");
				}
			}
		});
	}

	@Bean
	public CommonsMultipartResolver multipartResolver(){
		CommonsMultipartResolver commonsMultipartResolver = new CommonsMultipartResolver();
		commonsMultipartResolver.setDefaultEncoding("UTF-8");
		commonsMultipartResolver.setMaxUploadSize(50000000);
		return commonsMultipartResolver;
	}
	
}

