package com.naivebayes.exceptions;

import org.springframework.beans.factory.annotation.Value;

import java.io.IOException;

public class UnsupportedFiletypeException extends IOException {

    @Value("${data.supported_filetypes}")
    private String filetypes;

    @Override
    public String getMessage() {
        return "This filetype is not supported. Supported filetypes are: " + filetypes;
    }
}
