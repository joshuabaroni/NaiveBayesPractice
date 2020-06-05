package Exceptions;

import java.io.IOException;

public class UnsupportedFiletypeException extends IOException {
    @Override
    public String getMessage() {
        return "This filetype is not supported. Supported filetypes are: " + utilities.Utils.SUPPORTED_FILETYPES;
    }
}
