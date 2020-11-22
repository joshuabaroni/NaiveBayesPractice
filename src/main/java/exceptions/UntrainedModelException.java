package main.java.exceptions;

public class UntrainedModelException extends NullPointerException {
    @Override
    public String getMessage() {
        return "UntrainedModelException: MODEL NOT TRAINED. " + this.getStackTrace();
    }
}
