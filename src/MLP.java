import java.util.Arrays;
import java.util.Random;

public class MLP
{
    private int numInputs;
    private int numHiddenUnits;
    private int numOutputs;

    private double[][] w1, w2;
    private double[][] dW1, dW2;

    private double[] z1, z2;

    private double[] inputValues;
    private double[] hidden;
    private double[] outputs;



    public MLP(int ni, int nh, int no)
    {

        numInputs = ni;
        numHiddenUnits = nh;
        numOutputs = no;

        w1 = new double [numInputs][numHiddenUnits];
        w2 = new double [numHiddenUnits][numOutputs];

        dW1 = new double [numInputs][numHiddenUnits];
        dW2 = new double [numHiddenUnits][numOutputs];

        z1 = new double[numHiddenUnits];
        z2 = new double[numOutputs];

        hidden = new double[numHiddenUnits];
        outputs = new double[numOutputs];
        //initialise various arrays

        w1 = randomise(w1);
        w2 = randomise(w2);
        Arrays.fill(z1,0);
        Arrays.fill(z2, 0);
        Arrays.fill(hidden,0);
        Arrays.fill(outputs, 0);
        //fill weight arrays with small random initial weights

        dW1 = fill2Dzeroes(dW1);
        dW2 = fill2Dzeroes(dW2);
    }

    private double[][] randomise(double[][] input)//method to fill 2d array with small random values
    {
        Random rand = new Random();

        double[][] result = new double[input.length][input[0].length];//output array same dimension as input array

        for(int i=0;i<input.length;i++)
        {
            for(int j=0;j<input[0].length;j++)
                result[i][j] = rand.nextDouble();//fill array with random doubles in range [0,0.5)
        }

        return result;
    }

    private double[][] fill2Dzeroes(double[][] input)
    {
        double[][] result = new double[input.length][input[0].length];

        for(double[] row : result)
            Arrays.fill(row,0);

        return result;
    }

    public void forwardPass(double[] input)//pass an input vector and calculate and store result in outputs array
    {
        inputValues = input;//keep track of input data - need to calculate deltas
        calculateHidden(input);
        calculateOutput();
    }

    private void calculateHidden(double[] input)//calculate hidden values for forward pass
    {
        //using matrix multiplication to calculate values in hidden units
        for(int i=0;i<numHiddenUnits;i++)
        {
            for(int j=0;j<numInputs;j++)
            {
                z1[i] = (input[j] * w1[j][i]);//activation for hidden units
                hidden[i] = sigmoid(z1[i]);//apply activation to get hidden values
            }
        }
    }

    private void calculateOutput()//calculate output of a forward pass
    {
        for(int i=0;i<numOutputs;i++)
        {
            for(int j=0;j<numInputs;j++)
            {
                z2[i] =z1[j] * w2[j][i];//activation for output
                outputs[i] = sigmoid(z2[i]);//apply activation to get output values
            }
        }
    }

    public double backProp(double target)
    {
        double error = calculateError(target);
        double deltaOutput = 0;

        for(int i=0;i<numOutputs;i++)
            deltaOutput += error * sigmoidDerivative(z2[i]);
        //need to compute deltas for upper layer

        for(int i=0;i<numHiddenUnits;i++)
        {
            for(int j=0;j<numOutputs;j++)
            {
                dW2[i][j] += deltaOutput * hidden[j];
            }
            //multiply deltas by hidden values
            //this gives dW2 values
        }

        for(int i=0;i<numInputs;i++)
        {
            for(int j=0;j<numHiddenUnits;j++)
            {
                double hiddenError = deltaOutput * sigmoidDerivative(hidden[j]);
                double deltaHidden = hiddenError * w2[j][0];
                dW1[i][j] +=  deltaHidden * inputValues[i];
            }
        }
        //compute deltas for lower layer
        //multiply deltas by input values

        return error;
    }

    public void updateWeights(double learningRate)
    {
        for(int i=0;i<numInputs;i++)
        {
            for(int j=0;j<numHiddenUnits;j++)
                w1[i][j] -=dW1[i][j]*learningRate;
        }

        for(int i=0;i<numHiddenUnits;i++)
        {
            for(int j=0;j<numOutputs;j++)
                w2[i][j] -= dW2[i][j]*learningRate;
        }

        dW1 = fill2Dzeroes(dW1);
        dW2 = fill2Dzeroes(dW2);
        //reinitialise the arrays to set all the values to 0
    }

    private double sigmoid(double x)//activation function
    {
        return (1.0/1.0 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x)//activation function derivative for backpropagation
    {
        return sigmoid(x)*(1.0-sigmoid(x));
    }



    private double calculateError(double target)//calculate error from target value
    {
        double error = 0;

        for(double y : outputs)
            error+= 0.5*(Math.pow((target-y),2));

        return error;//mean squared error
    }


    public double[] getOutput()//get output value(s)
    {
        return outputs;
    }
}
