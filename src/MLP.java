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

    private double[] hidden;

    private double[] outputs;



    public MLP(int ni, int nh, int no)
    {

        numInputs = ni;
        numHiddenUnits = nh;
        numOutputs = no;

        w1 = new double [numInputs][numHiddenUnits];
        w2 = new double [numHiddenUnits][numOutputs];

        z1 = new double[numHiddenUnits];
        z2 = new double[numOutputs];

        hidden = new double[numHiddenUnits];
        outputs = new double[numOutputs];
        //initialise various arrays

        w1 = randomise(w1);
        w2 = randomise(w2);
        Arrays.fill(hidden,0);
        Arrays.fill(outputs, 0);
        //fill weight arrays with small random initial weights
    }

    private double[][] randomise(double[][] input)//method to fill 2d array with small random values
    {
        Random rand = new Random();

        double[][] result = new double[input.length][input[0].length];

        for(int i=0;i<input.length;i++)
        {
            for(int j=0;j<input[0].length;j++)
                result[i][j] = rand.nextDouble();
                //result[i][j] = generateInitialWeight();
        }

        return result;
    }

//    private double generateInitialWeight()//method to generate small initial weights
//    {
//        Random rand = new Random();
////
////        int numerator = rand.nextInt(10);
////
////        double initialWeight = numerator/1000.0;
////
////        if(initialWeight == 0)
////            initialWeight += 0.05;
////
////        return initialWeight;
//        return rand.nextDouble();
//    }

    public void forwardPass(double[] input)
    {
        printWeights();
        calculateHidden(input);
        calculateOutput();
    }

    public void backProp(double target)
    {
        //need to compute deltas for upper layer
        //multiply deltas by hidden values
        //this gives dW2 values

        double error = calculateError(target);

        double deltaOutput = error * sigmoidDerivative(outputs[0]);//know there's only one output - hack?

        for(int i=0;i<numHiddenUnits;i++)
        {
            for(int j=0;j<numOutputs;j++)
            {
                double dw2Update = deltaOutput * hidden[i];
                dW2[i][j] +=  dw2Update;
            }
        }

        //compute deltas for lower layer
        //multiply deltas by input values


    }

    public void updateWeights(double learningRate)
    {
        //W1 += learningRate*dW1;
       // W2 += learningRate*dW2;

        Arrays.fill(dW1,0);
        Arrays.fill(dW2,0);
    }

    private double sigmoid(double x)//activation function
    {
        return (1.0/1.0 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x)//activation function derivative for backpropagation
    {
        return sigmoid(x)*(1.0-sigmoid(x));
    }

    private void calculateHidden(double[] input)
    {
        //using matrix multiplication to calculate values in hidden units
        for(int i=0;i<numHiddenUnits;i++)
        {
            for(int j=0;j<numInputs;j++)
                hidden[i] += input[j] * w1[j][i];
        }

        printHidden();

        for(int i=0;i<numHiddenUnits;i++)
            z1[i] = sigmoid(hidden[i]);//applying activation function

        printZ1();
    }

    private void calculateOutput()
    {
        for(int i=0;i<numOutputs;i++)
        {
            for(int j=0;j<numInputs;j++)
                outputs[i] += z1[j] * w2[j][i];
        }

        for(int i=0;i<numOutputs;i++)
            z2[i] = sigmoid(outputs[i]);//applying activation function

        printOutput();
        printZ2();
    }

    private double calculateError(double target)
    {
        double error = 0;

        for(double y : outputs)
            error += 0.5*Math.pow(target-y,2); // 1/2 *(target-y)^2

        return error;
    }









    private void printHidden()
    {
        for(double h : hidden)
            System.out.println("Hidden: " + h);
    }

    private void printOutput()
    {
        for(double o : outputs)
            System.out.println("Output: " + o);
    }

    private void printZ1()
    {
        for(double z : z1)
            System.out.println("Z1: " + z);
    }

    private void printZ2()
    {
        for(double z : z2)
            System.out.println("Output after activation: " + z);
    }

    private void printWeights()
    {
        System.out.println("w1:");
        for(double[] row : w1)
            for(double w : row)
                System.out.println(w);

        System.out.println("w2:");
        for(double[] column : w2)
            for(double w : column)
                System.out.println(w);
    }

}
