import java.util.Arrays;
import java.util.Random;

public class MLP
{
    private int numInputs;
    private int numHiddenUnits;
    private int numOutputs;

    private double[] w1, w2;
    private double[] dW1, dW2;

    private double[] z1, z2;

    private double[] hidden;

    private double[] outputs;



    public MLP(int ni, int nh, int no)
    {

        numInputs = ni;
        numHiddenUnits = nh;
        numOutputs = no;

        w1 = new double [numInputs];
        w2 = new double [numHiddenUnits];

        z1 = new double[numHiddenUnits];
        z2 = new double[numHiddenUnits];

        hidden = new double[numHiddenUnits];
        outputs = new double[numOutputs];
        //initialise various arrays

        randomise(w1);
        randomise(w2);
        Arrays.fill(outputs, 0);
        //fill weight arrays with small random initial weights
    }

    private void randomise(double[] input)//method to fill 2d array with small random values
    {

        for(double weight : input)
        {
            weight = generateInitialWeight();
            System.out.println(weight);
        }

    }

    private double generateInitialWeight()//method to generate small initial weights
    {
        Random rand = new Random();

        int numerator = rand.nextInt(10);

        double initialWeight = numerator/1000.0;

        if(initialWeight == 0)
            initialWeight += 0.05;

        return initialWeight;
    }

    public void forwardPass(double[] input)
    {
        if(input.length != numInputs)
            System.out.println("Invalid input");
        //check to make sure dimensions are okay

        hidden = multiply(input,w1);

        for(double h : hidden)
            h = sigmoid(h);
        //multiply inputs by weights in lower layer then apply activation function

        outputs = multiply(hidden, w2);

        for(double o : outputs)
            o = sigmoid(o);
        //multiply hidden values by weights in upper layer and apply activation function
    }

    public void backProp(double input)
    {

    }

    private double sigmoid(double x)
    {
        return (1.0/1.0 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x)
    {
        return sigmoid(x)*(1.0-sigmoid(x));
    }

    private double[] multiply(double[] a1, double[] a2)
    {
        double[] output = new double[a1.length];

        for(double  value: output)
        {
            value = 0;

            for (int i = 0; i < a1.length; i++)
                value += a1[i] * a2[i];
        }

        return output;
    }

}
