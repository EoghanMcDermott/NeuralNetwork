import java.util.concurrent.ThreadLocalRandom;

public class Vector
{
    private static final double min = -1;
    private static final double max = 1;
    //range of possible values

    private double[] values;

    public Vector()
    {
        values = new double[4];//4 inputs for this assignment
        initialiseValues();
    }

    private void initialiseValues()//initialise values array
    {
        for(int i=0;i<values.length;i++)
        {
            values[i] = ThreadLocalRandom.current().nextDouble(min, max);//generate random value in interval [-1,1]
        }
    }

    public double[] getValues()
    {
        return values;
    }

    public double[] getOutput()//returns as an array to comply with how networks are built
    {
         double[] result = {Math.sin(values[0] - values[1] + values[2] - values[3])};
         return result;
    }

    public String toString()//toString method for debugging
    {
        String str = "Input: ";

        for(double v : values)
            str += v + ", ";

        str += "Output: " + getOutput()[0];

        return str;
    }
}
