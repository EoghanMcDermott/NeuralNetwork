import java.io.FileWriter;
import java.util.ArrayList;

public class Main
{

    public static void main(String[] args)
    {
        try
        {
            FileWriter fw = new FileWriter("output_xor.txt");

            MLP nn = new MLP(2, 2, 1);

            double[][] input = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
            double[][] output = {{0}, {1}, {1}, {0}};

            int maxEpochs = 10000;
            double learningRate = 0.1;

            for(int e=0;e<maxEpochs;e++)
            {
                double totalError = 0;

                for(int i=0;i<input.length;i++)//train over all examples
                {
                    if (i % 2 == 0)
                        nn.updateWeights(learningRate);//update more than once every epoch

                    nn.forwardPass(input[i]);
                    double thisError = nn.backProp(output[i]);
                    totalError += thisError;
                    fw.write("Expected: " + output[i][0] + " Predicted: " + nn.getOutput()[0] + "\n");
                }

                fw.write("Total error at epoch " + e + ": " + totalError + "\n\n");
            }

            fw.close();
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }

        try
        {
            FileWriter fw2 = new FileWriter("output_sin_training.txt");

            MLP n2 = new MLP(4, 5, 1);

            ArrayList<Vector> list = new ArrayList<>();

            for (int i=0;i<200;i++)
                list.add(new Vector());//creating 200 vectors and storing in list

            double maxEpochs = 10000;
            double learningRate = 0.1;

            for (int e=0;e<maxEpochs;e++)
            {
                double totalError = 0;

                for (int i=0;i<150;i++)//train on 1st 150 examples
                {
                    Vector v = list.get(i);//get a single vector from list

                    n2.forwardPass(v.getValues());

                    //fw2.write("Expected: " + v.getOutput()[0] + " Predicted: " + n2.getOutput()[0] + "\n");

                    double thisError = n2.backProp(v.getOutput());

                    totalError += thisError;
                }

                n2.updateWeights(learningRate);//update every epoch

                fw2.write("Total error at epoch " + e + ": " + totalError + "\n\n");
            }

            fw2.close();

            FileWriter fw3 = new FileWriter("output_sin_test.txt");

            for (int i=150;i<200;i++)//test on last 50 examples
            {
                Vector v = list.get(i);//get a single vector from list

                n2.forwardPass(v.getValues());

                fw3.write("Expected: " + v.getOutput()[0] + " Predicted: " + n2.getOutput()[0] + "\n");
            }

            fw3.close();
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }
}
