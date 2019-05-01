public class Main {

    public static void main(String[] args)
    {
       MLP nn = new MLP(2,2,1);

       double[][] input =  {{0,0}, {0,1}, {1,0}, {1,1}};
       double[] output = {0, 1, 1, 0};

       int maxEpochs = 500;
       double learningRate = 0.1;

        for (int e=0; e<maxEpochs; e++)
        {
            double totalError = 0;

            for (int i=0;i<input.length;i++)
            {
                nn.forwardPass(input[i]);
                double thisError = nn.backProp(output[i]);
                totalError += thisError;
                System.out.println("Expected: " + output[i]);
            }

            nn.updateWeights(learningRate);

            System.out.println("Total error at epoch " + e + ": " + totalError + "\n");
        }

//        MLP n2 = new MLP(4, 5, 1);
//
//        ArrayList<Vector> list = new ArrayList<>(200);
//
//        for(int i=0;i<200;i++)
//           list.add(new Vector());
//
//        learningRate = 0.1;
//
//        for(int e=0;e<maxEpochs;e++)
//        {
//            double totalError = 0;
//
//            for (int i=0;i<150;i++)
//            {
//                Vector v = list.get(i);
//                System.out.println(v.toString());
//                n2.forwardPass(v.getValues());
//                double thisError = n2.backProp(v.getOutput());
//                totalError += thisError;
//                System.out.println("Expected: " + v.getOutput());
//            }
//
//            n2.updateWeights(learningRate);
//
//            learningRate += 0.1;
//
//            System.out.println("Total error at epoch " + e + ": " + totalError + "\n");
//
//        }
    }
}
