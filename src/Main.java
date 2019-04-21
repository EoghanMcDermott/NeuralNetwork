public class Main {

    public static void main(String[] args)
    {
       MLP nn = new MLP(2,2,1);

       double[] input = {0,1};
       nn.forwardPass(input);
       nn.backProp(1);
    }
}
