/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetworkdemo;

import java.text.DecimalFormat;
import java.text.NumberFormat;

/**
 *
 * @author yigitpolat
 */
public class NeuralNetworkDemo {

    Neuron input1;
    Neuron input2;
    Neuron output;
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
         
        NeuralNetworkDemo nnd = new NeuralNetworkDemo();
        
        nnd.generateNetwork();
        nnd.testNetwork();
        nnd.trainNetwork();
        nnd.output.printWeightsRecursively(0, 0);
        nnd.testNetwork();
        
        NeuralNetwork nn = new NeuralNetwork( 0.01, 2, 1, 2, 2);
        double[] data1 = {0, 0};
        double[] data2 = {0, 1};
        double[] data3 = {1, 0};
        double[] data4 = {1, 1};
        double[] t1 = {0};
        double[] t2 = {1};
        double[] t3 = {1};
        double[] t4 = {0};
        nn.addData(data1, t1);
        nn.addData(data2, t2);
        nn.addData(data3, t3);
        nn.addData(data4, t4);
        nn.trainNetwork( 1000000);
        
        NumberFormat f = new DecimalFormat("#0.000000");
        System.out.print( f.format(nn.classify(data1)[0]) + ", " );
        //System.out.println( f.format(nn.classify(data1)[1]) );
        
        System.out.print( f.format(nn.classify(data2)[0]) + ", " );
        //System.out.println( f.format(nn.classify(data2)[1]) );
        
        System.out.print( f.format(nn.classify(data3)[0]) + ", " );
        //System.out.println( f.format(nn.classify(data3)[1]) );
        
        System.out.print( f.format(nn.classify(data4)[0]) + ", " );
        //System.out.println( f.format(nn.classify(data4)[1]) );
        
    }
    
    public void testNetwork()
    {
        NumberFormat formatter = new DecimalFormat("#0.000000");   
        for( int i = 0; i < 4; i++)
        {
            input1.setInput( i % 2);
            input2.setInput( (i >> 1) % 2);
            
            System.out.println( ">>" + formatter.format(output.output()));
        }
    }
    
    public void trainNetwork()
    {
        for( int i = 0; i < 1000000; i++)
        {
            int r = (int) (Math.random() * 100000) % 4;
            int a = r % 2;
            int b = (r >> 1) % 2;
            int c = a ^ b;
            
            double y = output.output();
            double Err = 0.5*(y - c)*(y - c);
            double dErr = (y - c); //Err = 0.5*(y - c)*(y - c)
          
            input1.setInput( a);
            input2.setInput( b);
            
            output.backPropagate( dErr );
            
            double y2 = output.output();
            double Err2 = 0.5*(y2 - c)*(y2 - c);
            double m = (Err - Err2);
            output.setMomentum( output.getMomentum()*0.9 + m);
        }
    }
    
    public void generateNetwork()
    {
        double lRate = 0.01;
        input1 = new Neuron( 1.0);
        input2 = new Neuron( 1.0);
        
        Neuron c = new Neuron( true, lRate, input1, input2);
        Neuron d = new Neuron( true, lRate, input1, input2);
        
        output = new Neuron( true, lRate, c, d);
        
        // working xor weights.
        /*
        c.setWeights( -3.719096, -3.741718, 1.011936);
        d.setWeights( -2.306784, -1.134736, 2.456571);
        output.setWeights( -3.439813, 2.410087, -0.717824);
        */
        
    }
    
}
