package org.sparkexample;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

import java.util.Arrays;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import scala.Tuple2;

public class WordCount {

  private static final PairFunction<String, String, Integer> WORDS_MAPPER =
      new PairFunction<String, String, Integer>() {
        @Override
        public Tuple2<String, Integer> call(String s) throws Exception {
          return new Tuple2<String, Integer>(s, 1);
        }
      };

  public static void main(String[] args) {
    if (args.length < 1) {
      System.err.println("Please provide the input file full path as argument");
      System.exit(0);
    }

    SparkConf conf = new SparkConf().setAppName("org.sparkexample.WordCount").setMaster("local");
    System.out.println("<-----------Application name and master set----------->");
    JavaSparkContext context = new JavaSparkContext(conf);
    System.out.println("<-----------Java Spark context set----------->");
    
    JavaRDD training = context.textFile(args[0]).cache().map(new Function<String,LabeledPoint> () {

        @Override
        public LabeledPoint call(String v1) throws Exception {
            //System.out.println("Index of , = "+v1.indexOf(","));
            String substring = v1.substring(0, v1.indexOf(",")).toString();
            //System.out.println("substring = ::"+substring+"::");
            //int intlabel = Integer.parseInt(substring);
            double label = Double.parseDouble(substring.trim());
            //System.out.println("label = "+label);
            String featureString[] = v1.substring(v1.indexOf(",") + 1).trim().split(" ");
            double[] v = new double[featureString.length];
	    //System.out.println("featureString length = "+featureString.length);
            int i = 0;
            for (String s : featureString) {
                if (s.trim().equals(""))
                    continue;
                v[i++] = Double.parseDouble(s.trim());
            }
            return new LabeledPoint(label, Vectors.dense(v));
        }
    });
    System.out.println("Training count = "+ training.countByValue());

    JavaRDD test = context.textFile(args[1]).cache().map(new Function<String,LabeledPoint> () {

        @Override
        public LabeledPoint call(String v1) throws Exception {
            double label = Double.parseDouble(v1.substring(0, v1.indexOf(",")));
            String featureString[] = v1.substring(v1.indexOf(",") + 1).trim().split(" ");
            double[] v = new double[featureString.length];
            int i = 0;
            for (String s : featureString) {
                if (s.trim().equals(""))
                    continue;
                v[i++] = Double.parseDouble(s.trim());
            }
            return new LabeledPoint(label, Vectors.dense(v));
        }
    });
    System.out.println("Test count = "+test.count());

    long startTime = System.nanoTime();    
    final NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);
    long endTime = System.nanoTime();
    System.out.println("Time taken for NaiveBayes training in nano seconds = "+(endTime-startTime));

    JavaPairRDD<Double, Double> predictionAndLabel = test.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
        @Override
        public Tuple2<Double, Double> call(LabeledPoint p) {
            return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
        }
    });

    double accuracy = 1.0 * predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
        @Override
        public Boolean call(Tuple2<Double, Double> pl) {
            //System.out.println(pl._1() + " -- " + pl._2());
            return pl._1().intValue() == pl._2().intValue();
        }
    }).count() / (double)test.count();
    System.out.println("navie bayes accuracy : " + accuracy);

    startTime = System.nanoTime();    
    final SVMModel svmModel = SVMWithSGD.train(training.rdd(), Integer.parseInt(args[2]));
    endTime = System.nanoTime();
    System.out.println("Time taken for SVM training in nano seconds = "+(endTime-startTime));
    //System.out.println("Training complete");

    JavaPairRDD<Double, Double> predictionAndLabelSVM = test.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
        @Override
        public Tuple2<Double, Double> call(LabeledPoint p) {
            return new Tuple2<Double, Double>(svmModel.predict(p.features()), p.label());
        }
    });

    double accuracySVM = 1.0 * predictionAndLabelSVM.filter(new Function<Tuple2<Double, Double>, Boolean>() {
        @Override
        public Boolean call(Tuple2<Double, Double> pl) {
            //System.out.println(pl._1() + " -- " + pl._2());
            return pl._1().intValue() == pl._2().intValue();
        }
    }).count() / (double)test.count();
    System.out.println("svm accuracy : " + accuracySVM);


    /*JavaRDD<String> file = context.textFile(args[0]);
    JavaRDD<String> words = file.flatMap(WORDS_EXTRACTOR);
    JavaPairRDD<String, Integer> pairs = words.mapToPair(WORDS_MAPPER);
    JavaPairRDD<String, Integer> counter = pairs.reduceByKey(WORDS_REDUCER);

    counter.saveAsTextFile(args[1]);*/
  }
}
