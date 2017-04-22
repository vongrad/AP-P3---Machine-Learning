// Advanced Programming. Andrzej Wasowski. IT University
// To execute this example, run "sbt run" or "sbt test" in the root dir of the project
// Spark needs not to be installed (sbt takes care of it)

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._



object Main {

	type Embedding       = (String, org.apache.spark.mllib.linalg.Vector)
	type ParsedReview    = (Integer, String, Double)
	type TokenizedReviewRaw    = (Integer, String, Double, Seq[String])
	type TokenizedReview    = (Integer, Double, String)
	type EmbeddingReviewJoin    = (Integer, Double, org.apache.spark.mllib.linalg.Vector)
	type EmbeddingReviewAggr    = (Integer, org.apache.spark.mllib.linalg.Vector, Double)
	type EmbeddingReviewFormatted    = (Integer, org.apache.spark.ml.linalg.Vector, Integer)


	org.apache.log4j.Logger getLogger "org"  setLevel (org.apache.log4j.Level.WARN)
	org.apache.log4j.Logger getLogger "akka" setLevel (org.apache.log4j.Level.WARN)
	val spark =  SparkSession.builder
		.appName ("Sentiment")
		.master  ("local[9]")
		.getOrCreate

  import spark.implicits._

	val reviewSchema = StructType(Array(
			StructField ("reviewText", StringType, nullable=false),
			StructField ("overall",    DoubleType, nullable=false),
			StructField ("summary",    StringType, nullable=false)))

	// Read file and merge the text abd summary into a single text column

	def loadReviews (path: String): Dataset[ParsedReview] =
		spark
			.read
			.schema (reviewSchema)
			.json (path)
			.rdd
			.zipWithUniqueId
			.map[(Integer,String,Double)] { case (row,id) => (id.toInt, s"${row getString 2} ${row getString 0}", row getDouble 1) }
			.toDS
			.withColumnRenamed ("_1", "id" )
			.withColumnRenamed ("_2", "text")
			.withColumnRenamed ("_3", "overall")
			.as[ParsedReview]


	/**
		* Load the GLoVe embeddings file
		* @param path Path to the file
		* @return Dataset[Embedding]
		*/
  def loadGlove (path: String): Dataset[Embedding] =
		spark
			.read
			.text (path)
      .map  { _ getString 0 split " " }
      .map  (r => (r.head, r.tail.toList.map (_.toDouble))) // yuck!
			.withColumnRenamed ("_1", "word" )
			.withColumnRenamed ("_2", "vec")
			.withColumn("vec", toMLLibVector($"vec"))
			.as[Embedding]

	/**
		* Parse reviews
		* @param data Data to be parsed
		* @return
		*/
	def parseReviews (data: Dataset[ParsedReview]): Dataset[TokenizedReview] = {

		val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")

		val transformedDFRaw = tokenizer.transform(data)
		val transformedDSRaw = transformedDFRaw.as[TokenizedReviewRaw]

		val transformed = for {
			r <- transformedDSRaw
			prop <- r._4
		} yield (r._1, r._2, r._3, prop)

		transformed
			.drop("_2")
			.withColumnRenamed("_1", "id")
			.withColumnRenamed("_3", "overall")
			.withColumnRenamed("_4", "word")
			.as[TokenizedReview]
	}

	def performAnalysis (data: Dataset[EmbeddingReviewFormatted]) = {

		// Split the data into train and test
		val splits = data.randomSplit(Array(0.9, 0.1), seed = 1234L)
		val train = splits(0)
		val test = splits(1)

		// specify layers for the neural network:
		// input layer of size 4 (features), two intermediate of size 5 and 4
		// and output of size 3 (classes)
		val layers = Array[Int](300, 5, 4, 3)

		// create the trainer and set its parameters
		val trainer = new MultilayerPerceptronClassifier()
			.setLayers(layers)
			.setBlockSize(128)
			.setSeed(1234L)
			.setMaxIter(100)

		// train the model
		val model = trainer.fit(train)

		// compute accuracy on the test set
		val result = model.transform(test)
		val predictionAndLabels = result.select("prediction", "label")
		val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

		println("Accuracy: " + evaluator.evaluate(predictionAndLabels))
	}

	/**
		* Convert Seq[Double] into Vector
		* @return converted Vector
		*/
	def toMLLibVector = udf[org.apache.spark.mllib.linalg.Vector, Seq[Double]](a => Vectors.dense(a.toArray))

  def main(args: Array[String]) = {

    val glove  = loadGlove ("/Users/vongrad/Downloads/reviews/glove.6B.300d.txt")
		val reviews = parseReviews(loadReviews ("/Users/vongrad/Downloads/reviews/reviews_Automotive_5.json"))

		// Join reviews with embeddings
		val joined = reviews.join(glove)
			.where(reviews("word") === glove("word"))
			.drop("word")
			.filter($"vec".isNotNull)
			.as[EmbeddingReviewJoin]

		//joined.na.drop()

		// Calculate mean of vectors grouped by id
		val aggr = joined.groupBy(joined("id")).agg(VectorSumarizer("vec").toColumn.alias("features"), mean(joined("overall")).as("label")).as[EmbeddingReviewAggr]

		val formatted = aggr.map {
			case (id, vec, rating) if rating <= 2.0 => (id, vec.asML, 0)
			case (id, vec, rating) if rating == 3.0 => (id, vec.asML, 1)
			case (id, vec, _)  => (id, vec.asML, 2)
		}.withColumnRenamed("_1", "id").withColumnRenamed("_2", "features").withColumnRenamed("_3", "label").as[EmbeddingReviewFormatted]

		performAnalysis(formatted)

		spark.stop
  }

	type Summarizer = MultivariateOnlineSummarizer

	case class VectorSumarizer(f: String) extends Aggregator[Row, Summarizer, org.apache.spark.mllib.linalg.Vector] with Serializable {
		def zero = new Summarizer
		def reduce(acc: Summarizer, x: Row) = acc.add(x.getAs[org.apache.spark.mllib.linalg.Vector](f))
		def merge(acc1: Summarizer, acc2: Summarizer) = acc1.merge(acc2)

		def finish(acc: Summarizer) = acc.mean

		def bufferEncoder: Encoder[Summarizer] = Encoders.kryo[Summarizer]
		def outputEncoder: Encoder[org.apache.spark.mllib.linalg.Vector] = ExpressionEncoder()
	}

}
