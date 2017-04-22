import org.scalatest.{FreeSpec, Matchers, BeforeAndAfterAll}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Dataset

class SentimentSpec extends FreeSpec with Matchers with BeforeAndAfterAll {

  org.apache.log4j.Logger getLogger "org"  setLevel (org.apache.log4j.Level.WARN)
  org.apache.log4j.Logger getLogger "akka" setLevel (org.apache.log4j.Level.WARN)

  val spark =  SparkSession.builder
    .appName ("Sentiment")
    .master  ("local[12]")
    .getOrCreate

  override def afterAll = spark.stop

  import spark.implicits._


  val glove  = Main.loadGlove ("path/to/glove/file/in/your/filesystem")

  "put your tests here"   - {

    "nested tests group" - {
      "individual test #1" in { ??? }
      "individual test #2" in { ??? }
    }

  }

}
