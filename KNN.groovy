@Grab(group="org.apache.commons", module="commons-math3", version="3.6.1")
@Grab(group="com.google.guava", module="guava", version="19.0")
import com.google.common.collect.ArrayListMultimap;
import com.google.common.primitives.Doubles;
import org.apache.commons.math3.ml.distance.EuclideanDistance; 


def createDataSet() { 
    def file = new File("data.txt");
    def set = ArrayListMultimap.create(); 
    file.eachLine{line->
        def tokens = line.split("\t");
        tokens*.trim();
        def label = tokens[-1];
        def features = tokens[0..-2]*.toDouble();
        set.put(label, features);
        
    } 
    return set; 
 
} 

def normalizeFeatures(vector,matrix){
    def numFeatures = vector.size();
    def newVector = [];
    for(index in 0..<numFeatures){
        def column = getColumn(matrix,index);
        def min = column.min();
        def max = column.max();
        def range = max - min;
        def oldValue = vector[ index ];
        def newValue = (oldValue-min) / range;
        newVector[index] = newValue;
    }
    return newVector;
}
//-----------------------------------------------------------------------------
def getColumn(matrix,n){
    return matrix*.get(n);
}

//Attempts to classify the quadrant of a point based on a provided data set, the point and the nearest neighbours
def classify(point,set,k){
    def matrix = set.values(); // just the list of features [ [0,1,0], ... ]
    def points = set.entries(); // list of examples [ (A,[0,1,0]), ... ]
    def normalizedPoint = normalizeFeatures(point,matrix);
    def distanceMeasure = new EuclideanDistance();
    def distances = ArrayListMultimap.create();
    
    // calculate distances
    points.each{
        def label = it.key; // label
        def features = it.value; // feature vector
        def normalizedFeatures = normalizeFeatures(features,matrix);
        def distance = distanceMeasure.compute(Doubles.toArray(point),Doubles.toArray(features));
        // println "Distance between $point and $features is $distance"
        distances.put(label,distance);
    }
    
    // in-place sorted list, which looks like [ (B,8), (A,9), (A,10), ... ]
    def sortedDistances = distances.entries().sort{ it.value }
    println "Sorted distances \n$sortedDistances\n"
    // count the k nearest neighbors
    def counts = ArrayListMultimap.create();
    sortedDistances[0..<k].each{distance->
        def label = distance.key;
        if(counts.containsKey(label)){
            // we need to index into it becuase it's an ArrayListMultimap
            counts.get(label)[0]++;
        }else{
            def count = 1;
            counts.put(label,count);
        }
    }
    def sortedCounts = counts.entries().sort{ -1 * it.value };
    println "Majority count \n$sortedCounts\n"
    def maxCount = sortedCounts[0];
    def label = maxCount.key;
    return label;
}


def set = createDataSet(); //Creates the data set from the provided text file (data.txt)
def point = [-1, -2]; //This is the point we will attempt to classify
def k = 2;           
println "Classifying point \n $point\n"
def label = classify(point, set, k);
println "Predicted label \n$label\n";