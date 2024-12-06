#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <sstream>
using namespace std;

class NNClass
{
    private:
        vector<vector<double>> trainingFeats; // Store features
        vector<int> trainingLabels; // Store labels
    public:
        // Train the Nearest Neighbor Classifier with training data provided
        void Train(const vector<vector<double>>& feats, const vector<int>& labels);
        // Test a single instance against the training data with a subset of features
        int Test(const vector<double>& instance, const vector<int>& subset);
};

// Train Function - Assigns features and Labels
void NNClass::Train(const vector<vector<double>>& feats, const vector<int>& labels)
{
    trainingFeats = feats; // Store features into the private variable of NNClass
    trainingLabels = labels; // Store labels into the private variable of NNClass
}

// Test Function - Predicts Class Label using NN
int NNClass::Test(const vector<double>& instance, const vector<int>& subset)
{
    // Check if the training data has been stored yet
    if (trainingFeats.empty() || trainingLabels.empty())
    {
        throw runtime_error("No training data."); // Error if there's no data yet
    }

    double minDis = 1e10; // Make the mininum distance to be a huge number
    int PCL = 0; // Predicted Class Label, can be 1 or 2, initialized to 0

    // Go through all the training data to find the NN
    for (int i = 0; i < trainingFeats.size(); i++)
    {
        double dis = 0.0; // Initialize current distance to 0.0

        // Compute the Euclidean distance using the subset put into the Test() function
        for (int feat : subset)
        {
            // Check beforehand if the index of the feature is valid
            if (feat - 1 < 0 || feat - 1 >= trainingFeats[0].size())
            {
                throw runtime_error("Invalid index.");
            }
            // Calculate the squared difference. Not square rooting should have no downside
            double difference = instance[feat - 1] - trainingFeats[i][feat - 1];
            dis += difference * difference;
        }
        
        // Update the minDis and PCL if a closer neighbor is found
        if (dis < minDis)
        {
            minDis = dis;
            PCL = trainingLabels[i];
        }
    }

    return PCL; // Test function returns an int of the predicted class label
}

// Validator Function:
// Input - A feature subset, a classifier, and the dataset
// Output - Accuracy of the classifier on the dataset, when using the given feature subset
class ValClass
{
    public:
        double validator(const vector<vector<double>>& feats, const vector<int>& labels, const vector<int>& subset, NNClass& nearestNeighbor);
};

// The function uses leave-one-out cross-validation, evaluating the classifier's
// performance by leaving one instance out for testing and using the rest for training
double ValClass::validator(const vector<vector<double>>& feats, const vector<int>& labels, const vector<int>& subset, NNClass& nearestNeighbor)
{
    int ccount = 0; // Define a counter for the correct prediction
    double accuracy = 0.0; // The accuracy to be returned at the end

    if (feats.empty() || labels.empty()) // Check if my dataset is empty
    {
        throw runtime_error("Your data is empty. Cannot use the validator class.");
    }

    if (feats.size() != labels.size()) // Check if not every feature set is partnered with a label
    {
        throw runtime_error("There should be an equal number of feature sets to labels. Go check your data.");
    }

    // Loop through all instances in my dataset
    for (int i = 0; i < feats.size(); i++)
    {
        vector<vector<double>> trainFeats; // Training features (not to include current)
        vector<int> trainLabels; // Training labels (not to include current)

        // Leaving one out
        for (int j = 0; j < feats.size(); j++)
        {
            if (i != j) // We leave out the current instance
            {
                trainFeats.push_back(feats[j]);
                trainLabels.push_back(labels[j]);
            }
        }

        // Train the NN Classifier on the training dataset
        nearestNeighbor.Train(trainFeats, trainLabels);
        // Test on the instance that was left out
        int predicted = nearestNeighbor.Test(feats[i], subset);

        // Check if the prediction matches the label from the dataset
        if (predicted == labels[i])
        {
            ccount++;
        }
    }

    // Calculate accuracy as a ratio of correct predictions to the total number of instances
    accuracy = static_cast<double>(ccount) / feats.size();

    return accuracy; // Accuracy of the classifier on the dataset, when using the given feature subset
}

void portMyData(string myData, vector<int>& classType, vector<vector<double>>& feats)
{
    string row; // The line of text across
    ifstream file(myData); // Open my dataset

    if (!file.is_open())
    {
        cerr << "Error: Could not open file " << myData << endl;
        return;
    }

    while (getline(file, row)) // Keep going till I run out of rows in my dataset
    {
        stringstream ss(row); // Stringstream handles each row
        vector<double> featsVector; // Make a vector of features that are doubles
        
        double num; // Used both for the class and the feature values
        ss >> num; // The first double is the class
        classType.push_back(static_cast<int>(num)); // Turn the double to an int (should be 1 or 2) and add it to the class vector

        while (ss >> num) // Keep going till I run out of columns
        {
            featsVector.push_back(num); // Add my feature value to my vector of feature values
        }

        feats.push_back(featsVector); // Add my row of features to my vector of all features
    }
}

void minMax(vector<vector<double>>& feats) // Use this function to normalize the data
{
    if (feats.empty() || feats[0].empty())
    {
        cerr << "Dataset is empty or not complete.";
        return;
    }

    int count = feats[0].size(); // Gives the number of features (should be 10 or 40)

    for (int i = 0; i < count; i++)
    {
        double smallestVal = feats[0][i]; // Start with first number
        double biggestVal = feats[0][i]; // Start with first number
        
        for (int j = 0; j < feats.size(); j++) // Go through all the rows
        {
            if (feats[j][i] < smallestVal) // Check if the feat at that row/column is the smallest
            {
                smallestVal = feats[j][i];
            }
            if (feats[j][i] > biggestVal) // Check if the feat at that row/column is the biggest
            {
                biggestVal = feats[j][i];
            }
        }

        for (int j = 0; j < feats.size(); j++) // Go through all the rows
        {
            if (biggestVal > smallestVal) // Check if the features have a valid range to avoid dividing by 0
            {
                // Normalize the feature value to [0, 1] scale using minMax
                // Normalized value is equal to (start - smallest) / (biggest - smallest)
                feats[j][i] = (feats[j][i] - smallestVal) / (biggestVal - smallestVal);
                // Subtracting smallestVal shift the range so that the smallest value becomes 0
                // Dividing by biggestVal - smallestVal scales my range to [0, 1].
            }
            else
            {
                // If all values in the column are equal, (biggest == smallest), set the normalized number to 0.0
                feats[j][i] = 0.0;
                // In this case, all feature values are identical, so there is no need to normalize
                // I set the normalized value to 0 to ensure consistency
            }
        }
    }
}

void printSet(vector<int>& set)
{
    cout << "{";
    for (int i = 0; i < set.size(); i++)
    {
        cout << set[i];
        if (i < set.size() - 1)
        {
            cout << ", ";
        }
    }
    cout << "}";
}

int main()
{
    string small = "small-test-dataset.txt";
    string large = "large-test-dataset.txt";
    string subsetSmall = "{3, 5, 7}";
    string subsetLarge = "{1, 15, 27}";
    vector<vector<double>> feats;
    vector<int> labels;
    NNClass nearestNeighbor;
    ValClass valid;
    double accuracy = 0.0;

    cout << "Jarod Hendrickson's Project 2 Part 2 Trace." << endl << endl;

    cout << "Commencing timed test for NN classifier and the validator." << endl;
    cout << "Testing on small-test-dataset.txt." << endl;
    cout << "If we use only features {3, 5, 7}, accuracy should be about 0.89." << endl << endl;

    cout << "Loading " << small << endl;
    auto start = chrono::high_resolution_clock::now();
    portMyData(small, labels, feats);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> loadT = end - start;
    if (feats.empty() || labels.empty())
    {
        cerr << "Dataset did not load properly. Ending program." << endl;
        return -1;
    }
    cout << "Done loading. Time: " << loadT.count() << "s" << endl << endl;

    cout << "Normalizing " << small << endl;
    start = chrono::high_resolution_clock::now();
    minMax(feats);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> normalT = end - start;
    cout << "Done normalizing. Time: " << normalT.count() << "s" << endl << endl;

    cout << "Picking subset of " << small << endl;
    vector<int> subsetS = {3, 5, 7};
    cout << "Done Picking. Subset is " << subsetSmall << endl << endl;

    cout << "Training and testing with validator for " << small << endl;
    start = chrono::high_resolution_clock::now();
    accuracy = valid.validator(feats, labels, subsetS, nearestNeighbor);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> validT = end - start;
    cout << "Done training, testing, and validating. Time: " << validT.count() << "s" << endl << endl;

    cout << "Expected Accuracy: 0.89" << endl;
    cout << "Computed Accuracy: " << accuracy << endl << endl;

    feats.clear();
    labels.clear();

    cout << "Commencing timed test for NN classifier and the validator." << endl;
    cout << "Testing on large-test-dataset.txt." << endl;
    cout << "If we use only features {1, 15, 27}, accuracy should be about 0.949." << endl << endl;

    cout << "Loading " << large << endl;
    start = chrono::high_resolution_clock::now();
    portMyData(large, labels, feats);
    end = chrono::high_resolution_clock::now();
    loadT = end - start;
    cout << "Done loading. Time: " << loadT.count() << "s" << endl << endl;

    cout << "Normalizing " << large << endl;
    start = chrono::high_resolution_clock::now();
    minMax(feats);
    end = chrono::high_resolution_clock::now();
    normalT = end - start;
    cout << "Done normalizing. Time: " << normalT.count() << "s" << endl << endl;

    cout << "Picking subset of " << large << endl;
    vector<int> subsetL = {1, 15, 27};
    cout << "Done Picking. Subset is " << subsetLarge << endl << endl;

    cout << "Training and testing with validator for " << large << endl;
    start = chrono::high_resolution_clock::now();
    accuracy = valid.validator(feats, labels, subsetL, nearestNeighbor);
    end = chrono::high_resolution_clock::now();
    validT = end - start;
    cout << "Done training, testing, and validating. Time: " << validT.count() << "s" << endl << endl;

    cout << "Expected Accuracy: 0.949" << endl;
    cout << "Computed Accuracy: " << accuracy << endl << endl;

    return 0;
}