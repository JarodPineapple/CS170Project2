/* 
    Group: Jarod Hendrickson – jhend023 – Section 1
    Small Dataset Results:
	    Forward: Feature Subset: {5, 3}, Acc: 92%
	    Backward: Feature Subset: {3, 5}, Acc: 92%
    Large Dataset Results:
	    Forward: Feature Subset: {27, 1}, Acc: 95.5%
	    Backward: Feature Subset: {27}, Acc: 84.7%
    Titanic Dataset Results:
	    Forward: Feature Subset: {2}, Acc: 78.0112%
	    Backward: Feature Subset: {2}, Acc: 78.0112%
*/

#include <iostream>
#include <vector>
#include <cstdlib>
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
    if (set.empty())
    {
        cout << "{}";
        return;
    }
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

void forwardSelection(const vector<vector<double>>& feats, const vector<int>& labels, NNClass& nearestNeighbor)
{
    vector<int> current;
    vector<int> optimal;
    double optimalAccuracy = 0.0;
    double max = 0.0;
    double accuracy = 0.0;
    int optimalFeat = -1;
    bool inSet = false;

    ValClass defaultVal;
    double defaultRate = defaultVal.validator(feats, labels, {}, nearestNeighbor);

    cout << "Running nearest neighbor with no features (default rate), using leaving-one-out evaluation, I get an accuracy of " << defaultRate * 100 << "%" << endl << endl;
    cout << "Beginning search." << endl << endl;

    for (int i = 1; i <= feats[0].size(); i++)
    {
        max = 0.0;
        optimalFeat = -1;

        for (int j = 1; j <= feats[0].size(); j++)
        {
            inSet = false;
            for (int k : current)
            {
                if (k == j)
                {
                    inSet = true;
                    break;
                }
            }

            if (!inSet)
            {
                vector<int> temp = current;
                temp.push_back(j);
                ValClass valid;
                accuracy = valid.validator(feats, labels, temp, nearestNeighbor);
                cout << "Using feature(s) ";
                printSet(temp);
                cout << " accuracy is " << accuracy * 100 << "%" << endl;

                if (accuracy > max)
                {
                    max = accuracy;
                    optimalFeat = j;
                }
            }
        }

        if (optimalFeat != -1)
        {
            current.push_back(optimalFeat);
            cout << "Feature set ";
            printSet(current);
            cout << " was best, accuracy is " << max * 100 << "%" << endl << endl;
        }

        if (max > optimalAccuracy)
        {
            optimalAccuracy = max;
            optimal = current;
        }
        else
        {
            cout << "(Warning, accuracy has decreased! Continuing search in case of local maxima.)" << endl << endl;
        }
    }

    cout << "Finished search!! The best feature subset is ";
    printSet(optimal);
    cout << ", which has an accuracy of " << optimalAccuracy * 100 << "%" << endl;
}

void backwardsElimination(const vector<vector<double>>& feats, const vector<int>& labels, NNClass& nearestNeighbor)
{
    vector<int> current;
    vector<int> optimal;
    double optimalAccuracy = 0.0;
    double max = 0.0;
    double accuracy = 0.0;
    int badFeat = -1;

    if (feats.empty() || labels.empty())
    {
        cerr << "Dataset is empty or not input correctly." << endl;
        return;
    }

    for (int i = 1; i <= feats[0].size(); i++)
    {
        current.push_back(i);
    }

    ValClass valid;
    optimalAccuracy = valid.validator(feats, labels, current, nearestNeighbor);
    optimal = current;

    cout << "Using all of the features and initial evaluation, I get an accuracy of " << optimalAccuracy * 100 << "%" << endl << endl;
    cout << "Beginning search." << endl << endl;

    while (!current.empty())
    {
        max = 0.0;
        badFeat = -1;

        for (int j : current)
        {
            vector<int> temp;
            
            // Make a new set without the feature
            for (int k : current)
            {
                if (k != j)
                {
                    temp.push_back(k);
                }
            }

            accuracy = valid.validator(feats, labels, temp, nearestNeighbor);
            cout << "Using feature(s) ";
            printSet(temp);
            cout << " accuracy is " << accuracy * 100 << "%" << endl;

            if (accuracy > max)
            {
                max = accuracy;
                badFeat = j;
            }
        }

        if (badFeat != -1)
        {
            vector<int> updated;
            for (int l : current)
            {
                if (l != badFeat)
                {
                    updated.push_back(l);
                }
            }
            current = updated;

            cout << "Feature set ";
            printSet(current);
            cout << " was best, accuracy is " << max * 100 << "%" << endl << endl;
        }

        if (max > optimalAccuracy)
        {
            optimalAccuracy = max;
            optimal = current;
        }
        else
        {
            cout << "(Warning, accuracy has decreased! Continuing search in case of local maxima.)" << endl << endl;
        }
    }

    cout << "Finished search!! The best feature subset is ";
    printSet(optimal);
    cout << ", which has an accuracy of " << optimalAccuracy * 100 << "%" << endl;
}

int main()
{
    int featNum = 0;
    int choiceNum = 0;
    int flag = 1;
    string filename;
    vector<vector<double>> feats;
    vector<int> labels;
    NNClass nearestNeighbor;

    cout << "Welcome to Jarod Hendrickson's Feature Selection Algorithm." << endl << endl;
    
    cout << "Type in the number of the file to test: " << endl << endl;
    while (flag == 1)
    {
        cout << "1. small-test-dataset.txt" << endl;
        cout << "2. large-test-dataset.txt" << endl;
        cout << "3. titanic-clean.txt" << endl << endl;

        cout << "Your choice is: ";
        cin >> choiceNum;
        cout << endl;

        if (choiceNum == 1)
        {
            flag = 0;
            filename = "small-test-dataset.txt";
        }
        else if (choiceNum == 2)
        {
            flag = 0;
            filename = "large-test-dataset.txt";
        }
        else if (choiceNum == 3)
        {
            flag = 0;
            filename = "titanic-clean.txt";
        }
        else
        {
            cout << "Not a valid choice." << endl << endl;
        }
    }

    flag = 1;
    portMyData(filename, labels, feats);

    if (feats.empty() || labels.empty())
    {
        cerr << "Dataset did not load properly. Ending program." << endl;
        return -1;
    }

    cout << "Type the number of the algorithm you want to run." << endl << endl;
    while (flag == 1)
    {
        cout << "1. Forward Selection" << endl;
        cout << "2. Backward Elimination" << endl << endl;

        cout << "Your choice is: ";
        cin >> choiceNum;
        cout << endl;

        if (choiceNum == 1)
        {
            flag = 0;
            cout << "This dataset has " << feats[0].size() << " features (not including the class attribute), with " << feats.size() << " instances." << endl << endl;
            cout << "Please wait while I normalize the data... ";
            minMax(feats);
            if (feats.empty())
            {
                cerr << "Whoops, normalizing failed." << endl;
                return -1;
            }
            cout << "Done!" << endl << endl;
            forwardSelection(feats, labels, nearestNeighbor);
        }
        else if (choiceNum == 2)
        {
            flag = 0;
            cout << "This dataset has " << feats[0].size() << " features (not including the class attribute), with " << feats.size() << " instances." << endl << endl;
            cout << "Please wait while I normalize the data... ";
            minMax(feats);
            if (feats.empty())
            {
                cerr << "Whoops, normalizing failed." << endl;
                return -1;
            }
            cout << "Done!" << endl << endl;
            backwardsElimination(feats, labels, nearestNeighbor);
        }
        else
        {
            cout << "Not a valid choice." << endl << endl;
        }
    }

    return 0;
}