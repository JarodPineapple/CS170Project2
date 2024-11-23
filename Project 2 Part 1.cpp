#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace std;

double evaluation()
{
    double randomNum = 0.0;
    randomNum = ((rand() % 1000) / 10.0);

    return randomNum;
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

void forwardSelection(int featNum)
{
    vector<int> current;
    vector<int> optimal;
    double optimalAccuracy = 0.0;
    double max = 0.0;
    double accuracy = 0.0;
    int optimalFeat = -1;
    bool inSet = false;

    cout << "Using no features and random evaluation, I get an accuracy of " << evaluation() << "%" << endl << endl;
    cout << "Beginning search." << endl;

    for (int i = 1; i <= featNum; i++)
    {
        max = 0.0;
        optimalFeat = -1;

        for (int j = 1; j <= featNum; j++)
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
                accuracy = evaluation();
                cout << "Using feature(s) ";
                printSet(temp);
                cout << " accuracy is " << accuracy << "%" << endl;

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
            cout << " was best, accuracy is " << max << "%" << endl;
        }

        if (max > optimalAccuracy)
        {
            optimalAccuracy = max;
            optimal = current;
        }
        else
        {
            cout << "(Warning, accuracy has decreased!)" << endl;
        }
    }

    cout << endl << "Finished search!! The best feature subset is ";
    printSet(optimal);
    cout << ", which has an accuracy of " << optimalAccuracy << "%" << endl;
}

void backwardsElimination(int featNum)
{
    cout << "FIXME" << endl;
}

int main()
{
    srand(time(0));

    int featNum = 0;
    int choiceNum = 0;
    int choice = 1;

    cout << "Welcome to Jarod Hendrickson's Feature Selection Algorithm." << endl << endl;

    cout << "Please enter total number of features: ";
    cin >> featNum;
    cout << endl;
    cout << "Type the number of the algorithm you want to run." << endl << endl;

    while (choice == 1)
    {
        cout << "1. Forward Selection" << endl;
        cout << "2. Backward Elimination" << endl;

        cout << "Your choice is: ";
        cin >> choiceNum;
        cout << endl;

        if (choiceNum == 1)
        {
            choice = 0;
            forwardSelection(featNum);
        }
        else if (choiceNum == 2)
        {
            choice = 0;
            backwardsElimination(featNum);
        }
        else
        {
            cout << "Not a valid choice." << endl << endl;
        }
    }

    return 0;
}