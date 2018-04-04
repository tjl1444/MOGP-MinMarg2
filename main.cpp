// Example of multi-objective steady-state GP for classification -- pir -- 17.4.2014

// REVISION HISTORY:
// Created from MO-SS-Regression-Example -- pir -- 17.4.2014
// Delay initialisation of g_uNoTreeEvaluations to just before main evolutionary loop: issue with counting function evaluations in multiple initialisations -- pir -- 2.9.2014
// Modified deletion of previous population in initialisation + removed inclusion of CGaussian header -- pir -- 21.1.2015
// Added scan search to find optimum threshold -- pir -- 6.3.2015
// Set tree nodes size to INFINITY in TrainingSetEvaluation() if response interval is too small -- pir -- 11.3.2015

//*****************************************************************************

#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <limits>
#include <fstream>

#include <CMatrix/CVector.h>
#include <CMatrix/CMatrix.h>
#include <Evolutionary/CPopulation/CPopulation.h>
#include <Evolutionary/GP/GPNodes.h>
#include <Generic/BinaryFileUtilities/BinaryFileUtilities.h>
#include <Optimisation/COBYLA/COBYLA.h>
#include <Optimisation/GoldenSectionSearch/GoldenSectionSearch.h>

using namespace std;
/**
TODO:
Calculate distance matrix and starting points

**/

// Globals

CHROMOSOME g_pstRootNode;
double* g_adY;
CMatrix* g_pDistanceMatrix;



//-----------------------------------------------------------------------------
// Global constants

const int32_t g_nNoConstraints = 2;
double g_dRhoBegin = 0.5;  //Was 0.25 - 0.5 is good
double g_dRhoEnd = 1e-6;
int32_t g_nMessageLevel = 0;
int32_t g_nFunctionEvaluations = 100000;

uint32_t g_uNoInitialisationRepeats = 1;
const uint32_t g_uPopulationSize = 100;

const enum enInitialisationMethod_t
    {
    enRampedHalfHalf, enRandomSizes
    }
g_enInitialisationMethod = enRandomSizes;

// Random number base seed values
const int64_t g_lnGP_BaseSeed = 10000L;
const int64_t g_lnLocalSeed = 1867L;
const int64_t g_lnDataSeed = 220L;

const uint32_t g_uRandomTreeNodeMax = 63;
const uint32_t g_uMaxInitialDepth = 6;
const double g_dTerminalProbability =  0.1;
const double g_dConstantProbability = 0.1;
const double g_dUnaryProbability = 0.1;
const double g_dBinaryProbability = 1.0 - g_dTerminalProbability - g_dConstantProbability - g_dUnaryProbability;
const double g_dTernaryProbability = 0.0;
const double g_dConstantRange = 1.0;
const double g_dConstantPrecision = 0.1;

const enum enOffspringSelect_t
    {
    enOneChild, enBothChildren
    }
g_enOffspringSelect = enBothChildren;

// Crossover parameters
pfnCrossOverOperator_t g_pfnCrossOverOperator = PointCrossOver;
const double g_dCrossOverFunctionNodeBias = 0.9;

// Mutation parameters
const double g_dMutationProbability = 1.0;
pfnMutateOperator_t g_pfnMutateOperator = PointMutate;
const enReplacementTree_t g_enReplacementTree = enGP_FULL_DEPTH;
const uint32_t g_uMutationDepth = 4;

const uint32_t g_uMaxNoTreeEvaluations = 20000;

//-----------------------------------------------------------------------------

// Datasets: Assumes labels tags are uint16_t in {0,1}
CDataset<uint16_t> g_TrainingSet;
CDataset<uint16_t> g_TestSet;

// Pointers for global arrays to calculate trainings set threshold
double* g_padTrainingSetResponses = NULL;
uint16_t* g_psuLabel = NULL;

uint32_t g_uNoTreeEvaluations;	// No of tree evaluations

//-----------------------------------------------------------------------------

class CFitnessVector : public CColumnVector
// Definition of fitness vector class
    {
    public:
        CFitnessVector() : CColumnVector(2)
            {
            /* EMPTY*/
            };
        ~CFitnessVector()
            {
            /* EMPTY*/
            };
    };

//-----------------------------------------------------------------------------

bool operator < (CFitnessVector& A, CFitnessVector& B)
// Pareto comparison of CFitnessVector record: A < B
    {
    assert(A.NoElements() == B.NoElements());

    // Check A[i] <= B[i] for all i
    for(uint32_t j = 1; j <= A.NoElements(); j++)
        {
        if(A[j] > B[j])
            {
            return false;
            }
        }

    // Check that A[i] < B[i] for at least one i
    for(uint32_t j = 1; j <= A.NoElements(); j++)
        {
        if(A[j] < B[j])
            {
            return true;
            }
        }

    return false;
    } // operator < ()

//-----------------------------------------------

bool operator > (CFitnessVector& A, CFitnessVector& B)
// Pareto comparison of CFitnessVector record: A > B
    {
    assert(A.NoElements() == B.NoElements());

    // Check A[i] >= B[i] for all i
    for(uint32_t j = 1; j <= A.NoElements(); j++)
        {
        if(A[j] < B[j])
            {
            return false;
            }
        }

    // Check that A[i] > B[i] for at least one i
    for(uint32_t j = 1; j <= A.NoElements(); j++)
        {
        if(A[j] > B[j])
            {
            return true;
            }
        }

    return false;
    } // operator > ()

//-----------------------------------------------------------------------------

void DumpDecisionBoundary(stGPNode_t* pTree, char* pszFilename)
    {
    const double dX1_min = -6.0;
    const double dX1_max = +6.0;
    const double dX2_min = -6.0;
    const double dX2_max = +6.0;  //+1,-1
    const double dX_step = 0.001;  //0.01
    const double dThreshold = 0.0005;  //0.005

    FILE* pFile = fopen(pszFilename, "w");
    if(pFile == NULL)
        {
        ErrorHandler("Unable to open decision boundary file");
        }

    double dX1 = dX1_min;
    while(dX1 <= dX1_max)
        {
        double dX2 = dX2_min;
        while(dX2 <= dX2_max)
            {
            CColumnVector x(2);
            x[1] = dX1;
            x[2] = dX2;
            const double dY = TreeEvaluate(x, pTree);
            if(fabs(dY) <= dThreshold)
                {
                fprintf(pFile, "%lf, %lf\n", dX1, dX2);
                }
            dX2 += dX_step;
            }
        dX1 += dX_step;
        }

    fclose(pFile);

    return;
    } // DumpDecisionBoundary()

//-----------------------------------------------------------------------------

double g(CColumnVector& PatternVector, CHROMOSOME pstRootNode)
//Discriminant function
    {
    return TreeEvaluate(PatternVector, pstRootNode); //returns GP(x)
    }

double g2(CColumnVector& PatternVector, void* pParameters)
//Square of discriminant function
    {
    CHROMOSOME pstRootNode = static_cast<CHROMOSOME>(pParameters);
    return (g(PatternVector, pstRootNode) * g(PatternVector, pstRootNode));
    }

//-----------------------------------------------------------------------------
COBYLA_Function_t COBYLA_Function;

int32_t COBYLA_Function(
    const int32_t nNoDimensions,
    const int32_t nNoConstraints,
    double* padX,
    double* padFnValue,
    double* padConstraints,
    void* pvState)
    {
    CColumnVector pdX((static_cast<uint32_t>(nNoDimensions)));

    // Minimise ||Xd-Xs||^2 subject to g(padX) = 0
    (*padFnValue) = 0.0;
    for(int32_t i = 0; i < nNoDimensions; i++)
        {
        *padFnValue += ((padX[i] - g_adY[i]) * (padX[i] - g_adY[i]));

        //Set up Column Vector to use for calculating constraints with g(X)
        pdX[i + 1] = padX[i];
        }

    // DEBUG
    // Calculate constraints
    padConstraints[0] = g(pdX, g_pstRootNode);
    padConstraints[1] = -g(pdX, g_pstRootNode);





    return 0;
    }

//-----------------------------------------------------------------------------

bool isMisclassified(CHROMOSOME pstRootNode, CColumnVector PatternVector, const uint16_t suLabel)
// Returns true if the point is misclassified
    {
    const double dTreeOutput = TreeEvaluate(PatternVector, pstRootNode);
    bool returnValue = false;

    if((dTreeOutput < 0.0) and (suLabel != 0))
        {
        returnValue = true;
        }

    if((dTreeOutput >= 0.0) and (suLabel != 1))
        {
        returnValue = true;
        }

    return returnValue;
    }

//-----------------------------------------------------------------------------
double TrainingSetError(CHROMOSOME pstRootNode)
    {
// Returns expected 0/1 loss over training set

    const uint32_t  uVectorLength = g_TrainingSet.VectorLength();
    CColumnVector PatternVector(uVectorLength);

    uint32_t uNoErrors = 0;
    for(uint32_t i = 1; i <= g_TrainingSet.NoStoredPatterns(); i++)
        {
        PatternVector = g_TrainingSet[i];
        const double dTreeOutput = TreeEvaluate(PatternVector, pstRootNode); // return GP(x)
        const uint16_t suTag = g_TrainingSet.Tag(i);

        if((dTreeOutput < 0.0) and (suTag != 0))
            {
            uNoErrors++;
            }

        if((dTreeOutput >= 0.0) and (suTag != 1))
            {
            uNoErrors++;
            }
        }

    const double dMisclassificationError = static_cast<double>(uNoErrors) / static_cast<double>(g_TrainingSet.NoStoredPatterns());
    return dMisclassificationError;


    }

//-----------------------------------------------------------------------------

double CalculateMargin(CColumnVector& InitialVector, CColumnVector& PatternVector)
// Returns square of the Margin for the PatternVector which is a misclassified point
    {
    //Initialize COBYLA parameters
    const int32_t nNoDimensions = g_TrainingSet.VectorLength();
    double x[nNoDimensions];
    g_adY = new double[nNoDimensions];

    double margin2 = 0.0;

    //Set up initial estimate of solution
    for(uint32_t i = 0; i < (static_cast<uint32_t>(nNoDimensions)); i++)
        {
        x[i] = InitialVector[i + 1];
        g_adY[i] = PatternVector[i + 1];
        }


    int32_t nLocalMaxFnEvaluations = g_nFunctionEvaluations;    // HACK!
    g_nFunctionEvaluations = 100000;
    //cout << "FnEval Before:" <<  nLocalMaxFnEvaluations << endl;
    int32_t nReturnValue = COBYLA(nNoDimensions, g_nNoConstraints, x, g_dRhoBegin, g_dRhoEnd, g_nMessageLevel, &g_nFunctionEvaluations, COBYLA_Function, NULL);
    g_nFunctionEvaluations = 100000;
    //cout << "FnEval After:" << nLocalMaxFnEvaluations << endl;
    int counter = 0;
    while(nReturnValue != 0)
        {
        //g_nMessageLevel = 1;

        //Improves the optimisation points returned on the first run with COBYLA if it ran out of function evaluations
        nReturnValue =  COBYLA(nNoDimensions, g_nNoConstraints, x, g_dRhoBegin, g_dRhoEnd, g_nMessageLevel, &g_nFunctionEvaluations, COBYLA_Function, NULL);
        g_nFunctionEvaluations = 100000;

        if(nReturnValue != 0)
            {
            nReturnValue = COBYLA(nNoDimensions, g_nNoConstraints, x, g_dRhoBegin, g_dRhoEnd, g_nMessageLevel, &g_nFunctionEvaluations, COBYLA_Function, NULL);
            g_nFunctionEvaluations = 100000;

            if(nReturnValue != 0)
                {
                //DEBUG
                cout << "Cobyla Error No." << nReturnValue << endl;
                cout << "TreeEvaluate - Initial: " << TreeEvaluate(InitialVector, g_pstRootNode) << endl;

                cout << "The point we are calculating distance from X1:" << PatternVector[1] << endl;
                cout << "The point we are calculating distance from X2:" << PatternVector[2] << endl;

                cout << "Starting point X1::" << InitialVector[1] << endl;
                cout << "Starting point X2:" << InitialVector[2] << endl;

                cout << "Optimisation endpoint X1: " << x[0] << endl;
                cout << "Optimisation endpoint X2: " << x[1] << endl;

                // DEBUG
                for(uint32_t i = 0; i < (static_cast<uint32_t>(nNoDimensions)); i++)
                    {
                    InitialVector[i + 1] = x[i];
                    }
                cout << "TreeEvaluate - after COBYLA: " << TreeEvaluate(InitialVector, g_pstRootNode) << endl;
                DumpDecisionBoundary(g_pstRootNode,"RipleyDecisionBoundary.txt");

                //Output Tree
//                char szStringForTree[4096];
//                strcpy(szStringForTree, "");
//                TreeToString(g_pstRootNode, szStringForTree, enTreeToStringMaths);
//                cout << szStringForTree << endl;
                margin2 = INFINITY;
                return margin2;
                }

            }

        }



    //Calculate margin
    for(uint32_t i = 0; i < (static_cast<uint32_t>(nNoDimensions)); i++)
        {
        margin2 += ((x[i] - g_adY[i]) * (x[i] - g_adY[i]));
        }

    // DEBUG - Check accuracy of COBYLA solution
//    double dCobylaResult = TreeEvaluate(InitialVector, g_pstRootNode);
//    if(dCobylaResult > 0.01)
//        {
//        cout << "COBYLA Accuracy " << dCobylaResult << endl;
//        //exit(0);
//        }
//







    return margin2;
    } // CalculateMargin()

//-----------------------------------------------------------------------------
double GetLargestMargin(const CHROMOSOME pstRootNode)
    {

    uint32_t uNoData = g_TrainingSet.NoStoredPatterns();
    double dLargestMargin = 0.0;
    g_pstRootNode = pstRootNode;


    //Find missclassified points
    bool abMisclassified[uNoData];

    for(uint32_t i = 1; i <= uNoData; i++)
        {
        CColumnVector x = g_TrainingSet[i];
        const uint16_t suLabel = g_TrainingSet.Tag(i);
        abMisclassified[i - 1] = isMisclassified(pstRootNode, x, suLabel);
        }

    // When f(X) doesn't go through f(X) = 0 then a decision boundary won't exist
    int positiveCounter = 0;
    int negativeCounter = 0;
    for(uint32_t k = 1; k <= uNoData; k++)
        {
        double dTreeOutput = TreeEvaluate(g_TrainingSet[k], pstRootNode);

        if(dTreeOutput >= 0)
            {
            positiveCounter++;
            }

        if(dTreeOutput < 0)
            {
            negativeCounter++;
            }

        if(positiveCounter > 0 and negativeCounter > 0)
            {
            break;
            }

        if(k == uNoData)
            {
            dLargestMargin = INFINITY;
            return dLargestMargin;
            }
        }


    for(uint32_t i = 1; i <= uNoData; i++)
        {
        // Take a misclassified point and find the largest margin using all points across boundary
        if(abMisclassified[i - 1] == true)
            {
            for(uint32_t j = 1; j <= uNoData; j++)
                {
                //Find a point across the boundary
                if(((TreeEvaluate(g_TrainingSet[i], pstRootNode) < 0) and (TreeEvaluate(g_TrainingSet[j], pstRootNode) >= 0))  or ((TreeEvaluate(g_TrainingSet[i], pstRootNode) >= 0) and (TreeEvaluate(g_TrainingSet[j], pstRootNode) < 0)))
                    {
                    //Calculate initial estimate of boundary vector
                    CColumnVector TargetPoint = g_TrainingSet[i];
                    CColumnVector PointAcrossBoundary = g_TrainingSet[j];
                    double dFuncMin;
                    const double dAlpha = GoldenSectionLineSearch(TargetPoint, PointAcrossBoundary, g2, static_cast<void*>(g_pstRootNode), dFuncMin);

                    // DEBUG - May help to speed up optimisation as GSS is returning bad points
//                    if((dFuncMin > 0.1) and (j == uNoData))
//                    {
//                        cout << "The function minimum is: " << dFuncMin << endl;
//                        cout << "The alpha value is: " << dAlpha << endl;
//
//                    }
                    CColumnVector InitialVector = (TargetPoint * (1.0 - dAlpha)) + (dAlpha * PointAcrossBoundary);    // Initial point on decision surface


                    //Calculate the margin and compare with current largest margin
                    double dInitialMargin = (p2_Norm(TargetPoint - InitialVector)) * (p2_Norm(TargetPoint - InitialVector));
                    double dMargin = CalculateMargin(InitialVector, TargetPoint);

                    if((dMargin == INFINITY) and (j == uNoData))
                        {
                        cout << "The Target point - x1:x2 is: " <<  TargetPoint[1] << ":" << TargetPoint[2] << endl;
                        cout << "The Point across the boundary - x1:x2 is: " <<  PointAcrossBoundary[1] << ":" << PointAcrossBoundary[2] << endl;
                        cout << "The function minimum is: " << dFuncMin << endl;

//                        exit(0);
                        }

                    // Basic check to see if COBYLA has actually done some optimisation
                    if((dMargin > dLargestMargin) and (dFuncMin < 0.1))
                        {
                        if(dMargin <= dInitialMargin)
                            {
                            // This only works if the the optimisation problem is non-convex
                            dLargestMargin = dMargin;
                            //end loop
                            break;

                            }


                        }
                    // CHECK - Is this necessary?
                    else if((dMargin != INFINITY) and (dMargin > dLargestMargin))
                        {
                        dLargestMargin = dMargin;

                        }

                    // DEBUG
                    if(dMargin == INFINITY)
                        {
                        cout << "The point chosen across the boundary X1: " << PointAcrossBoundary[1] << endl;
                        cout << "The point chosen across the boundary X2: " << PointAcrossBoundary[2] << endl;
//                        exit(0);
                        }


                    }
                }
            }
        }



    if(dLargestMargin == 0 and TrainingSetError(pstRootNode) > 0)
        {
        cout << "The training error for this classifier (0 margin) is: " << TrainingSetError(pstRootNode) << endl;
        dLargestMargin = INFINITY;
        }

    return dLargestMargin;


    } // GetLargestMargin()
//-----------------------------------------------------------------------------



inline double CalculateTrainingSetError(const double dThreshold)
// Return error count for specified threshold
    {
    uint32_t uNoErrors = 0;
    const uint32_t uNoTrainingPatterns = g_TrainingSet.NoStoredPatterns();
    for(uint32_t i = 1; i <= uNoTrainingPatterns; i++)
        {
        if(g_padTrainingSetResponses[i] < dThreshold)
            {
            // Predicted class = 0
            if(g_psuLabel[i] != 0)
                {
                uNoErrors++;
                }
            }
        else // g_padTrainingSetResponses[i] >= dThreshold
            {
            // Predicted class = 1
            if(g_psuLabel[i] != 1)
                {
                uNoErrors++;
                }
            }
        }

    return static_cast<double>(uNoErrors) / static_cast <double>(uNoTrainingPatterns);
    } // CalculateTrainingSetError()

//-----------------------------------------------------------------------------

CFitnessVector TrainingSetEvaluation(const CHROMOSOME pstRootNode)
// Returns fitness vector: node count & 0/1 loss
    {
    g_uNoTreeEvaluations++;

    CFitnessVector FitnessVector;
    FitnessVector[1] = NoTreeNodes(pstRootNode, true);

    const uint32_t uVectorLength = g_TrainingSet.VectorLength();
    CColumnVector PatternVector(uVectorLength);
    const uint32_t uNoStoredPatterns = g_TrainingSet.NoStoredPatterns();

    // Calculate number of errors in training set vector
    uint32_t uNoErrors = 0;
    for(uint32_t i = 1; i <= uNoStoredPatterns; i++)
        {
        PatternVector = g_TrainingSet[i];
        const double y = TreeEvaluate(PatternVector, pstRootNode); // return GP(x)
        const uint16_t suLabel = g_TrainingSet.Tag(i);

        if((y < 0) and (suLabel != 0))
            {
            uNoErrors++;
            }

        if((y >= 0) and (suLabel != 1))
            {
            uNoErrors++;
            }
        }
    // Calculate classification error over the training set
    FitnessVector[2] = static_cast<double>(uNoErrors) / static_cast<double>(uNoStoredPatterns);

    return FitnessVector;
    } // TrainingSetEvaluation()

//-----------------------------------------------------------------------------

CFitnessVector TrainingSetEvaluation2(CHROMOSOME pstRootNode)
// Returns fitness vector: node count & largest margin
    {
    g_uNoTreeEvaluations++;
    CFitnessVector FitnessVector;

    //Calculate number of nodes in tree

    FitnessVector[1] = NoTreeNodes(pstRootNode, true);

    // Calculate smallest margin for that individual
    FitnessVector[2] = GetLargestMargin(pstRootNode);

    return FitnessVector;
    } // TrainingSetEvaluation2()

//-----------------------------------------------------------------------------

inline double TestSetEvaluation(CHROMOSOME pstRootNode)
// Returns expected 0/1 loss over test set
    {
    const uint32_t  uVectorLength = g_TrainingSet.VectorLength();
    CColumnVector PatternVector(uVectorLength);

    uint32_t uNoErrors = 0;
    for(uint32_t i = 1; i <= g_TestSet.NoStoredPatterns(); i++)
        {
        PatternVector = g_TestSet[i];
        const double dTreeOutput = TreeEvaluate(PatternVector, pstRootNode); // return GP(x)
        const uint16_t suTag = g_TestSet.Tag(i);

        if((dTreeOutput < 0.0) and (suTag != 0))
            {
            uNoErrors++;
            }

        if((dTreeOutput >= 0.0) and (suTag != 1))
            {
            uNoErrors++;
            }
        }

    const double dMisclassificationError = static_cast<double>(uNoErrors) / static_cast<double>(g_TestSet.NoStoredPatterns());
    return dMisclassificationError;
    } // TestEvaluation()

//-----------------------------------------------------------------------------



//bool ReadFitness(FILE* pFile, stChromosomeTag_t& stChromosomeTag)
//	{
//	double dTemp1;
//	if(!ReadDoubleFromFile(pFile, dTemp1))
//		{
//		return false;
//		}
//	stChromosomeTag.m_Fitness[1] = dTemp1;
//
//	double dTemp2;
//	if(!ReadDoubleFromFile(pFile, dTemp2))
//		{
//		return false;
//		}
//	stChromosomeTag.m_Fitness[2] = dTemp2;
//
//	return true;
//	} // ReadFitness()

//-----------------------------------------------------------------------------

//bool WriteFitness(FILE* pFile, stChromosomeTag_t& stChromosomeTag)
//	{
//	const double dTemp1 = stChromosomeTag.m_Fitness[1];
//	if(!WriteDoubleToFile(pFile, dTemp1))
//		{
//		return false;
//		}
//
//	const double dTemp2 = stChromosomeTag.m_Fitness[2];
//	if(!WriteDoubleToFile(pFile, dTemp2))
//		{
//		return false;
//		}
//
//	return true;
//	} // WriteFitness()

//-----------------------------------------------------------------------------

bool ReadChromosome(FILE* pFile, stGPNode_t*& pTree)
// Read chromosome from a file
    {
    pTree = ReadTreeFromFile(pFile);

    return (pTree != NULL);
    } // ReadChromosome()

//-----------------------------------------------------------------------------

bool WriteChromosome(FILE* pFile, stGPNode_t* pTree)
// Write a chromosome to a file
    {
    return WriteTreeToFile(pFile, pTree);
    } // WriteChromosome()

//-----------------------------------------------------------------------------
// NEED TO FIGURE OUT A WAY TO KEEP THIS GLOBAL AND ASSIGN  NEW VALUE OF g_uNoInitialisationRepeats
stGP_Parameters_t g_stGP_Parameters(g_lnGP_BaseSeed, g_uNoInitialisationRepeats);

//*****************************************************************************

int main(int argc, char* argv[])
    {

    //Open file to write results
    std::fstream GPResults;
    std::fstream GPTrees;

    GPResults.open("RipleyResults.txt", std::fstream::in | std::fstream::out | std::fstream::app);
    GPTrees.open("RipleyTrees.txt", std::fstream::in | std::fstream::out | std::fstream::app);


    // Process command line
    const uint32_t g_uNoInitialisationRepeats = atoi(argv[1]);  // Causes Segmentation fault if you run from IDE

    if(g_uNoInitialisationRepeats < 1)
        {
        ERROR_HANDLER("Initialisation number incorrectly set");
        }


    // Load training and test datasets
    // Ripley2TrainingSet.dat  2D-GaussianTraining.dat Hastie_Training.dat
    g_TrainingSet.Load("Ripley2TrainingSet.dat");
    g_TestSet.Load("Ripley2TestSet.dat");

    assert(g_TrainingSet.VectorLength() == g_TestSet.VectorLength());	// Sensible general check, especially when loading datasets from files!

    //-----------------------------------------------

    //Calculate inter-point distances and store in a matrix
    const uint32_t uNoData = g_TrainingSet.NoStoredPatterns();
    g_pDistanceMatrix = new CMatrix(uNoData, uNoData);

    for(uint32_t i = 1; i <= uNoData; i++)
        {
        (*g_pDistanceMatrix)[i][i] = NAN;
        }

    for(uint32_t i = 1; i <= (uNoData - 1); i++)
        {
        for(uint32_t j = (i + 1); j <= uNoData; j++)
            {
            (*g_pDistanceMatrix)[i][j] = p2_Norm(g_TrainingSet[i] - g_TrainingSet[j]);
            (*g_pDistanceMatrix)[j][i] = (*g_pDistanceMatrix)[i][j];
            }
        }

    //-----------------------------------------------

    // Set MOGP parameters
    const uint32_t uVectorLength  = g_TrainingSet.VectorLength();
    g_stGP_Parameters.SetVectorLength(uVectorLength);
    g_stGP_Parameters.SetMutationDepth(g_uMutationDepth);

    // Set constant parameters
    g_stGP_Parameters.SetConstantRange(g_dConstantRange);
    g_stGP_Parameters.SetConstantPrecision(g_dConstantPrecision);

    // Set node selection probabilities
    g_stGP_Parameters.SetNodeSelectionProbabilities(g_dTerminalProbability, g_dConstantProbability, g_dUnaryProbability, g_dBinaryProbability, g_dTernaryProbability);

    //-----------------------------------------------

    CPopulation<CHROMOSOME, CFitnessVector> Population(g_uPopulationSize);

    CUniform2* pMutationSelector = NULL;
    CUniform2* pOffspringSelector = NULL;
    CUniform2* pRandomInitialTreeSizeSelector = NULL;

    assert(g_uNoInitialisationRepeats >= 1);
    int64_t lnLocalSeed = g_lnLocalSeed;
    for(uint32_t i = 1; i <= g_uNoInitialisationRepeats; i++)
        {
        // Initialise mutation and offspring selectors
        if(pMutationSelector != NULL)
            {
            delete pMutationSelector;
            }
        lnLocalSeed++;
        pMutationSelector = new CUniform2(lnLocalSeed);

        if(pOffspringSelector != NULL)
            {
            delete pOffspringSelector;
            }
        lnLocalSeed++;
        pOffspringSelector = new CUniform2(lnLocalSeed);

        if(pRandomInitialTreeSizeSelector != NULL)
            {
            delete pRandomInitialTreeSizeSelector;
            }
        lnLocalSeed++;
        pRandomInitialTreeSizeSelector = new CUniform2(lnLocalSeed);

        // Delete previous population (apart from the first time through the loop)
        if(i > 1)
            {
            for(uint32_t j = 1; j <= g_uPopulationSize; j++)
                {
                DeleteChromosome(Population[j]);
                }
            }

        //Creation of initial population (half full-depth , half random depth)
        cout << "Creating initial population (" << i << ")..." << endl;
        for(uint32_t j = 1; j <= (g_uPopulationSize / 2); j++)
            {
            CHROMOSOME pTree;
            if(g_enInitialisationMethod == enRampedHalfHalf)
                {
                // Create full-depth trees
                pTree = CreateRandomTree1(g_uMaxInitialDepth, true);
                }
            else
                {
                // Create random depth trees
                const double dTreeSizeSelector = pRandomInitialTreeSizeSelector->NextVariate();
                const double dTreeSize = (static_cast<double>(g_uRandomTreeNodeMax - 1) * dTreeSizeSelector) + 1.0;
                const uint32_t uTreeSize = static_cast<uint32_t>(round(dTreeSize));
                pTree = CreateRandomTree2(uTreeSize);
                }
            // Assign fitness to each individual in the population
            Population[j] = pTree;
            Population.Fitness(j) = TrainingSetEvaluation2(Population[j]);
            }

        for(uint32_t j = ((g_uPopulationSize / 2) + 1); j <= (g_uPopulationSize + 2); j++)
            {
            CHROMOSOME pTree;
            if(g_enInitialisationMethod == enRampedHalfHalf)
                {
                // Create half-depth trees
                pTree = CreateRandomTree1(g_uMaxInitialDepth, false);
                }
            else
                {
                // Create random depth trees
                const double dTreeSizeSelector = pRandomInitialTreeSizeSelector->NextVariate();
                const double dTreeSize = (static_cast<double>(g_uRandomTreeNodeMax - 1) * dTreeSizeSelector) + 1.0;
                const uint32_t uTreeSize = static_cast<uint32_t>(round(dTreeSize));
                pTree = CreateRandomTree2(uTreeSize);
                }
            // Assign fitness to each individual in the population
            Population[j] = pTree;
            Population.Fitness(j) = TrainingSetEvaluation2(Population[j]);
            }
        }

    Population.MOSort(enASCENDING);

    // Print initial population
    cout << "Initial population..." << endl;
    for(uint32_t i = 1; i <= g_uPopulationSize; i++)
        {
        cout << i
             << "   Node count = "
             << Population.Fitness(i)[1]
             << ", Margin = "
             << Population.Fitness(i)[2]
             << ",  Rank = "
             << Population.Rank(i)
             << endl;
        }
    cout << endl;

//    // Print Problem Trees
//    cout << "Initial population..." << endl;
//    for(uint32_t i = 1; i <= g_uPopulationSize; i++)
//
//        {
//
//
//        if((Population.Fitness(i)[2] == INFINITY) or (Population.Fitness(i)[1]  < 10))
//            {
//            //Output Tree Information
//            cout << i
//                 << "   Node count = "
//                 << Population.Fitness(i)[1]
//                 << ", Margin = "
//                 << Population.Fitness(i)[2]
//                 << ",  Rank = "
//                 << Population.Rank(i)
//                 << endl;
//            cout << endl;
//
//
//            //Output Tree
//            char szStringForTree[4096];
//            strcpy(szStringForTree, "");
//            TreeToString(g_pstRootNode, szStringForTree, enTreeToStringMaths);
//            cout << szStringForTree << endl;
//            cout << endl;
//
//            }
//        }
//    cout << endl;
//
//    exit(0);

    //-----------------------------------------------
    // Start of genetic evolution loop using margin

    uint32_t uNoIterations = 0;
    g_uNoTreeEvaluations = 0;

    double dMinLargestMargin;
    uint16_t uMinLargestMarginIndex;
    cout << "Entering evolutionary loop the 2nd time..." << endl;

    do
        {
        uNoIterations++;
        if((uNoIterations % 1000) == 0)
            {
            cout << "No of iterations = " << uNoIterations << endl;
            }

        uint32_t uParent1Index;
        uint32_t uParent2Index;
        Population.SelectParents(uParent1Index, uParent2Index);


        // Perform crossover & mutation
        CHROMOSOME Parent1Chromosome = Population[uParent1Index];
        CHROMOSOME Parent2Chromosome = Population[uParent2Index];
        CHROMOSOME Child1Chromosome;
        CHROMOSOME Child2Chromosome;
        g_pfnCrossOverOperator(Parent1Chromosome, Parent2Chromosome, &Child1Chromosome, &Child2Chromosome, g_dCrossOverFunctionNodeBias);

        const double dMutateSelector = pMutationSelector->NextVariate();
        if(dMutateSelector <= g_dMutationProbability)
            {
            g_pfnMutateOperator(&Child1Chromosome, g_enReplacementTree);
            g_pfnMutateOperator(&Child2Chromosome, g_enReplacementTree);
            }

        // Evaluate child fitness & insert into population
        if(enOneChild == g_enOffspringSelect)
            {
            // Select which child to keep
            const double dOffspringSelector = pOffspringSelector->NextVariate();
            if(dOffspringSelector < 0.5)
                {
                // Evaluate child fitness & insert into child population
                CFitnessVector FitnessVector = TrainingSetEvaluation2(Child1Chromosome);
                Population.InsertChild(Child1Chromosome, FitnessVector);
                DeleteChromosome(Child2Chromosome);
                }
            else
                {
                // Evaluate child fitness & insert into child population
                CFitnessVector FitnessVector = TrainingSetEvaluation2(Child2Chromosome);
                Population.InsertChild(Child2Chromosome, FitnessVector);
                DeleteChromosome(Child1Chromosome);
                }

            }
        else
            {
            // Add both children to population
            CFitnessVector FitnessVector1 = TrainingSetEvaluation2(Child1Chromosome);
            CFitnessVector FitnessVector2 = TrainingSetEvaluation2(Child2Chromosome);
            Population.AppendChildren(Child1Chromosome, FitnessVector1, Child2Chromosome, FitnessVector2);
            }


        // Sort the new population
        Population.MOSort(enASCENDING);

        }
    while(g_uNoTreeEvaluations < g_uMaxNoTreeEvaluations);


    // End of genetic evolution loop
    //-------------------------------------------------------

    Population.MOSort(enASCENDING);
    // Print final population
    cout << "Final resorted population..." << endl;
    for(uint32_t i = 1; i <= g_uPopulationSize; i++)
        {
        cout << i
             << " -> ("
             << Population.Fitness(i)[1]
             << ", "
             << Population.Fitness(i)[2]
             << ")   rank = "
             << Population.Rank(i)
             << endl;
        }
    cout << endl;

    //Write population to file
    for(uint32_t i = 1; i <= g_uPopulationSize; i++)
        {
        // Put Number of Nodes, Margin^2, Training Error, Test Error into file
        GPResults << Population.Fitness(i)[1] << "   " << Population.Fitness(i)[2] << "   " << TrainingSetError(Population[i]) << "   " << TestSetEvaluation(Population[i]) << endl;
        }

    for(uint32_t i = 1; i <= g_uPopulationSize; i++)
        {

        //Output Tree Information
        GPTrees << i
                << "   Node count = "
                << Population.Fitness(i)[1]
                << ", Margin = "
                << Population.Fitness(i)[2]
                << ",  Rank = "
                << Population.Rank(i)
                << endl;
        cout << endl;


        //Output Tree
        char szStringForTree[4096];
        strcpy(szStringForTree, "");
        TreeToString(Population[i], szStringForTree, enTreeToStringMaths);
        GPTrees << szStringForTree << endl;
        cout << endl;
        }


    //-------------------------------------------------------------------------

    cout << "No of tree evaluations = " << g_uNoTreeEvaluations << endl;

    //-------------------------------------------------------------------------


    cout << endl;

    // Tidy-up
    delete pMutationSelector;
    GPResults.close();
    GPTrees.close();
    return EXIT_SUCCESS;

    } // main()

//*****************************************************************************












