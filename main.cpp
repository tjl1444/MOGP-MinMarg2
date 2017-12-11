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

//-----------------------------------------------------------------------------
// Globals

CHROMOSOME g_pstRootNode;
double* g_adY;
CMatrix* g_pDistanceMatrix;

//-----------------------------------------------------------------------------
// Global constants

const int32_t g_nNoConstraints = 2;
double g_dRhoBegin = 0.1;
double g_dRhoEnd = 1e-6;
int32_t g_nMessageLevel = 1;
int32_t g_nFunctionEvaluations = 3500;

const uint32_t g_uNoInitialisationRepeats = 1;
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

const uint32_t g_uRandomTreeNodeMax = 63;  // This is normally 63
const uint32_t g_uMaxInitialDepth = 6;     // This is normally 6
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

    //minimise ||Xd-Xs||^2 subject to g(padX) = 0
    (*padFnValue) = 0.0;
    for(int32_t i = 0; i < nNoDimensions; i++)
        {
        *padFnValue += ((padX[i] - g_adY[i]) * (padX[i] - g_adY[i]));

        //Set up Column Vector to use for calculating constraints with g(X)
        pdX[i + 1] = padX[i];
        }

    // Calculate constraints
    padConstraints[0] = g(pdX, g_pstRootNode);
    padConstraints[1] = -g(pdX, g_pstRootNode);

    return 0;
    }

//-----------------------------------------------------------------------------

bool isMisclassified(CHROMOSOME pstRootNode, CColumnVector& PatternVector, const uint16_t suLabel)
// Returns true if the point is misclassified
    {
    const double y = TreeEvaluate(PatternVector, pstRootNode);

    if((y < 0) and (suLabel != 0))
        {
        //Misclassified pattern vector
        return true;
        }
    else if((y >= 0) and (suLabel != 1))
        {
        //Misclassified pattern vector
        return true;
        }
    return false;
    }

//-----------------------------------------------------------------------------

double CalculateMargin(CColumnVector& InitialVector, CColumnVector& PatternVector)
// Returns square of the Margin for the PatternVector which is a misclassified point
    {
    //Initialize COBYLA parameters
    const int32_t nNoDimensions = g_TrainingSet.VectorLength();

    // DEBUG
    cout << endl;
    cout << "initial vector is:" << InitialVector[1] << ":  " << InitialVector[2] << endl;
    cout << "pattern vector is:" << PatternVector[1] << ":  " << PatternVector[2] << endl;
//    cout << "0/1 loss = " << CalculateTrainingSetError(g_pstRootNode) << endl;

    double* x = new double[nNoDimensions];
    g_adY = new double[nNoDimensions];

    //Set up initial estimate of solution
    for(uint32_t i = 0; i < (static_cast<uint32_t>(nNoDimensions)); i++)
        {
        x[i] = InitialVector[i + 1];
        g_adY[i] = PatternVector[i + 1];
        }

    int32_t nReturnValue = COBYLA(nNoDimensions, g_nNoConstraints, x, g_dRhoBegin, g_dRhoEnd, g_nMessageLevel, &g_nFunctionEvaluations, COBYLA_Function, NULL);

    if(nReturnValue != 0)
        {
        cout << "Cobyla Error" << endl;
        //ErrorHandler("COBYLA returned error code");
        }


    //Calculate margin
    double margin2 = 0.0;
    for(uint32_t i = 0; i < (static_cast<uint32_t>(nNoDimensions)); i++)
        {
        margin2 += ((g_adY[i] - x[i]) * (g_adY[i] - x[i]));
        }

    delete[] x;
    return margin2;
    } // CalculateMargin()

//-----------------------------------------------------------------------------

double GetLargestMargin(const CHROMOSOME pstRootNode)
    {
    double dLargestMargin = 0.0;

    const uint32_t uNoData = g_TrainingSet.NoStoredPatterns();
    g_pstRootNode = pstRootNode;


    // Find misclassified points
    bool abCorrectlyClassified[uNoData];

    for(uint32_t i = 1; i <= uNoData; i++)
        {
        CColumnVector x = g_TrainingSet[i];
        const uint16_t suLabel = g_TrainingSet.Tag(i);

        abCorrectlyClassified[i - 1] = !(isMisclassified(pstRootNode, x, suLabel));

        }

    //Find margin for misclassified points
    for(uint32_t i = 1; i <= uNoData; i++)
        {

        uint32_t uNearestNeighbourIndex = UINT_MAX;
        if(abCorrectlyClassified[i - 1] == false)
            {
            //Find closest correctly classified point across the boundary
            double dNearestNeighbourDistance = INFINITY;
            for(uint32_t j = 1; j <= uNoData; j++)
                {
                double dDistance = (*g_pDistanceMatrix)[i][j];
                if((abCorrectlyClassified[j - 1] == true) and (i != j) and (g_TrainingSet.Tag(i) == g_TrainingSet.Tag(j)) and (dDistance < dNearestNeighbourDistance))
                    {
                    dNearestNeighbourDistance = dDistance;
                    uNearestNeighbourIndex = j;
                    }
                }

            }

//        if(uNearestNeighbourIndex == UINT_MAX){
//            ErrorHandler("No misclassified  points in the data set");
//            return dLargestMargin;
//
//        }

        // Calculate initial estimate of boundary vector
        if((i != uNearestNeighbourIndex) and (uNearestNeighbourIndex != UINT_MAX))
            {
            double dFuncMin;
            CColumnVector TargetPoint = g_TrainingSet[i];


            CColumnVector NearestPoint = g_TrainingSet[uNearestNeighbourIndex];
            const double dAlpha = GoldenSectionLineSearch(TargetPoint, NearestPoint, g2, static_cast<void*>(g_pstRootNode), dFuncMin);
            CColumnVector InitialVector = (TargetPoint * (1.0 - dAlpha)) + (dAlpha * NearestPoint);    // Initial point on decision surface

            double margin = CalculateMargin(InitialVector, TargetPoint);
            if(margin > dLargestMargin)
                {
                dLargestMargin = margin;
                }



            }

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
// Returns fitness vector
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

CFitnessVector TrainingSetEvaluation2(const CHROMOSOME pstRootNode)
// Returns fitness vector
    {
    g_uNoTreeEvaluations++;
    CFitnessVector FitnessVector;

    //Calculate number of nodes in tree
    FitnessVector[1] = NoTreeNodes(pstRootNode, true);

    // Calculate smallest margin for that individual
    FitnessVector[2] = GetLargestMargin(pstRootNode);

    return FitnessVector;
    } // TrainingSetEvaluation()



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

stGP_Parameters_t g_stGP_Parameters(g_lnGP_BaseSeed, g_uNoInitialisationRepeats);

//*****************************************************************************

int main()
    {
    // Load training and test datasets

    //Ripley2TrainingSet.dat  2D-GaussianTraining.dat SineWaveTrainingSet.dat
    g_TrainingSet.Load("2D-GaussianTraining.dat");
    g_TestSet.Load("2D-GaussianTest.dat");

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
            Population.Fitness(j) = TrainingSetEvaluation(Population[j]);
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

            Population[j] = pTree;
            Population.Fitness(j) = TrainingSetEvaluation(Population[j]);
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
             << ", Mean squared error = "
             << Population.Fitness(i)[2]
             << ",  Rank = "
             << Population.Rank(i)
             << endl;
        }
    cout << endl;

    //-----------------------------------------------
    // Start of genetic evolution loop
    double dLargestTrainError;
    uint32_t uLargestTrainIndex;

    uint32_t uMinTrainIndex;
    double dMinTrainError;

    uint32_t uNoIterations = 0;
    g_uNoTreeEvaluations = 0;
    cout << "Entering evolutionary loop..." << endl;

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
                CFitnessVector FitnessVector = TrainingSetEvaluation(Child1Chromosome);
                Population.InsertChild(Child1Chromosome, FitnessVector);
                DeleteChromosome(Child2Chromosome);
                }
            else
                {
                // Evaluate child fitness & insert into child population
                CFitnessVector FitnessVector = TrainingSetEvaluation(Child2Chromosome);
                Population.InsertChild(Child2Chromosome, FitnessVector);
                DeleteChromosome(Child1Chromosome);
                }
            }
        else
            {
            // Add both children to population
            CFitnessVector FitnessVector1 = TrainingSetEvaluation(Child1Chromosome);
            CFitnessVector FitnessVector2 = TrainingSetEvaluation(Child1Chromosome);
            Population.AppendChildren(Child1Chromosome, FitnessVector1, Child2Chromosome, FitnessVector2);
            }

        // Sort the new population
        Population.MOSort(enASCENDING);

        // Find smallest training set error
        dMinTrainError = INFINITY;
        uMinTrainIndex = 1;

        dLargestTrainError = 0.0;
        uLargestTrainIndex = 1;

        for(uint32_t i = 1; i <= g_uPopulationSize; i++)
            {
            if(Population.Fitness(i)[2] < dMinTrainError)
                {
                dMinTrainError = Population.Fitness(i)[2];
                uMinTrainIndex = i;
                }
            if(Population.Fitness(i)[2] > dLargestTrainError)
                {
                dLargestTrainError = Population.Fitness(i)[2];
                uLargestTrainIndex = i;
                }

            }

        }
    while((g_uNoTreeEvaluations < g_uMaxNoTreeEvaluations) or (dLargestTrainError >= 0.5));

    cout << "The worst individual in the population has error of: " << dLargestTrainError << endl;


    //---------------------------------------------------------------------------------------------
    // Run evolutionary loop using margin to evaluate training set
    // FitnessVector[2] is now largest margin


    uNoIterations = 0;
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
            CFitnessVector FitnessVector2 = TrainingSetEvaluation2(Child1Chromosome);
            Population.AppendChildren(Child1Chromosome, FitnessVector1, Child2Chromosome, FitnessVector2);
            }

        // Sort the new population
        Population.MOSort(enASCENDING);

        // Find smallest of the large margins
        dMinLargestMargin = INFINITY;
        uMinLargestMarginIndex = 1;

        for(uint32_t i = 1; i <= g_uPopulationSize; i++)
            {
            if(Population.Fitness(i)[2] < dMinLargestMargin)
                {
                dMinLargestMargin = Population.Fitness(i)[2];
                uMinLargestMarginIndex = i;
                }

            }

        }
    while(g_uNoTreeEvaluations < g_uMaxNoTreeEvaluations);



    // End of genetic evolution loop
    //-------------------------------------------------------

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

    //-------------------------------------------------------------------------

    cout << "No of tree evaluations = " << g_uNoTreeEvaluations << endl;

    //-------------------------------------------------------------------------

    // Best training individual
    stGPNode_t* pBestTrainedIndividual = Population[uMinTrainIndex];
    cout << "Smallest training error = " << dMinTrainError << " with ";
    cout << NoTreeNodes(pBestTrainedIndividual, true) << " nodes & depth = ";
    cout << MaxTreeDepth(pBestTrainedIndividual);
    cout << " with test error = " << TestSetEvaluation(Population[uMinTrainIndex]);
    cout << endl;

    // Get best test error over population
    uint32_t uMinTestIndex = UINT32_MAX;
    double dMinTestError = INFINITY;
    for(uint32_t i = 1; i <= g_uPopulationSize; i++)
        {
        if(TestSetEvaluation(Population[i]) < dMinTestError)
            {
            dMinTestError = TestSetEvaluation(Population[i]);
            uMinTestIndex = i;
            }
        }

    stGPNode_t* pBestTestIndividual = Population[uMinTestIndex];
    cout << "Smallest test error = " << dMinTestError << " with ";
    cout << NoTreeNodes(pBestTestIndividual, true) << " nodes & depth = ";
    cout << MaxTreeDepth(pBestTestIndividual);
    cout << endl;

    cout << "Index of best test individual = " << uMinTestIndex << endl;
    cout << "Best test individual has a margin = " << GetLargestMargin(Population[uMinTestIndex]);

    // Output best test
    stGPNode_t* pBestTree = Population[uMinTestIndex];
    char szStringForTree[4096];
    strcpy(szStringForTree, "");
    TreeToString(pBestTree, szStringForTree, enTreeToStringMaths);
    cout << szStringForTree << endl;

    cout << endl;

    cout << "Index of (minimum) largest margin individual = " << uMinLargestMarginIndex << endl;
    cout << "It has a training error of = " << TrainingSetEvaluation(Population[uMinLargestMarginIndex])[2] << endl;
    cout << "It has a test error of = " << TestSetEvaluation(Population[uMinLargestMarginIndex]) << endl;
    cout << "It has a margin of = " << GetLargestMargin(Population[uMinLargestMarginIndex]);

    // Output minimum largest margin tree
    stGPNode_t* pLargestMargin = Population[uMinLargestMarginIndex];
    char szStringForTree1[4096];
    strcpy(szStringForTree1, "");
    TreeToString(pLargestMargin, szStringForTree1, enTreeToStringMaths);
    cout << szStringForTree1 << endl;

    // Tidy-up
    delete pMutationSelector;

    return EXIT_SUCCESS;
    } // main()

//*****************************************************************************













