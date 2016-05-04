from LoadActionDatabaseAndTestProbe import *
from random import shuffle

ACTION_LABELS = ['Boxing', 'Clapping', 'Waving', 'Running', 'Walking']
LABEL_COUNT = len(ACTION_LABELS)

BOXING_PATH = r'D:\Working\Datasets\KTH\training\boxing'
CLAPPING_PATH = r'D:\Working\Datasets\KTH\training\handclapping'
WAVING_PATH = r'D:\Working\Datasets\KTH\training\handwaving'
RUNNING_PATH = r'D:\Working\Datasets\KTH\training\running'
WALKING_PATH = r'D:\Working\Datasets\KTH\training\walking'


def getImgList(path):
    imgs = os.listdir(path)
    imgs = [os.path.join(path, x) for x in imgs]
    return imgs


def shuffleAndSplit(mylist):
    shuffle(mylist)
    return mylist[:len(mylist)/2], mylist[len(mylist)/2:]


def crossValidate(dim_to_keep):
    # Two fold cross validation
    boxing_list = getImgList(BOXING_PATH)
    handclapping_list = getImgList(CLAPPING_PATH)
    handwaving_list = getImgList(WAVING_PATH)
    running_list = getImgList(RUNNING_PATH)
    walking_list = getImgList(WALKING_PATH)

    boxing_1, boxing_2 = shuffleAndSplit(boxing_list)
    handclapping_1, handclapping_2 = shuffleAndSplit(handclapping_list)
    handwaving_1, handwaving_2 = shuffleAndSplit(handwaving_list)
    running_1, running_2 = shuffleAndSplit(running_list)
    walking_1, walking_2 = shuffleAndSplit(walking_list)

    # PART I
    c1 = trainAndTest((boxing_1, handclapping_1, handwaving_1, running_1, walking_1),
                      (boxing_2, handclapping_2, handwaving_2, running_2, walking_2),
                      dim_to_keep)

    # PART II
    c2 = trainAndTest((boxing_2, handclapping_2, handwaving_2, running_2, walking_2),
                      (boxing_1, handclapping_1, handwaving_1, running_1, walking_1),
                      dim_to_keep)

    # Add c1 to c2
    confusion = [[0 for x in range(LABEL_COUNT)] for y in range(LABEL_COUNT)]  # Make LABEL_COUNT x LABEL_COUNT matrix
    for truth in range(LABEL_COUNT):
        for predicted in range(LABEL_COUNT):
            confusion[truth][predicted] = c1[truth][predicted] + c2[truth][predicted]

    return confusion


def trainAndTest(train_lists, test_lists, dim_to_keep):
    # Databases of eigenvectors
    dbT, dbW, dbH = trainAll(train_lists)

    # Build confusion matrix
    confusion = testAll(dbT, dbW, dbH, test_lists, dim_to_keep)

    return confusion


def trainAll(train_lists):
    # Databases of eigenvectors
    dbT = []
    dbW = []
    dbH = []

    for file_list in train_lists:
        # Get list of eigenvectors for given label
        eT, eW, eH = train(file_list)

        # Append to database
        dbT.append(eT)
        dbW.append(eW)
        dbH.append(eH)

    return dbT, dbW, dbH


def testAll(dbT, dbW, dbH, test_lists, dim_to_keep):
    # Build confusion matrix
    confusion = []

    for label, file_list in enumerate(test_lists):
        c = test(dbT, dbW, dbH, file_list, dim_to_keep)
        confusion.append(c)

    return confusion


def train(file_list):
    eigenvectorsTime = []
    eigenvectorsWidth = []
    eigenvectorsHeight = []

    for path in file_list:
        # Load training video
        cube = loadCube(path)

        vectorsTime, vectorsWidth, vectorsHeight = getEigenVectors(cube)

        eigenvectorsTime.append(vectorsTime)
        eigenvectorsWidth.append(vectorsWidth)
        eigenvectorsHeight.append(vectorsHeight)

    eigenvectorsTime = np.array(eigenvectorsTime)
    eigenvectorsWidth = np.array(eigenvectorsWidth)
    eigenvectorsHeight = np.array(eigenvectorsHeight)

    return eigenvectorsTime, eigenvectorsWidth, eigenvectorsHeight


def test(databaseT, databaseW, databaseH, file_list, dim_to_keep):
    confusion = [0 for x in range(LABEL_COUNT)]  # Number of labels

    for path in file_list:
        # Load testing video
        testCube = loadCube(path)

        # Get test eigenvectors
        teT, teW, teH = getEigenVectors(testCube)

        scoresTime = getPrincipalAnglesScores(databaseT, teT, dim_to_keep)
        scoresWidth = getPrincipalAnglesScores(databaseW, teW, dim_to_keep)
        scoresHeight = getPrincipalAnglesScores(databaseH, teH, dim_to_keep)

        scores = anglesDistance(scoresTime, scoresWidth, scoresHeight)

        """
        best_list = []
        for label, best_five in enumerate(best):
            for val in best_five:
                best_list.append((label, best_five))
        best_list.sort()
        """

        # Find best score (predicted label)
        maxScore = scores[0]
        predLabel = 0

        for i, score in enumerate(scores):
            if score > maxScore:
                maxScore = score
                predLabel = i

        # Add result to confusion vector
        confusion[predLabel] += 1

    return confusion


def main():
    """
    # Build training database
    boxing_list = getImgList(BOXING_PATH)
    handclapping_list = getImgList(CLAPPING_PATH)
    handwaving_list = getImgList(WAVING_PATH)
    running_list = getImgList(RUNNING_PATH)
    walking_list = getImgList(WALKING_PATH)

    boxing_1, boxing_2 = shuffleAndSplit(boxing_list)
    handclapping_1, handclapping_2 = shuffleAndSplit(handclapping_list)
    handwaving_1, handwaving_2 = shuffleAndSplit(handwaving_list)
    running_1, running_2 = shuffleAndSplit(running_list)
    walking_1, walking_2 = shuffleAndSplit(walking_list)

    dbT1, dbW1, dbH1 = trainAll((boxing_1, handclapping_1, handwaving_1, running_1, walking_1))
    dbT2, dbW2, dbH2 = trainAll((boxing_2, handclapping_2, handwaving_2, running_2, walking_2))

    print 'Dimensions, BoxingF, ClappingF, WavingF, RunningF, WalkingF, AverageF'

    for dimensions in range(1, 31):
        try:
            c1 = testAll(dbT1, dbW1, dbH1, (boxing_2, handclapping_2, handwaving_2, running_2, walking_2), dimensions)
            c2 = testAll(dbT2, dbW2, dbH2, (boxing_1, handclapping_1, handwaving_1, running_1, walking_1), dimensions)

            # Add c1 to c2
            confusion = [[int(c1[truth][predicted] + c2[truth][predicted]) for predicted in range(LABEL_COUNT)] for truth in range(LABEL_COUNT)]

            precision = [confusion[label][label] / float(sum(zip(*confusion)[label])) for label in range(LABEL_COUNT)]
            recall = [confusion[label][label] / float(sum(confusion[label])) for label in range(LABEL_COUNT)]
            f_score = [(2 * precision[label] * recall[label]) / (precision[label] + recall[label])
                   for label in range(LABEL_COUNT)]

            avg_f = sum(f_score) / len(f_score)

            print str([dimensions, f_score[BOXING], f_score[HAND_CLAPPING], f_score[HAND_WAVING],
                       f_score[RUNNING], f_score[WALKING], avg_f]).strip('[]')
        except:
            print 'Error for dimension', dimensions

    """
    confusion = crossValidate(5)

    print ', Boxing, Clapping, Waving, Running, Walking'
    for i, pred in enumerate(confusion):
        print ACTION_LABELS[i] + ', ' + str(pred).strip('[]')
    #"""


if __name__ == '__main__':
    main()
