import numpy as np
import ValidateActions


def main():
    # Build training database
    boxing_list = ValidateActions.getImgList(ValidateActions.BOXING_PATH)
    handclapping_list = ValidateActions.getImgList(ValidateActions.CLAPPING_PATH)
    handwaving_list = ValidateActions.getImgList(ValidateActions.WAVING_PATH)
    running_list = ValidateActions.getImgList(ValidateActions.RUNNING_PATH)
    walking_list = ValidateActions.getImgList(ValidateActions.WALKING_PATH)

    dbT, dbW, dbH = ValidateActions.trainAll((boxing_list, handclapping_list, handwaving_list, running_list, walking_list))

    np.save("Data/dbEigenvectorsTime.npy", dbT)
    np.save("Data/dbEigenvectorsWidth.npy", dbW)
    np.save("Data/dbEigenvectorsHeight.npy", dbH)


if __name__ == '__main__':
    main()
