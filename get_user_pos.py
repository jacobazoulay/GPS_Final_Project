import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def getExpectedPseudoRanges(x_est, x_sat, B, b_u):
    norm = np.linalg.norm(x_sat - x_est, axis=1)
    pseudo_exp = norm - B + b_u
    return pseudo_exp


def getGMatrix(x_est, x_sat):
    G = np.ones((x_sat.shape[0], 4))
    norm = np.linalg.norm(x_sat - x_est, axis=1)
    norm = np.array([norm, norm, norm]).T
    G[:, 0:3] = -(x_sat - x_est) / norm
    return G


def solve_pos(x_sat, B, pseudo_meas):
    b_u = 0
    delta_x_b_u = np.ones(4)
    x_est = np.zeros(3)
    count = 0
    max_count = 300
    while np.linalg.norm(delta_x_b_u[:3]) > 0.01:
        G = getGMatrix(x_est, x_sat)
        pseudo_exp = getExpectedPseudoRanges(x_est, x_sat, B, b_u)
        delta_p = pseudo_meas - pseudo_exp
        GTGmat = np.matmul(G.T, G)
        if np.linalg.det(GTGmat) == 0:
            return np.array([np.nan, np.nan, np.nan])
        delta_x_b_u = np.matmul(np.matmul(np.linalg.inv(GTGmat), G.T),  delta_p)

        x_est += delta_x_b_u[:-1]
        b_u += delta_x_b_u[-1]

        # Sometimes our accuracy requirement is too high
        if count >= max_count:
            if np.linalg.norm(delta_x_b_u[:3]) > 10:
                return np.array([np.nan, np.nan, np.nan])
            else:
                return x_est
        count += 1

    return x_est


def getTimeIdxs(data):
    idxs = [0]
    for i in range(1, data.shape[0]):
        if data[i - 1][0] != data[i][0]:
            idxs.append(i)
    idxs.append(data.shape[0])

    return idxs


def solveAll(data):
    idxs = getTimeIdxs(data)

    x_ests = []
    for i in range(len(idxs) - 1):
        pseudo_meas = data[idxs[i]:idxs[i + 1], 1]
        x_sat = data[idxs[i]:idxs[i + 1], 2:5]
        B = data[idxs[i]:idxs[i + 1], 5]

        x_est = solve_pos(x_sat, B, pseudo_meas)
        x_ests.append(np.concatenate((x_est, [data[idxs[i], 0]])))

    return np.array(x_ests)


def plotXYZ(x_ests):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_ests[:, 0], x_ests[:, 1], x_ests[:, 2])

    ax.set_title("GPS Positions")
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    plt.show()


def getUserXYZ(data):
    """
    Function that takes in GPS data to give user position
    :param data: pd data frame that has ["RxTime_s", "Pseudo_m", "X", "Y", "Z", "B"] columns
    :return: pd data frame of ["X_u", "Y_u", "Z_u", "RxTime_s"]
    """
    data = data[["RxTime_s", "Pseudo_m", "X", "Y", "Z", "B"]]
    data = data.to_numpy()

    x_ests = solveAll(data)

    x_ests = pd.DataFrame(x_ests)
    x_ests.columns = ["X_u", "Y_u", "Z_u", "RxTime_s"]

    return x_ests


def main():
    df = pd.read_csv("data/Test Data/gnss_log.csv")
    df = pd.read_csv("get_sat_data_test/test_sat_out.csv")
    print(df[["X", "Y", "Z", "B"]])
    x_ests = getUserXYZ(df)
    print(x_ests)
    x_ests.to_csv("test_out.csv")
    plotXYZ(x_ests)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()