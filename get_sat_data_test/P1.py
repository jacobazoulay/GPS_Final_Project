import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MU = 3.986005 * 10**14
OMDOT_E = 7.2921151467 * 10**-5


def solveNR(M, e, eps=10**-10):
    delta = np.inf
    E = M
    while delta > eps:
        E_new = E - ((E - e * np.sin(E) - M) / (1 - e * np.cos(E)))
        delta = np.abs(E_new - E)
        E = E_new
    return E


def get_sat_ECEF(ephem, tx_time):
    a = ephem['sqrtA'] ** 2
    n = np.sqrt(MU / a**3) + ephem['DeltaN']
    t_k = tx_time - ephem['Toe']
    M_k = ephem['M0'] + (n * t_k)

    e = ephem['Eccentricity']
    E_k = solveNR(M_k, e)

    sn_nu = (np.sqrt(1 - e ** 2) * np.sin(E_k)) / (1 - e*np.cos(E_k))
    cs_nu = (np.cos(E_k) - e) / (1 - e*np.cos(E_k))
    nu_k = np.arctan2(sn_nu, cs_nu)

    phi_k = nu_k + ephem['omega']

    d_phi_k = ephem['Cus'] * np.sin(2*phi_k) + ephem['Cuc'] * np.cos(2*phi_k)

    u_k = phi_k + d_phi_k

    d_rk = ephem['Crs'] * np.sin(2*phi_k) + ephem['Crc'] * np.cos(2*phi_k)

    d_ik = ephem['Cis'] * np.sin(2*phi_k) + ephem['Cic'] * np.cos(2*phi_k)

    Om_k = ephem['Omega0'] - OMDOT_E * tx_time + ephem['OmegaDot'] * t_k

    r_k = a * (1 - e*np.cos(E_k)) + d_rk

    i_k = ephem['Io'] + ephem['IDOT'] * t_k + d_ik

    x_p = r_k * np.cos(u_k)

    y_p = r_k * np.sin(u_k)

    x_ECEF = x_p * np.cos(Om_k) - y_p * np.cos(i_k) * np.sin(Om_k)
    y_ECEF = x_p * np.sin(Om_k) + y_p * np.cos(i_k) * np.cos(Om_k)
    z_ECEF = y_p * np.sin(i_k)

    return np.array([x_ECEF, y_ECEF, z_ECEF]), E_k


def get_B(ephem, tx_time, E_k):
    af0 = ephem['SVclockBias']
    af1 = ephem['SVclockDrift']
    af2 = ephem['SVclockDriftRate']
    t0c = ephem['TransTime']
    tgd = ephem['TGD']

    dt_r = -4.442807633 * 10**-10 * ephem['Eccentricity']**ephem['sqrtA'] * np.sin(E_k)

    d_tsv = af0 + af1 * (tx_time - t0c) + af2 * (tx_time - t0c) ** 2 + dt_r - tgd

    B = d_tsv * 2.99792458*10**8

    return B


def toDict(ephem):
    dict = ephem.to_dict('list')
    for key in dict.keys():
        dict[key] = dict[key][0]
    return dict


def main():
    ephem = pd.read_csv("../ephem.csv")
    gnss = pd.read_csv("gnss_log.csv")
    gnss = gnss[['Svid', 'ReceivedSvTimeNanos', 'X', 'Y', 'Z', 'B']].to_numpy()
    log_ECEF = gnss[:, 2:]

    ECEFs = []
    Bs = []
    for i in range(240):
        Svid = gnss[i][0]
        data = ephem[ephem['Svid'] == Svid]
        data = toDict(data)
        tx_time = gnss[i][1] / 10 ** 9
        ECEF, E_k = get_sat_ECEF(data, tx_time)
        ECEFs.append(ECEF)

        B = get_B(data, tx_time, E_k)
        Bs.append(B)

    ECEFs = np.array(ECEFs)
    P1b = np.hstack((gnss[0:7, 0:2], ECEFs[0:7]))
    np.savetxt("P1b.csv", P1b, delimiter=",")

    d_x = ECEFs[:, 0] - gnss[:len(ECEFs), 2]
    d_y = ECEFs[:, 1] - gnss[:len(ECEFs), 3]
    d_z = ECEFs[:, 2] - gnss[:len(ECEFs), 4]

    fig, ax = plt.subplots()
    ax.plot(range(len(d_x)), d_x, label='X Error')
    ax.plot(range(len(d_x)), d_y, label='Y Error')
    ax.plot(range(len(d_x)), d_z, label='Z Error')
    ax.set_title("XYZ Calculated Error")
    ax.set_xlabel("Measurement Number")
    ax.set_ylabel("Error (m)")
    ax.legend()
    ax.grid()
    plt.show()

    Bs = np.array(Bs)
    P1c = np.hstack((gnss[0:7, 0:2], Bs[0:7, np.newaxis]))
    print(P1c)
    np.savetxt("P1c.csv", P1c, delimiter=",")

    d_B = Bs - gnss[:len(Bs), 5]
    fig, ax = plt.subplots()
    ax.plot(range(len(d_B)), d_B, label='B Error')
    ax.set_title("Clock Bias Calculated Error")
    ax.set_xlabel("Measurement Number")
    ax.set_ylabel("Error (m)")
    ax.legend()
    ax.grid()
    plt.show()


if __name__ == '__main__':
    main()
