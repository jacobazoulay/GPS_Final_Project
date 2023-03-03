import os

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
    n = np.sqrt(MU / a**3) + ephem['deltaN']
    t_k = tx_time - ephem['t_oe']
    M_k = ephem['M_0'] + (n * t_k)

    e = ephem['e']
    E_k = solveNR(M_k, e)

    sn_nu = (np.sqrt(1 - e ** 2) * np.sin(E_k)) / (1 - e*np.cos(E_k))
    cs_nu = (np.cos(E_k) - e) / (1 - e*np.cos(E_k))
    nu_k = np.arctan2(sn_nu, cs_nu)

    phi_k = nu_k + ephem['omega']

    d_phi_k = ephem['C_us'] * np.sin(2*phi_k) + ephem['C_uc'] * np.cos(2*phi_k)

    u_k = phi_k + d_phi_k

    d_rk = ephem['C_rs'] * np.sin(2*phi_k) + ephem['C_rc'] * np.cos(2*phi_k)

    d_ik = ephem['C_is'] * np.sin(2*phi_k) + ephem['C_ic'] * np.cos(2*phi_k)

    Om_k = ephem['Omega_0'] - OMDOT_E * tx_time + ephem['OmegaDot'] * t_k

    r_k = a * (1 - e*np.cos(E_k)) + d_rk

    i_k = ephem['i_0'] + ephem['IDOT'] * t_k + d_ik

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

    dt_r = -4.442807633 * 10**-10 * ephem['e']**ephem['sqrtA'] * np.sin(E_k)

    d_tsv = af0 + af1 * (tx_time - t0c) + af2 * (tx_time - t0c) ** 2 + dt_r - tgd

    B = d_tsv * 2.99792458*10**8

    return B


def toDict(ephem):
    dict = ephem.to_dict('list')
    for key in dict.keys():
        dict[key] = dict[key][0]
    return dict


def calcPseudo(gnss):
    gnss["week_num"] = np.floor(-gnss["FullBiasNanos"]/604800e9)
    gnss["t_rx_corrected"] = gnss["TimeNanos"] + gnss["TimeOffsetNanos"]
    gnss["adj_bias"] = gnss["FullBiasNanos"] + gnss["BiasNanos"]
    gnss["rx_time_in_gps"] = gnss["t_rx_corrected"] - gnss["adj_bias"]
    gnss["rx_time_cur_wk"] = gnss["rx_time_in_gps"] - gnss["week_num"]*604800e9
    gnss["pseudo_nano"] = gnss["rx_time_cur_wk"] - gnss["ReceivedSvTimeNanos"]
    gnss["Pseudo_m"] = gnss["pseudo_nano"] * 299792458/1e9
    return gnss


def getSatXYZB(ephem, gnss):
    """
    Function will take in ephemeris data and gnss log data to output the
    satellite XYZ location and clock bias B
    :param ephem: pandas dataframe of corresponding ephemeris data
    :param gnss: pandas dataframe of raw gnss logger data
    :return: pandas dataframe consisting of ["RxTime_s", "Pseudo_m", "X", "Y", "Z", "B"] columns
    """
    gnss = gnss[['Svid', 'ReceivedSvTimeNanos', 'Pseudo_m']].to_numpy()

    ECEFs = []
    Bs = []
    for i in range(len(gnss)):
        Svid = gnss[i][0]
        data = ephem[ephem['sv'] == Svid]
        data = toDict(data)
        tx_time = gnss[i][1] / 10 ** 9
        ECEF, E_k = get_sat_ECEF(data, tx_time)
        ECEFs.append(ECEF)

        B = get_B(data, tx_time, E_k)
        Bs.append(B)

    ECEFs = np.array(ECEFs)
    Bs = np.array(Bs)
    out = np.hstack((gnss[:, 1:], ECEFs, Bs[:, np.newaxis]))
    out = pd.DataFrame(out)
    out.columns = ["RxTime_s", "Pseudo_m", "X", "Y", "Z", "B"]

    return out


def parseFile(filepath, transittype="n/a"):
    file1 = open(filepath, 'r')

    data = []
    header = ""
    for line in file1:
        if line[:5] == "# Raw":
            header = line[2:-1].split(",") + ["TransitType"]
        if line[:3] == "Raw":
            curRow = line[0:-1].split(",") + [transittype]
            data.append(curRow)

    df = pd.DataFrame(data)
    df.columns = header

    return df


def main():
    print(os.getcwd())
    ephem = pd.read_csv("ephem.csv")
    gnss = pd.read_csv("gnss_log.csv")
    needed_cols = ['Svid', 'ReceivedSvTimeNanos', "FullBiasNanos", "TimeNanos", "TimeOffsetNanos", "BiasNanos",
                   "ConstellationType"]
    gnss = gnss[gnss.columns.intersection(needed_cols)]
    gnss = gnss.apply(pd.to_numeric)

    gnss = calcPseudo(gnss)
    out = getSatXYZB(ephem, gnss)
    print(out)

    # out.to_csv("test_sat_out.csv", index=False)


if __name__ == '__main__':
    main()
