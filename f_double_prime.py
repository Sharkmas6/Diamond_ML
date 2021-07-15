import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit


if __name__ == "__main__":
    a = pd.Series({0.7749:2.5787, 0.8266:2.8647, 0.8856:3.2005, 0.9050:3.3128, 0.9158:3.3758, 0.9184:3.3910,
         0.9483:3.5910, 0.9537:3.6339, 0.9700:3.7652, 0.9724:3.7851, 0.9759:3.8143, 0.9762:3.8168, 0.9778:3.8302,
         0.9787:3.8378, 0.9789:3.8395, 0.9790:3.8404, 0.9791:3.8412, 0.9792:3.8421, 0.9793:3.8429, 0.9794:3.8438,
         0.9795:0.4992, 0.9796:0.4993, 0.9798:0.4995, 0.9801:0.4997, 0.9824:0.5019, 0.9840:0.5034, 1.0032:0.5216,
         1.0359:0.5532, 1.0721:0.5891, 1.2716:0.8048, 1.5498:1.1494, 1.6531:1.2890, 1.9075:1.6591}, name="F''")

    a_df = pd.DataFrame(a)
    a_high, a_low = a_df[a > 2.5], a_df[a < 2]

    # fit linear model
    model_low, model_high = [LinearRegression(), LinearRegression()]
    model_low.fit(a_low.index.values.reshape((-1, 1)), a_low.values)
    model_high.fit(a_high.index.values.reshape((-1, 1)), a_high.values)

    # fit to range
    x_low = np.linspace(a_low.index.min(), a_low.index.max()).reshape((-1, 1))
    x_high = np.linspace(a_high.index.min(), a_high.index.max()).reshape((-1, 1))
    print(x_low[[0, -1], 0], x_high[[0, -1], 0])
    linear_low = model_low.predict(x_low)
    linear_high = model_high.predict(x_high)


    # fit power models
    def power1(x, a, b, c):
         return a ** (x - c) + b

    def power2(x, a, b, c):
         return (x - c) ** a + b


    p1l_params, p1l_err = curve_fit(power1, a_low.index.values, a_low.values.flatten())
    p1_low = power1(x_low, *p1l_params)
    p1h_params, p1h_err = curve_fit(power1, a_high.index.values, a_high.values.flatten())
    p1_high = power1(x_high, *p1h_params)

    p2l_params, p2l_err = curve_fit(power2, a_low.index.values, a_low.values.flatten(), bounds=([.1, -10, -10], [10, 10, 10]))
    p2_low = power2(x_low, *p2l_params)
    p2h_params, p2h_err = curve_fit(power2, a_high.index.values, a_high.values.flatten(), bounds=([.1, -10, -10], [10, 10, 10]))
    p2_high = power2(x_high, *p2h_params)

    #plot
    mpl.style.use("ggplot")
    fig, (ax_high, ax_low) = plt.subplots(1, 2, figsize=(12,8))

    ax_high.plot(a_high.index, a_high, label="Original High")
    ax_low.plot(a_low.index, a_low, label="Original Low")

    ax_low.plot(x_low, linear_low, linestyle="dashed", label="Linear Low")
    ax_high.plot(x_high, linear_high, linestyle="dashed", label="Linear High")

    ax_low.plot(x_low, p1_low, linestyle="dashed", label="P1 (a^x) Low")
    ax_high.plot(x_high, p1_high, linestyle="dashed", label="P1 (a^x) High")

    ax_low.plot(x_low, p2_low, linestyle="dashdot", label="P2 (x^a) Low")
    ax_high.plot(x_high, p2_high, linestyle="dashdot", label="P2 (x^a) High")

    ax_low.set_xlabel("Wavelength")
    ax_low.set_ylabel("f''")
    ax_low.legend()
    ax_high.set_xlabel("Wavelength")
    ax_high.set_ylabel("f''")
    ax_high.legend()

    # sum of square residuals
    p1l_ssr = np.sum((power1(a_low.index.values, *p1l_params) - a_low.values.flatten()) ** 2)
    p1h_ssr = np.sum((power1(a_high.index.values, *p1h_params) - a_high.values.flatten()) ** 2)
    p2l_ssr = np.sum((power2(a_low.index.values, *p2l_params) - a_low.values.flatten()) ** 2)
    p2h_ssr = np.sum((power2(a_high.index.values, *p2h_params) - a_high.values.flatten()) ** 2)

    print(f"LOW:\n"
          f"P1 (a^x): {np.diag(p1l_err)} -> {p1l_ssr}\n"
          f"P2 (x^a): {np.diag(p2l_err)} -> {p2l_ssr}\n"
          f"HIGH:\n"
          f"P1 (a^x): {np.diag(p1h_err)} -> {p1h_ssr}\n"
          f"P2 (x^a): {np.diag(p2h_err)} -> {p2h_ssr}\n")

    ''' Equal performance, use x^a '''


def FDoublePrime(wavelength, threshold=0.9795):
    mask_low = wavelength >= threshold
    mask_high = wavelength < threshold
    result = wavelength.copy()

    a_low, b_low, c_low = 1.37919369, 0.2780599, 0.64429346
    a_high, b_high, c_high = 3.92791177, 1.34277082, -0.28303464

    result[mask_low] = (result[mask_low] - c_low) ** a_low + b_low
    result[mask_high] = (result[mask_high] - c_high) ** a_high + b_high

    return result
