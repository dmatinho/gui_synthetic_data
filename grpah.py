import pandas as pd
import numpy as np
import os.path
import tkinter
import seaborn as sns
import matplotlib.pyplot as plt


from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

def main(generated_data=None):
    # path real data
    os.path.dirname(__file__)
    fullpath = os.path.join(os.path.dirname(__file__), 'dataSample_2012_update.csv')
    print(__file__)


    data = pd.read_csv(fullpath)
    data.sample(2)

    print(data[:2])


    # path fake data
    if generated_data is None:
        os.path.dirname(__file__)
        fullpath1 = os.path.join(os.path.dirname(__file__), 'Fake_trip_data500log_2012.csv')
        fakedata = pd.read_csv(fullpath1)
    else:
        fakedata = generated_data
        fakedata.sample(2)


    # columns to show
    sample_corr = ['UNITS_carbbev', 'UNITS_yogurt', 'UNITS_milk', 'UNITS_saltsnck', 'UNITS_soup',
                   'DOLLARS__carbbev', 'DOLLARS__yogurt', 'DOLLARS__milk', 'DOLLARS__saltsnck', 'DOLLARS__soup']

    # define units, dollars, demographic
    units_colum = ['UNITS_beer', 'UNITS_blades', 'UNITS_carbbev', 'UNITS_cigets', 'UNITS_coffee', 'UNITS_coldcer',
                   'UNITS_deod', 'UNITS_diapers', 'UNITS_factiss', 'UNITS_fzdinent', 'UNITS_fzpizza',
                   'UNITS_hhclean', 'UNITS_hotdog', 'UNITS_laundet', 'UNITS_margbutr', 'UNITS_mayo', 'UNITS_milk',
                   'UNITS_mustketc', 'UNITS_paptowl', 'UNITS_peanbutr', 'UNITS_photo', 'UNITS_razor', 'UNITS_saltsnck',
                   'UNITS_shamp', 'UNITS_soup', 'UNITS_spagsauc', 'UNITS_sugarsub', 'UNITS_toitisu', 'UNITS_toothbr',
                   'UNITS_yogurt']

    dollars_colum = ['DOLLARS__beer', 'DOLLARS__blades', 'DOLLARS__carbbev', 'DOLLARS__cigets', 'DOLLARS__coffee',
                     'DOLLARS__coldcer', 'DOLLARS__deod', 'DOLLARS__diapers', 'DOLLARS__factiss',
                     'DOLLARS__fzdinent', 'DOLLARS__fzpizza', 'DOLLARS__hhclean', 'DOLLARS__hotdog', 'DOLLARS__laundet',
                     'DOLLARS__margbutr', 'DOLLARS__mayo', 'DOLLARS__milk', 'DOLLARS__mustketc', 'DOLLARS__paptowl',
                     'DOLLARS__peanbutr', 'DOLLARS__photo', 'DOLLARS__razor', 'DOLLARS__saltsnck', 'DOLLARS__shamp',
                     'DOLLARS__soup', 'DOLLARS__spagsauc', 'DOLLARS__sugarsub', 'DOLLARS__toitisu', 'DOLLARS__toothbr',
                     'DOLLARS__yogurt']

    demo_col = ['OUTLET', 'Combined Pre-Tax Income of HH', 'Family Size', 'Type of Residential Possession',
                'Age Group Applied to Household Head', 'Education Level Reached by Household Head',
                'Occupation Code of Household Head', 'Children Group Code', 'Marital Status', 'FIPSCODE']

    continues_col = units_colum + dollars_colum
    continus_column = units_colum + dollars_colum

    ### gui

    ####


    root=tkinter.Tk()
    root.wm_title("Validation")

    # fig=Figure(figsize=(5,5), dpi=100)
    ## correlation
    fig, ax = plt.subplots(figsize=(17, 9))

    # ## mean difference
    #

    ###
    plt.subplot(3, 2, 3)
    sns.barplot(x='Combined Pre-Tax Income of HH', y='DOLLARS__beer', data=data.groupby('Combined Pre-Tax Income of HH')[dollars_colum].mean().reset_index(),
                color=sns.light_palette("navy")[3], alpha=0.7)
    sns.barplot(x='Combined Pre-Tax Income of HH', y='DOLLARS__beer', data=fakedata.groupby('Combined Pre-Tax Income of HH')[dollars_colum].mean().reset_index(),
                color=sns.light_palette("navy")[1], alpha=0.7)
    plt.ylabel("Dollars", fontsize=10)
    plt.title("Avg shopping dollars (Real vs Synthetic) for Income")

    plt.subplot(3, 2, 6)
    matrix = np.triu(fakedata[sample_corr].corr())
    sns.heatmap(fakedata[sample_corr].corr(), annot=True, mask=matrix, fmt=".1f", cmap=sns.color_palette("Oranges"),
                annot_kws={"size": 10})
    plt.title("Correlation plot of Synthetic data")
    plt.subplots_adjust(wspace=0.3)

    plt.subplot(3, 2, 5)
    matrix = np.triu(data[sample_corr].corr())
    sns.heatmap(data[sample_corr].corr(), annot=True, mask=matrix, fmt=".1f", cmap=sns.color_palette("Oranges"),
                annot_kws={"size": 10})
    plt.title("Correlation plot of Real data")
    plt.subplots_adjust(wspace=0.3)



    ##
    plt.subplot(3, 2, 4)
    sns.barplot(x='Family Size', y='DOLLARS__beer', data=data.groupby('Family Size')[dollars_colum].mean().reset_index(),
                color=sns.light_palette("navy")[3], alpha=0.7)
    sns.barplot(x='Family Size', y='DOLLARS__beer', data=fakedata.groupby('Family Size')[dollars_colum].mean().reset_index(),
                color=sns.light_palette("navy")[1], alpha=0.7)
    plt.ylabel("Dollars", fontsize=10)
    plt.title("Avg shopping dollars (Real vs Synthetic) for Family Size")

    #display(df.sort_values('Mean_diff', ascending=False))

    ###

    # plt.subplot(2, 2, 3)
    real = np.log(data[continus_column].mean())
    fake = np.log(fakedata[continus_column].mean())
    # log mean
    plt.subplot(3, 2, 1)
    plt.scatter(real, fake, color='purple')
    # add a 45 degree line
    plt.plot([-7.5, 4], [-7.5, 4], color='black', linewidth=2)

    plt.xlabel("Log Mean Real", fontsize=10)
    plt.ylabel("Log Mean Synthetic", fontsize=10)
    plt.legend(['Log Mean'], fontsize=10)
    plt.subplots_adjust(hspace = 0.5)
    plt.title('Means of Real vs Synthetic data', fontsize=12)

    # std
    plt.subplot(3, 2, 2)
    real = np.log(data[continus_column].std())
    fake = np.log(fakedata[continus_column].std())

    # add a 45 degree line
    plt.plot([-5, 5], [-5, 5], color='black', linewidth=2)
    plt.scatter(real, fake, color='purple')
    plt.xlabel("Log Standard Deviation Real", fontsize=10)
    plt.ylabel("Log Standard Deviation Synthetic", fontsize=10)
    plt.legend(['Log STDs'], fontsize=10)
    plt.subplots_adjust(hspace = 0.5)
    plt.title('Standard Deviations of real vs Synthetic data', fontsize=12)

    fig.suptitle('Data Visualization Real vs ' + str(len(fakedata)) + ' Synthetic Data', fontsize=20)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    def on_key(event):
        print("You pressed {}".format(event.key))
        key_press_handler(event,canvas,toolbar)

    canvas.mpl_connect("key_press_event",on_key)

    def quit():
        root.quit()
        root.destroy()


    button = tkinter.Button(master=root, text="Quit", font=('Verdana', 12), command=quit)
    button.pack(side=tkinter.BOTTOM)

    tkinter.mainloop()

if __name__ == "__main__":
    main()

    # set up validation functions coorelation

