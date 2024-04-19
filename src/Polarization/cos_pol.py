from Polarization.polarization_plots import *
from Axes.bootstraping import *
import os 

print(os.getcwd())

def draw_cos_pol(left_side, right_side, curves_by_company=None, axis=None,
    percentiles=[10, 90], print_random_pol=True, force_i_lim=None, with_parliament=True):

    if curves_by_company:
        raise ValueError("Not implemented with company curves yet")
    
    if not axis:
        raise ValueError("It only works on an axis")
    
    companies = ["all"]

    sources = left_side + right_side

    if os.path.exists(f"plots/Polarization/Polarization and cosine between {left_side} VS {right_side} ; axis = {axis}, companies = {companies}, percentiles = {percentiles}, with parliament = {with_parliament}.png"):
        img = mpimg.imread(f'plots/Polarization/Polarization and cosine between {left_side} VS {right_side} ; axis = {axis}, companies = {companies}, percentiles = {percentiles}, with parliament = {with_parliament}.png')
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    else :
        if not os.path.exists(f"notebooks/polarization/polarization values/Polarization between {left_side} VS {right_side} ; axis = {axis}, companies = {companies}, percentiles = {percentiles}, with parliament = {with_parliament}.csv"):
            print('computin polarization')
            choose_pol(left_side=left_side, right_side=right_side, curves_by_company=None, axis=axis, percentiles=percentiles, print_random_pol=print_random_pol, force_i_lim=force_i_lim, with_parliament=with_parliament)

        else:
            print('polarization already computed')

        if with_parliament:
            df_proj = pd.read_csv("data/with parliament/current_dataframes/df.csv")
        if not with_parliament:
            df_proj = pd.read_csv("data/without parliament/current_dataframes/df.csv")
            df_proj['party'], df_proj['Speaker'] = 0, 0

        df_par = df_proj.loc[df_proj["source"].isin(sources) | df_proj["party"].isin(sources)]

        # Further refine the DataFrame structure for analysis
        df_par = df_par[['year', 'party', 'text', 'source', 'keywords', 'Speaker', 'cos axe 1', 'cos axe 2']]

        # Split the data into two DataFrames based on a specific source criterion
        df1 = df_par[df_par['source'] == 'par']
        df2 = df_par[df_par['source'] != 'par']

        # Define a function to translate newspaper source to party
        def translate_party(newspaper):
            """
            Translates newspaper sources to their corresponding political party.
            
            :param newspaper: The source to be translated.
            :return: The political party corresponding to the source.
            """
            if newspaper in left_side:
                return "Lab"
            if newspaper in right_side:
                return "Con"
        
        # Apply the translation function to assign parties based on sources
        df2['party'] = df2['source'].apply(translate_party)
        df2['Speaker'] = range(len(df2))

        # Combine the two DataFrames and reset index for continuity
        df_par = pd.concat([df1, df2]).reset_index(drop=True)

        df_par_grouped = df_par[["party", "year", f"cos axe {axis}"]]
        df_par_grouped = df_par_grouped.groupby(["party", "year"]).mean()
        df_par_grouped = df_par_grouped.reset_index()

        def change_year(old_year):
            if int(old_year) == 20110:
                return 2020
            if int(old_year) == 20111:
                return 2021
            if int(old_year) == 20112:
                return 2022
            if int(old_year) == 20113:
                return 2023
            else :
                return int(old_year)
            
        df_par['year'] = df_par['year'].apply(change_year)
        df_par_grouped['year'] = df_par_grouped['year'].apply(change_year)

        df_par_grouped = bootstrap(df_par_grouped, df_par, source_column="party", axis=axis)

        df_par_grouped["cos axe"] = df_par_grouped[f"cos axe {axis}"]

        df_pol = pd.read_csv(
        f"notebooks/polarization/polarization values/Polarization between {left_side} VS {right_side} ; axis = {axis}, companies = {companies}, percentiles = {percentiles}, with parliament = {with_parliament}.csv"
        )

        real_pol = np.array(df_pol["real_pol"])
        random_pol = np.array(df_pol["random_pol"])
        CI_lows_real = np.array(df_pol["CI_lows_real"])
        CI_high_real = np.array(df_pol["CI_high_real"])
        CI_lows_random = np.array(df_pol["CI_lows_random"])
        CI_high_random = np.array(df_pol["CI_high_random"])

        Con_cos = np.array(df_par_grouped[df_par_grouped["party"] == "Con"]["cos axe"])
        Con_CI_low = np.array(
            df_par_grouped[df_par_grouped["party"] == "Con"][f"CI_{axis}_inf"], dtype="float"
        )
        Con_CI_high = np.array(
            df_par_grouped[df_par_grouped["party"] == "Con"][f"CI_{axis}_sup"], dtype="float"
        )

        Lab_cos = np.array(df_par_grouped[df_par_grouped["party"] == "Lab"]["cos axe"])
        Lab_CI_low = np.array(
            df_par_grouped[df_par_grouped["party"] == "Lab"][f"CI_{axis}_inf"], dtype="float"
        )
        Lab_CI_high = np.array(
            df_par_grouped[df_par_grouped["party"] == "Lab"][f"CI_{axis}_sup"], dtype="float"
        )

        x = [2010 + i for i in range(len(real_pol))]
        len_x = len(x)
        len_y = len(Con_cos)

        if len_y < len_x:
            x = x[len_x-len_y:]
            real_pol = real_pol[len_x-len_y:]
            CI_lows_real = CI_lows_real[len_x-len_y:]
            CI_high_real = CI_high_real[len_x-len_y:]

        fig, ax1 = plt.subplots(figsize=(15, 7))

        # Plotting the first set of data
        ax1.plot(x, real_pol, label="Polarisation réelle", color="blue", linewidth=2)
        ax1.fill_between(x, CI_lows_real, CI_high_real, color="blue", alpha=0.05)
        ax1.set_xlabel("Année")
        ax1.set_ylabel("Polarisation")
        ax1.legend(loc="upper left")

        # Creating a second y-axis
        ax2 = ax1.twinx()

        # Plotting the second set of data on the secondary y-axis
        ax2.plot(
            x,
            Con_cos[:len(x)],
            label=f"Cosine similarity of {left_side}",
            color="green",
            linestyle="--",
            linewidth=2,
        )
        ax2.fill_between(x, Con_CI_low[:len(x)], Con_CI_high[:len(x)], color="green", alpha=0.05)
        ax2.plot(
            x,
            Lab_cos[:len(x)],
            label=f"Cosine similarity of {right_side}",
            color="red",
            linestyle="--",
            linewidth=2,
        )
        ax2.fill_between(x, Lab_CI_low[:len(x)], Lab_CI_high[:len(x)], color="red", alpha=0.05)
        ax2.set_ylabel("Cosine similarity")
        ax2.legend(loc="upper right")

        # Adding vertical dotted lines for each year
        for year in x:
            ax1.axvline(year, color="gray", linestyle="--", alpha=0.5)

        plt.title(f"Polarization and cosine between {left_side} VS {right_side} ; axis = {axis}, companies = {companies}, percentiles = {percentiles}, with parliament = {with_parliament}")
        plt.grid(True, linestyle="--", alpha=0.5, axis="x")  # Adding both x and y gridlines
        plt.tight_layout()
        plt.savefig(f"plots/Polarization/Polarization and cosine between {left_side} VS {right_side} ; axis = {axis}, companies = {companies}, percentiles = {percentiles}, with parliament = {with_parliament}.png")
        plt.show()