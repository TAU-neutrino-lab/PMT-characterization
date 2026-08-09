from pathlib import Path

import pandas as pd


from pmt.fit.core import *
from pmt.fit.poisson import fit_poisson_spe
from pmt.fit.bellamy import fit_bellamy_spe
from pmt.plotting import save_plot



def fit_charge(inputV, file_name, fit_inputs, cfg, save_dir, maxPE=3, nbins=250, fit_model="poisson", default_key="WA0089_20MHz_led100_defaults", lbl=None, save_results=False, initPars=None):

    #--- save stuff
    save_fitresults = Path(f"{save_dir}/fit_results")
    save_fitresults.mkdir(parents=True, exist_ok=True)
    save_fitplots = Path(f"{save_dir}/fit_plots")
    save_fitplots.mkdir(parents=True, exist_ok=True)

    #--- load dataframe
    df_file = Path(fit_inputs) / f"{file_name}_df.pkl"
    print(f"\nLoading cached dataframe from {df_file}")
    df = pd.read_pickle(df_file)

    #--- fit variable
    charge_mV_ns = df.charge_cfd50_window_mV_ns.values 
    fit_range_full = (np.min(charge_mV_ns), np.max(charge_mV_ns))

    #--- initial parameters
    if initPars is None:
        initPars = get_initPars( cfg, file_name, fit_model, default_key)
    try:
        fit_rg = cfg[file_name]['fit_range']
        print(f'Fit range = {fit_rg}')
    except:
        fit_rg = fit_range_full
        print(f'No fit range specified, using full range {fit_rg}')
    initPars["n_total"] = [len(charge_mV_ns), 0.5 * len(charge_mV_ns), 1.5 * len(charge_mV_ns), False]
    #--- perform fit
    model_name=""
    if fit_model=="poisson":
        model_name = "Poisson"
        fit = fit_poisson_spe( charge_mV_ns, p0=initPars, max_pe=maxPE, bins=nbins, fit_range=fit_rg )
    elif fit_model=="bellamy":
        model_name = "Bellamy"
        fit = fit_bellamy_spe( charge_mV_ns, p0=initPars, max_pe=maxPE, bins=nbins, fit_range=fit_rg )
    print_fit_result_table(fit)
    if fit["diagnostics"] != []:
        print(fit["diagnostics"])

    G, G_err = compute_gain(fit['parameters']['q1_mV_ns'], fit['errors']['q1_mV_ns'])
    print(f"\n Gain is {G:.3g} +-{G_err:.2e} for Q1 = {fit['parameters']['q1_mV_ns']}")
    exp = 6
    G_scaled = G / 10**exp
    Gerr_scaled = G_err / 10**exp
    title = fr"{inputV} V - {model_name} Fit -  $G = ({G_scaled:.3f} \pm {Gerr_scaled:.3f}) \times 10^{exp}$"

    fig, _, _, _ = plot_fit_summary( fit, title=title, shrink_colorbar=0.7, component_visibility_fraction=0, logscale=True,
                                                    show_event_fractions=True, max_fraction_pe=maxPE) 

    if save_results:
        saveplot = f"fit_{model_name}_{nbins}bins"
        saveres = f"{file_name}_{model_name}"
        if lbl is not None:
            saveplot += f"f_{lbl}"
            saveres  += f"f_{lbl}"
        save_plot(fig, True, save_fitplots, file_name, saveplot,  Nevents=None);
        save_fit_results(fit, output_path=f"{save_fitresults}/{saveres}.yaml");