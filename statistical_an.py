from scipy.stats import f_oneway, t, sem
import numpy as np
import constants

## Prints ANOVA test results and confidence values for precision and recall
## using the 10-fold cross-validation computed in experiment 1

domain_pct=1024
mem_sizes=[16, 32, 64]
domain_sizes = [64, 128, 256, 512, 1024]
names = ['memory_precision', 'memory_recall']


## Get the results of the ten folds for each domain and quantizaiton level,
## and for each memory sizes nxm and filling percentage
def performance_msz(es, filling_pct=False):
    ret_dict = {key:None for key in mem_sizes} if filling_pct else {key: None for key in domain_sizes}

    dir_prefix = 'runs-'
    if filling_pct:
        dirname=f'{dir_prefix}{domain_pct}'
        constants.run_path=dirname

        fills = constants.memory_fills
        for msz in mem_sizes:
            print(f"\n====================================\nDomain: {domain_pct}, Memory size: {msz}")
            pre_pct=[]
            re_pct=[]
            
            for fname in names:
                flag_print=True
                filename = constants.csv_filename(fname+f"-sze_"+constants.int_suffix(msz), es)
                data = np.genfromtxt(filename, delimiter=',')*100
                for idx in range(len(fills)):
                    data_pct=data[:,idx]
                    if msz==64 and idx==0:
                        print(data_pct) # <---- We get nan since the results of the 10 fold is zero
                    mean_data_pct=np.mean(data_pct)
                    if fname=='memory_precision':
                        pre_pct.append(data_pct)
                        (l,h)=t.interval(confidence=0.95, df=len(data_pct)-1, loc=mean_data_pct, scale=sem(data_pct))
                        margin_error = (h-l)/2
                        print(f"Precision\nPercentage: {fills[idx]}, mean: {mean_data_pct:.3f}, confidence interval: {l:.3f}, {h:.3f} or {mean_data_pct:.3f} ± {margin_error:.3f}") if flag_print else print(f"Percentage: {fills[idx]}, mean: {mean_data_pct:.3f}, confidence interval: {l:.3f}, {h:.3f} or {mean_data_pct:.3f} ± {margin_error:.3f}")
                    else:
                        re_pct.append(data_pct)
                        (l,h)=t.interval(confidence=0.95, df=len(data_pct)-1, loc=mean_data_pct, scale=sem(data_pct))
                        margin_error = (h-l)/2
                        print(f"\nRecall\nPercentage: {fills[idx]}, mean: {mean_data_pct:.3f}, confidence interval: {l:.3f}, {h:.3f} or {mean_data_pct:.3f} ± {margin_error:.3f}") if flag_print else print(f"Percentage: {fills[idx]}, mean: {mean_data_pct:.3f}, confidence interval: {l:.3f}, {h:.3f} or {mean_data_pct:.3f} ± {margin_error:.3f}")
                    flag_print=False
            ret_dict[msz]={"pre_pct":pre_pct,"re_pct":re_pct}
    else:
        for domain in domain_sizes:
            print(f"\n====================================\nDomain: {domain}")
            dirname = f'{dir_prefix}{domain}'
            constants.run_path=dirname
        
            pre_msz=[]
            re_msz=[]
            sizes = constants.memory_sizes
            for fname in names:
                flag_print=True
                filename = constants.csv_filename(fname, es)
                data = np.genfromtxt(filename, delimiter=',')
                for idx in range(len(sizes)):
                    data_msz=data[:,idx]
                    mean_data_msz=np.mean(data_msz)
                    if fname=='memory_precision':
                        pre_msz.append(data_msz)
                        (l,h)=t.interval(confidence=0.95, df=len(data_msz)-1, loc=mean_data_msz, scale=sem(data_msz))
                        margin_error = (h-l)/2
                        print(f"Precision\nMem size: {sizes[idx]}, mean: {mean_data_msz:.2f}, confidence interval: {l:.2f}, {h:.2f} or {mean_data_msz:.3f} ± {margin_error:.3f}") if flag_print else print(f"Mem size: {sizes[idx]}, mean: {mean_data_msz:.2f}, confidence interval: {l:.2f}, {h:.2f} or {mean_data_msz:.3f} ± {margin_error:.3f}")
                    else:
                        re_msz.append(data_msz)
                        (l,h)=t.interval(confidence=0.95, df=len(data_msz)-1, loc=mean_data_msz, scale=sem(data_msz))
                        margin_error = (h-l)/2
                        print(f"\nRecall\nMem size: {sizes[idx]}, mean: {mean_data_msz:.2f}, confidence interval: {l:.2f}, {h:.2f} or {mean_data_msz:.3f} ± {margin_error:.3f}") if flag_print else print(f"Mem size: {sizes[idx]}, mean: {mean_data_msz:.2f}, confidence interval: {l:.2f}, {h:.2f} or {mean_data_msz:.3f} ± {margin_error:.3f}")
                    flag_print=False
            ret_dict[domain]={"pre_msz":pre_msz,"re_msz":re_msz}
                    
    return ret_dict


if __name__ == "__main__":
    es = constants.ExperimentSettings()

    ###########################
    ## Fixed memory size, get ANOVA for precision and recall of all domains and folds
    print("\n\n-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\nCompute performance")
    dict_domain = performance_msz(es)
    sizes = constants.memory_sizes

    print("\n========================================================================")
    print("========================================================================")
    print("Compute ANOVA value with fixed memory\nsize, for all domains and folds")
    ## ANOVA for precision of each memory size for all domains
    print("\n==================\n\tANOVA for precision p-values\n==================")
    for sz in range(len(sizes)):
        F_precision=f_oneway(dict_domain[64]["pre_msz"][sz],
                             dict_domain[128]["pre_msz"][sz],
                             dict_domain[256]["pre_msz"][sz],
                             dict_domain[512]["pre_msz"][sz],
                             dict_domain[1024]["pre_msz"][sz])
        print(f"Memory size {sizes[sz]}:\t {F_precision.pvalue:.3e}")
    ## ANOVA for recall of each memory size for all domains
    print("\n==================\n\tANOVA for recall p-values\n==================")
    for sz in range(len(sizes)):
        F_recall=f_oneway(dict_domain[64]["re_msz"][sz],
                          dict_domain[128]["re_msz"][sz],
                          dict_domain[256]["re_msz"][sz],
                          dict_domain[512]["re_msz"][sz],
                          dict_domain[1024]["re_msz"][sz])
        print(f"Memory size {sizes[sz]}:\t {F_recall.pvalue:.3e}")
    ###########################
    ###########################
    ## Fixed domain, get ANOVA for precision and recall of all memory sizes and folds
    print("\n\n========================================================================")
    print("========================================================================")
    print("Compute ANOVA value with fixed domain\nsize, for all memory sizes and folds")
    ## ANOVA for precision of each domain for all memory sizes
    print("\n==================\n\tANOVA for precision p-values\n==================")
    for domain in domain_sizes:
        F_precision=f_oneway(dict_domain[domain]["pre_msz"][0],
                             dict_domain[domain]["pre_msz"][1],
                             dict_domain[domain]["pre_msz"][2],
                             dict_domain[domain]["pre_msz"][3],
                             dict_domain[domain]["pre_msz"][4],
                             dict_domain[domain]["pre_msz"][5],
                             dict_domain[domain]["pre_msz"][6],
                             dict_domain[domain]["pre_msz"][7],
                             dict_domain[domain]["pre_msz"][8],
                             dict_domain[domain]["pre_msz"][9])
        print(f"Domain {domain}:\t {F_precision.pvalue:.3e}")
    ## ANOVA for recall of each domain for all memory sizes
    print("\n==================\n\tANOVA for recall p-values\n==================")
    for domain in domain_sizes:
        F_recall=f_oneway(dict_domain[domain]["re_msz"][0],
                          dict_domain[domain]["re_msz"][1],
                          dict_domain[domain]["re_msz"][2],
                          dict_domain[domain]["re_msz"][3],
                          dict_domain[domain]["re_msz"][4],
                          dict_domain[domain]["re_msz"][5],
                          dict_domain[domain]["re_msz"][6],
                          dict_domain[domain]["re_msz"][7],
                          dict_domain[domain]["re_msz"][8],
                          dict_domain[domain]["re_msz"][9])
        print(f"Domain {domain}:\t {F_recall.pvalue:.3e}")

    ######################################################
    ######################################################
    ######################################################
    ######################################################

    ## Domain 1,024. Fixed percentage, get ANOVA for precision and recall of memory sizes 16, 32 and 64, and all folds
    print("\n\n-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\nCompute performance")
    dict_msizes = performance_msz(es, True)
    fills = constants.memory_fills

    print("\n========================================================================")
    print("========================================================================")
    print("Compute ANOVA value with fixed filling percentage, and \ndomain of 1,024, memory sizes of 16, 32 and 64, and all folds")
    ## ANOVA for precision of each filling percentage for domain 1,024, memory sizes 16, 32 and 64, and all folds
    print("\n==================\n\tANOVA for precision p-values\n==================")
    for f_idx in range(len(fills)):
        F_precision=f_oneway(dict_msizes[16]["pre_pct"][f_idx],
                             dict_msizes[32]["pre_pct"][f_idx],
                             dict_msizes[64]["pre_pct"][f_idx])
        print(f"Percentage {fills[f_idx]}:\t {F_precision.pvalue:.3e}")
    ## ANOVA for recall of each filling percentage for domain 1,024, memory sizes 16, 32 and 64, and all folds
    print("\n==================\n\tANOVA for recall p-values\n==================")
    for f_idx in range(len(fills)):
        F_recall=f_oneway(dict_msizes[16]["re_pct"][f_idx],
                          dict_msizes[32]["re_pct"][f_idx],
                          dict_msizes[64]["re_pct"][f_idx])
        print(f"Percentage {fills[f_idx]}:\t {F_recall.pvalue:.3e}")
    ###########################
    ###########################
    ## Domain 1,024. Fixed memory size (16, 32, 64), get ANOVA for precision and recall of all filling percentages and folds
    print("\n\n========================================================================")
    print("========================================================================")
    print("Compute ANOVA value with fixed memory\nsize (16, 32, 64), and domain of 1,024, all filling percentages and folds")
    ## ANOVA for precision of each memory size (16, 32, 64) for all filling percentages
    print("\n==================\n\tANOVA for precision p-values\n==================")
    for msize in mem_sizes:
        F_precision=f_oneway(dict_msizes[msize]["pre_pct"][0],
                             dict_msizes[msize]["pre_pct"][1],
                             dict_msizes[msize]["pre_pct"][2],
                             dict_msizes[msize]["pre_pct"][3],
                             dict_msizes[msize]["pre_pct"][4],
                             dict_msizes[msize]["pre_pct"][5],
                             dict_msizes[msize]["pre_pct"][6],
                             dict_msizes[msize]["pre_pct"][7])
        print(f"Memory size {msize}:\t {F_precision.pvalue:.3e}")
    ## ANOVA for recall of each memory size (16, 32, 64) for all filling percentages
    print("\n==================\n\tANOVA for recall p-values\n==================")
    for msize in mem_sizes:
        F_recall=f_oneway(dict_msizes[msize]["re_pct"][0],
                          dict_msizes[msize]["re_pct"][1],
                          dict_msizes[msize]["re_pct"][2],
                          dict_msizes[msize]["re_pct"][3],
                          dict_msizes[msize]["re_pct"][4],
                          dict_msizes[msize]["re_pct"][5],
                          dict_msizes[msize]["re_pct"][6],
                          dict_msizes[msize]["re_pct"][7])
        print(f"Memory size {msize}:\t {F_recall.pvalue:.3e}")

