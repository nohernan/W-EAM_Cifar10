# Copyright [2020] Luis Alberto Pineda Cortés, Gibrán Fuentes Pineda,
# Rafael Morales Gamboa.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entropic Associative Memory Experiments
"""
from docopt import docopt
import gettext
import matplotlib.pyplot as plt
import numpy as np
import constants
import matplotlib as mpl

domain_sizes = [64]
memory_sizes = [16,32,64]
selected_domain = 1024
my_tag="revision1__"

def performance_plot(pre_mean, rec_mean, ent_mean, pre_std, rec_std,
                   es, tag='', xlabels=constants.memory_sizes,
                   xtitle=None, ytitle=None):

    plt.clf()
    plt.figure(figsize=(6.4, 4.8))

    full_length = 100.0
    step = 0.1
    main_step = full_length/len(xlabels)
    x = np.arange(0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + 0.25#+ step

    # Gives space to fully show markers in the top.
    ymax = full_length #+ 2

    # Replace undefined precision with 1.0.
    pre_mean = np.nan_to_num(pre_mean, copy=False, nan=100.0)

    plt.errorbar(x, pre_mean, fmt='r-o', yerr=pre_std, label='Precision', capthick=2, capsize=3)
    plt.errorbar(x, rec_mean, fmt='b--s', yerr=rec_std, label='Recall', capthick=2, capsize=3)
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.xticks(x, xlabels)

    if xtitle is None:
        xtitle = 'Range Quantization Levels'
    if ytitle is None:
        ytitle = 'Percentage'

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend(loc=4)
    plt.grid(True)

    entropy_labels = [str(e) for e in np.around(ent_mean, decimals=1)]

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'mycolors', ['cyan', 'purple'])
    Z = [[0, 0], [0, 0]]
    levels = np.arange(0.0, xmax, step)
    CS3 = plt.contourf(Z, levels, cmap=cmap)

    cbar = plt.colorbar(CS3, orientation='horizontal')
    cbar.set_ticks(x)
    cbar.ax.set_xticklabels(entropy_labels)
    cbar.set_label('Entropy')
    
    s = tag + 'graph_prse_MEAN' + '-english'
    graph_filename = constants.picture_filename(s, es)
    plt.savefig(graph_filename, dpi=600)


def behaviours_plot(no_response, no_correct, correct, es):

    for i in range(len(no_response)):
        total = (no_response[i] + no_correct[i] + correct[i])/100.0
        no_response[i] /= total
        no_correct[i] /= total
        correct[i] /= total

    plt.clf()

    full_length = 100.0
    step = 0.1
    main_step = full_length/len(constants.memory_sizes)
    x = np.arange(0.0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step
    ymax = full_length
    width = 5       # the width of the bars: can also be len(x) sequence

    plt.bar(x, correct, width, label='Correct response')
    #plt.bar(x, correct, width, label=_('Respuesta correcta seleccionada'))
    cumm = np.array(correct)
    plt.bar(x, no_correct, width, bottom=cumm, label='No correct response')
    #plt.bar(x, no_correct, width, bottom=cumm, label=_('Respuesta correcta no seleccionada'))
    cumm += np.array(no_correct)
    plt.bar(x, no_response, width, bottom=cumm, label='No response')
    #plt.bar(x, no_response, width, bottom=cumm, label=_('Sin respuesta'))

    plt.xlim(-width, xmax + width)
    plt.ylim(0.0, ymax)
    plt.xticks(x, constants.memory_sizes)

    plt.xlabel('Range Quantization Levels')
    #plt.xlabel(_('Niveles de discretización'))
    plt.ylabel('Labels')
    #plt.ylabel(_('Etiquetas'))

    plt.legend(loc=0)
    plt.grid(axis='y')

    graph_filename = constants.picture_filename(
        my_tag + 'graph_behaviours_MEAN' + '-english', es)
    #'graph_behaviours_MEAN' + _('-spanish'), es)
    plt.savefig(graph_filename, dpi=600)




def performance(precision_filename, recall_filename,
                entropy_filename, es):
    precision = np.genfromtxt(precision_filename, delimiter=',', dtype=float)
    pre_mean = np.mean(precision, axis=0)
    pre_std = np.std(precision, axis=0)
    #
    recall = np.genfromtxt(recall_filename, delimiter=',', dtype=float)
    rec_mean = np.mean(recall, axis=0)
    rec_std = np.std(recall, axis=0)
    #
    all_entropies = np.genfromtxt(entropy_filename, delimiter=',', dtype=float)
    ent_mean = np.mean(all_entropies, axis=0)
    #
    print("\nPloting performance graph\n")
    performance_plot(pre_mean, rec_mean, ent_mean, pre_std, rec_std, es, tag=my_tag)

    
if __name__ == "__main__":
    dir_prefix = 'runs-'
    es = constants.ExperimentSettings()
    for domain in domain_sizes:
        dirname = f'{dir_prefix}{domain}'
        constants.run_path=dirname
        print(f'Domain size: {domain}')

        ################### graph_prse_MEAN
        pre_filename=constants.csv_filename('memory_precision',es)
        rec_filename=constants.csv_filename('memory_recall',es)
        ent_filename=constants.csv_filename('memory_entropy',es)
        performance(pre_filename, rec_filename, ent_filename, es)

        ################### graph_behaviours_MEAN
        mean_behaviours_fn = constants.csv_filename('mean_behaviours',es)
        mean_behaviours = np.genfromtxt(mean_behaviours_fn, delimiter=',', dtype=float)
        behaviours_plot(mean_behaviours[0], mean_behaviours[1], mean_behaviours[2], es)

        ################### recall_sze
        if domain == selected_domain:
            for msize in memory_sizes:
                pre_mean_fn = constants.csv_filename('main_average_precision' + constants.numeric_suffix('sze',msize),es)
                pre_mean =np.genfromtxt(pre_mean_fn, delimiter=',', dtype=float)
                rec_mean_fn = constants.csv_filename('main_average_recall' + constants.numeric_suffix('sze',msize),es)
                rec_mean =np.genfromtxt(rec_mean_fn, delimiter=',', dtype=float)
                ent_mean_fn = constants.csv_filename('main_average_entropy' + constants.numeric_suffix('sze',msize),es)
                ent_mean =np.genfromtxt(ent_mean_fn, delimiter=',', dtype=float)
                pre_std_fn = constants.csv_filename('main_stdev_precision' + constants.numeric_suffix('sze',msize),es)
                pre_std =np.genfromtxt(pre_std_fn, delimiter=',', dtype=float)
                rec_std_fn = constants.csv_filename('main_stdev_recall' + constants.numeric_suffix('sze',msize),es)
                rec_std =np.genfromtxt(rec_std_fn, delimiter=',', dtype=float)
                performance_plot(pre_mean*100, rec_mean*100, ent_mean,
                                 pre_std*100, rec_std*100, es, tag=my_tag+'recall'+
                                 constants.numeric_suffix('sze', msize),
                                 xlabels=constants.memory_fills, xtitle='Percentage of memory corpus')

        
