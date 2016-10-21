import h5py
import numpy
import scipy.stats
from sklearn.metrics import confusion_matrix

from crowdastro.experiment.results import Results
from crowdastro.crowd.raykar import RaykarClassifier


def raykar_params(crowdastro_path, results_path, method, n_annotators=50):
    results = Results.from_path(results_path)
    n_splits = results.n_splits
    assert method in results.methods
    alphas = []
    betas = []
    for split_id in range(n_splits):
        model = results.get_model(method, split_id)
        rc = RaykarClassifier.unserialise(model)
        alphas.append(rc.a_)
        betas.append(rc.b_)
    alphas = numpy.array(alphas)
    betas = numpy.array(betas)
    print('estimated alphas:', alphas.mean(axis=0))
    print('stdevs:', alphas.std(axis=0))
    print('estimated betas:', betas.mean(axis=0))
    print('stdevs:', betas.std(axis=0))

    # Get the "true" alpha/beta.
    with h5py.File(crowdastro_path, 'r') as f:
        norris = f['/wise/cdfs/norris_labels'].value
        labels = numpy.ma.MaskedArray(
            f['/wise/cdfs/rgz_raw_labels'],
            mask=f['/wise/cdfs/rgz_raw_labels_mask'])

        annotator_accuracies = []
        true_alphas = []
        true_betas = []
        for t in range(labels.shape[0]):
            cm = confusion_matrix(norris[~labels[t].mask],
                                  labels[t][~labels[t].mask])
            if cm.shape[0] == 1:
                continue

            tp = cm[1, 1]
            n, p = cm.sum(axis=1)
            tn = cm[0, 0]
            if not n or not p or p + n < 700:
                annotator_accuracies.append(0)
                true_alphas.append(tp / p)
                true_betas.append(tn / n)
                continue

            ba = (tp / p + tn / n) / 2
            annotator_accuracies.append(ba)
            true_alphas.append(tp / p)
            true_betas.append(tn / n)
        assert len(annotator_accuracies) == len(true_alphas)
        ranked_annotators = numpy.argsort(annotator_accuracies)
        top_n_annotators = ranked_annotators[-n_annotators:]
        true_alphas = numpy.array(true_alphas)[top_n_annotators]
        true_betas = numpy.array(true_betas)[top_n_annotators]

    print(true_alphas)
    print(true_betas)

    all_pred_alphas = []
    all_true_alphas = []
    for alphas_ in alphas:
        # For each trial...
        all_true_alphas.extend(true_alphas)
        all_pred_alphas.extend(alphas_)

    all_pred_betas = []
    all_true_betas = []
    for betas_ in betas:
        # For each trial...
        all_true_betas.extend(true_betas)
        all_pred_betas.extend(betas_)

    pearson_alpha = scipy.stats.pearsonr(all_pred_alphas, all_true_alphas)
    pearson_beta = scipy.stats.pearsonr(all_pred_betas, all_true_betas)
    spearman_alpha = scipy.stats.spearmanr(all_pred_alphas, all_true_alphas)
    spearman_beta = scipy.stats.spearmanr(all_pred_betas, all_true_betas)

    print('Pearson for alpha:', pearson_alpha)
    print('Pearson for beta:', pearson_beta)
    print('Spearman for alpha:', spearman_alpha)
    print('Spearman for beta:', spearman_beta)

    # plt.subplot(1, 2, 1)
    # for alphas_ in alphas:
    #     plt.scatter(
    #         true_alphas,
    #         alphas_,
    #         marker='x')
    # plt.xlabel('$\\alpha$')
    # plt.ylabel('Estimated $\\alpha$')

    # plt.subplot(1, 2, 2)
    # for betas_ in betas:
    #     plt.scatter(
    #         true_betas,
    #         betas_,
    #         marker='x')
    # plt.xlabel('$\\beta$')
    # plt.ylabel('Estimated $\\beta$')

    # plt.show()


if __name__ == '__main__':
    raykar_params(
        'data/crowdastro.h5', 'data/results_50.h5', 'Raykar(RGZ-Top-50)')
