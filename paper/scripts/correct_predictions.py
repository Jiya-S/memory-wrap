import sys
sys.path.append('..')
import csv
import random
import torch # type: ignore
import absl.flags
import absl.app
from sklearn.neighbors import NearestNeighbors # type: ignore
from sklearn.cluster import KMeans # type: ignore
import utils.datasets as datasets
import utils.utils as utils

DEFAULT_INDICES = "132,133,136,176,211,233,225,240,251,253,257,262,268,271,305,314,318,319,346,357"

absl.flags.DEFINE_string("path_model", None, "Path of the trained model")
absl.flags.DEFINE_string("dir_dataset", '../datasets/', "dir path where datasets are stored")
absl.flags.DEFINE_string("target_indices", DEFAULT_INDICES, "Comma-separated abs_idx values of the wrong predictions to try to correct")
absl.flags.DEFINE_integer("mem_set_size", 100, "Number of samples per constructed memory set")
absl.flags.DEFINE_integer("num_clusters", 10, "Number of clusters for the clustering strategy")
absl.flags.DEFINE_integer("seed", 42, "Seed for reproducible sampling")
absl.flags.DEFINE_string("out_csv", "correction_results.csv", "Where to write the per-image results")
absl.flags.mark_flag_as_required("path_model")

FLAGS = absl.flags.FLAGS


def run(path: str, dataset_dir: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:{}".format(device))
    torch.manual_seed(FLAGS.seed)
    rng = random.Random(FLAGS.seed)

    checkpoint = torch.load(path, map_location=device)
    modality = checkpoint['modality']
    dataset_name = checkpoint['dataset_name']
    model = utils.get_model(checkpoint['model_name'], checkpoint['num_classes'], model_type=modality)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    train_examples = checkpoint['train_examples']
    load_dataset = getattr(datasets, 'get_' + dataset_name)
    _, _, test_loader, mem_loader = load_dataset(dataset_dir, batch_size_train=50, batch_size_test=3, batch_size_memory=100, size_train=train_examples)
    memory_iter = iter(mem_loader)

    target_indices = set(int(x) for x in FLAGS.target_indices.split(','))
    min_target = min(target_indices)
    max_target = max(target_indices)

    mem_images, mem_labels = zip(*[mem_loader.dataset[i] for i in range(len(mem_loader.dataset))])
    mem_images = torch.stack(mem_images).to(device)
    mem_labels = torch.tensor(mem_labels)
    with torch.no_grad():
        mem_vectors = model.forward_encoder(mem_images).cpu().numpy()

    knn = NearestNeighbors(n_neighbors=FLAGS.mem_set_size).fit(mem_vectors)
    kmeans = KMeans(n_clusters=FLAGS.num_clusters, random_state=FLAGS.seed, n_init=10).fit(mem_vectors)
    cluster_positions = {}
    for pos, cluster_id in enumerate(kmeans.labels_):
        cluster_positions.setdefault(cluster_id, []).append(pos)

    def sample_positions(positions, size):
        if len(positions) >= size:
            return rng.sample(positions, size)
        return [rng.choice(positions) for _ in range(size)]

    def predict(image, positions):
        memory = mem_images[positions]
        with torch.no_grad():
            outputs, _ = model(image, memory, return_weights=True)
        return torch.argmax(outputs, 1).item()

    rows = []
    abs_idx = 0
    for images, labels in test_loader:
        if abs_idx > max_target:
            break
        if abs_idx + len(images) <= min_target:
            abs_idx += len(images)
            continue
        try:
            memory, _ = next(memory_iter)
        except StopIteration:
            memory_iter = iter(mem_loader)
            memory, _ = next(memory_iter)

        images = images.to(device)
        memory = memory.to(device)
        with torch.no_grad():
            outputs, _ = model(images, memory, return_weights=True)
        predictions = torch.argmax(outputs, 1)

        for ind in range(len(images)):
            if abs_idx not in target_indices:
                abs_idx += 1
                continue

            true_label = labels[ind].item()
            original_pred = predictions[ind].item()
            reproducibly_wrong = original_pred != true_label
            row = {'abs_idx': abs_idx, 'true_class': true_label, 'original_pred': original_pred,
                   'reproducibly_wrong': reproducibly_wrong}

            if reproducibly_wrong:
                input_selected = images[ind].unsqueeze(0)
                with torch.no_grad():
                    query_vector = model.forward_encoder(input_selected).cpu().numpy()

                _, knn_positions = knn.kneighbors(query_vector)
                knn_pred = predict(input_selected, knn_positions[0].tolist())
                row['knn_pred'] = knn_pred
                row['knn_corrected'] = knn_pred == true_label

                cluster_id = int(kmeans.predict(query_vector)[0])
                cluster_set = sample_positions(cluster_positions[cluster_id], FLAGS.mem_set_size)
                cluster_pred = predict(input_selected, cluster_set)
                row['cluster_id'] = cluster_id
                row['cluster_pred'] = cluster_pred
                row['cluster_corrected'] = cluster_pred == true_label

                generator = torch.Generator().manual_seed(FLAGS.seed + abs_idx)
                sampler = torch.utils.data.RandomSampler(mem_images, replacement=False, generator=generator)
                random_positions = list(sampler)[:FLAGS.mem_set_size]
                random_pred = predict(input_selected, random_positions)
                row['random_pred'] = random_pred
                row['random_corrected'] = random_pred == true_label

            rows.append(row)
            abs_idx += 1

    fieldnames = ['abs_idx', 'true_class', 'original_pred', 'reproducibly_wrong',
            'knn_pred', 'knn_corrected', 'cluster_id', 'cluster_pred', 'cluster_corrected',
               'random_pred', 'random_corrected']
    with open(FLAGS.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    wrong_rows = [r for r in rows if r['reproducibly_wrong']]
    knn_hits = sum(1 for r in wrong_rows if r['knn_corrected'])
    cluster_hits = sum(1 for r in wrong_rows if r['cluster_corrected'])
    random_hits = sum(1 for r in wrong_rows if r['random_corrected'])
    print('Reproducibly wrong: {}/{}'.format(len(wrong_rows), len(rows)))
    print('kNN strategy corrected: {}/{}'.format(knn_hits, len(wrong_rows)))
    print('Cluster strategy corrected: {}/{}'.format(cluster_hits, len(wrong_rows)))
    print('Random strategy corrected: {}/{}'.format(random_hits, len(wrong_rows)))


def main(argv):
    run(FLAGS.path_model, FLAGS.dir_dataset)


if __name__ == '__main__':
    absl.app.run(main)
