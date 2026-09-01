import sys
sys.path.append('..')
import os
import csv
import random
import torch # type: ignore
import torchvision # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import absl.flags
import absl.app
import utils.datasets as datasets
import utils.utils as utils

# user flags
absl.flags.DEFINE_string("path_model", None, "Path of the trained model")
absl.flags.DEFINE_integer("batch_size_test", 3, "Number of samples for each image")
absl.flags.DEFINE_string("dir_dataset", '../datasets/', "dir path where datasets are stored")
absl.flags.DEFINE_string("target_indices", "", "Comma-separated abs_idx values to restrict processing to (empty = all wrong predictions)")
absl.flags.DEFINE_enum("memory_mode", "all", ["original", "correct_class", "random_search", "all"], "Which memory-set experiment(s) to run")
absl.flags.DEFINE_integer("max_random_trials", 500, "Max random memory sets to try per image in random_search mode")
absl.flags.DEFINE_integer("mem_set_size", 100, "Number of samples per constructed memory set")
absl.flags.DEFINE_integer("seed", 42, "Seed for reproducible sampling")
absl.flags.mark_flag_as_required("path_model")

FLAGS = absl.flags.FLAGS



def run(path:str,dataset_dir:str):
    """ Function to generate memory images for testing images using a given
    model. Memory images show the samples in the memory set that have an
    impact on the current prediction.

    Args:
        path (str): model path
        dataset_dir (str): dir where datasets are stored
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:{}".format(device))
    torch.manual_seed(FLAGS.seed)
    rng = random.Random(FLAGS.seed)
    # load model
    checkpoint = torch.load(path, map_location=device)
    modality = checkpoint['modality']
    if modality not in ['memory','encoder_memory']:
        raise ValueError(f'Model\'s modality (model type) must be one of [\'memory\',\'encoder_memory\'], not {modality}.')
    dataset_name = checkpoint['dataset_name']
    model = utils.get_model( checkpoint['model_name'],checkpoint['num_classes'],model_type=modality)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()


    # load data
    train_examples = checkpoint['train_examples']
    if dataset_name == 'CIFAR10' or dataset_name == 'CINIC10':
        name_classes= ['airplane','automobile',	'bird',	'cat','deer','dog',	'frog'	,'horse','ship','truck']
    else:
        name_classes = range(checkpoint['num_classes'])
    load_dataset = getattr(datasets, 'get_'+dataset_name)
    undo_normalization = getattr(datasets, 'undo_normalization_'+dataset_name)
    batch_size_test = FLAGS.batch_size_test
    _, _, test_loader, mem_loader = load_dataset(dataset_dir,batch_size_train=50, batch_size_test=batch_size_test,batch_size_memory=100,size_train=train_examples)
    memory_iter = iter(mem_loader)

    target_indices = set(int(x) for x in FLAGS.target_indices.split(',')) if FLAGS.target_indices else None

    # preload the memory pool so we can quickly build custom memory sets (class-only / random)
    mem_images, mem_labels = zip(*[mem_loader.dataset[i] for i in range(len(mem_loader.dataset))])
    mem_images = torch.stack(mem_images).to(device)
    mem_labels = torch.tensor(mem_labels)
    class_to_positions = {}
    for pos, lbl in enumerate(mem_labels.tolist()):
        class_to_positions.setdefault(lbl, []).append(pos)

    def sample_memory(positions, size):
        chosen = rng.sample(positions, size) if len(positions) >= size else [rng.choice(positions) for _ in range(size)]
        return mem_images[chosen], chosen

    def predict(image, memory):
        with torch.no_grad():
            outputs, rw = model(image, memory, return_weights=True)
        return torch.argmax(outputs, 1).item(), rw[0]

    #saving stuff
    dir_save = "../images/mem_images/"+dataset_name+"/"+modality+"/" + checkpoint['model_name'] + "/"
    dir_save_class = dir_save + "correct_class/"
    dir_save_random = dir_save + "random_search/"
    for d in [dir_save, dir_save_class, dir_save_random]:
        if not os.path.isdir(d):
            os.makedirs(d)

    def get_image(image, revert_norm=True):
        if revert_norm:
            im = undo_normalization(image)
        else:
            im = image
        im = im.squeeze().cpu().detach().numpy()
        transformed_im = np.transpose(im, (1, 2, 0))
        return transformed_im

    def save_figure(save_path, input_image, memory, sorted_idx, rw, title):
        used = sorted_idx[rw[sorted_idx] > 0]
        reduced_mem = undo_normalization(memory[used])
        npimg = torchvision.utils.make_grid(reduced_mem,nrow=4).cpu().numpy()
        fig = plt.figure(figsize=(2, 4),dpi=300)
        fig.add_subplot(2, 1, 1)
        plt.imshow((get_image(input_image)* 255).astype(np.uint8),interpolation='nearest', aspect='equal')
        plt.title(title)
        plt.axis('off')
        fig.add_subplot(2, 1, 2)
        plt.imshow((np.transpose(npimg, (1,2,0))* 255).astype(np.uint8),interpolation='nearest', aspect='equal')
        plt.title('Used Samples')
        plt.axis('off')
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close()

    min_target = min(target_indices) if target_indices else None
    max_target = max(target_indices) if target_indices else None

    log_rows = []
    abs_idx = 0
    for batch_idx, (images, labels) in enumerate(test_loader):
        print("Batch:{}/{}".format(batch_idx, len(test_loader)), end='\r')
        if target_indices is not None:
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

        # compute output
        outputs,rw = model(images,memory,return_weights=True)
        _, predictions = torch.max(outputs, 1)

        # compute memory outputs
        memory_sorted_index = torch.argsort(rw,dim=1,descending=True)
        for ind in range(len(images)):
            true_label_idx = labels[ind].item()
            pred_label_idx = predictions[ind].item()
            if pred_label_idx == true_label_idx or (target_indices is not None and abs_idx not in target_indices):
                abs_idx += 1
                continue

            input_selected = images[ind].unsqueeze(0)
            true_class_name = name_classes[true_label_idx]
            pred_class_name = name_classes[pred_label_idx]

            # original prediction/explanation image
            save_figure(
                dir_save+'{}_true-{}_pred-{}.png'.format(abs_idx, true_class_name, pred_class_name),
                input_selected, memory, memory_sorted_index[ind], rw[ind],
                'Idx:{} True:{}\nPred:{}'.format(abs_idx, true_class_name, pred_class_name))

            row = {'abs_idx': abs_idx, 'true_class': true_class_name, 'original_pred': pred_class_name,
                   'corrected_by_class_only': '', 'class_only_pred': '',
                   'corrected_by_random': '', 'random_trials_used': '', 'random_pred': '',
                   'winning_memory_class_histogram': ''}

            if target_indices is not None and FLAGS.memory_mode in ('correct_class', 'all'):
                cc_memory, _ = sample_memory(class_to_positions[true_label_idx], FLAGS.mem_set_size)
                cc_pred, cc_rw = predict(input_selected, cc_memory)
                cc_pred_name = name_classes[cc_pred]
                row['corrected_by_class_only'] = cc_pred == true_label_idx
                row['class_only_pred'] = cc_pred_name
                save_figure(
                    dir_save_class+'{}_true-{}_pred-{}.png'.format(abs_idx, true_class_name, cc_pred_name),
                    input_selected, cc_memory, torch.argsort(cc_rw, descending=True), cc_rw,
                    'Idx:{} True:{}\nPred:{}'.format(abs_idx, true_class_name, cc_pred_name))

            if target_indices is not None and FLAGS.memory_mode in ('random_search', 'all'):
                all_positions = list(range(len(mem_images)))
                found, trial, rs_memory, rs_rw, rs_pred, rs_positions = False, 0, None, None, None, None
                for trial in range(1, FLAGS.max_random_trials + 1):
                    rs_memory, rs_positions = sample_memory(all_positions, FLAGS.mem_set_size)
                    rs_pred, rs_rw = predict(input_selected, rs_memory)
                    if rs_pred == true_label_idx:
                        found = True
                        break
                rs_pred_name = name_classes[rs_pred]
                row['corrected_by_random'] = found
                row['random_trials_used'] = trial
                row['random_pred'] = rs_pred_name
                if found:
                    hist = {}
                    for p in rs_positions:
                        c = name_classes[mem_labels[p].item()]
                        hist[c] = hist.get(c, 0) + 1
                    row['winning_memory_class_histogram'] = ';'.join('{}:{}'.format(k, v) for k, v in sorted(hist.items()))
                suffix = '{}_trials-{}'.format(abs_idx, trial) if found else '{}_UNRESOLVED_trials-{}'.format(abs_idx, trial)
                save_figure(
                    dir_save_random+'{}_true-{}_pred-{}.png'.format(suffix, true_class_name, rs_pred_name),
                    input_selected, rs_memory, torch.argsort(rs_rw, descending=True), rs_rw,
                    'Idx:{} True:{}\nPred:{}'.format(abs_idx, true_class_name, rs_pred_name))

            if target_indices is not None:
                log_rows.append(row)

            abs_idx += 1
        print('Generated batch {}/{}'.format(batch_idx,len(test_loader)),end='\r')

    if target_indices is not None and log_rows:
        with open(dir_save+'experiment_log.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)


def main(argv):

    run(FLAGS.path_model,FLAGS.dir_dataset)

if __name__ == '__main__':
  absl.app.run(main)
