import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score, accuracy_score, hamming_loss
from sklearn.model_selection import KFold
from collections import OrderedDict
import json
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

def get_bag_feats(csv_file_df, args):

    feats_csv_path = csv_file_df.iloc[0]
    
    if args.mode == 'No Feature':
        image_folder = feats_csv_path.replace('.csv', '')
        image_folder = os.path.join('datasets/OHTS_1', '/'.join(image_folder.split('/')[1:]))
        
        image_files = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        feats = []
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                feats.append(img_tensor.unsqueeze(0))
            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
                continue
        
        if not feats:
            raise ValueError(f"No valid images found in {image_folder}")
            
        feats = torch.cat(feats, dim=0)  # Shape: [N, 3, 224, 224]
        
    else:
        df = pd.read_csv(feats_csv_path)
        feats = shuffle(df).reset_index(drop=True)
        feats = feats.to_numpy()
    
    label = np.zeros(args.num_classes)
    if args.num_classes == 1:
        label[0] = csv_file_df['label']
    else:
        if int(csv_file_df['label']) <= (len(label) - 1):
            label[int(csv_file_df['label'])] = 1
        
    return label, feats, feats_csv_path

def generate_pt_files(args, df):

    temp_train_dir = "temp_train"
    if os.path.exists(temp_train_dir):
        import shutil
        shutil.rmtree(temp_train_dir, ignore_errors=True)
    os.makedirs(temp_train_dir, exist_ok=True)
    
    print('Creating intermediate training files.')
    for i in tqdm(range(len(df))):
        try:
            label, feats, feats_csv_path = get_bag_feats(df.iloc[i], args)
            
            if args.mode == 'No Feature':
                # feats shape: [N, 3, 224, 224]
                bag_feats = feats  
                
                bag_label = torch.tensor(label, dtype=torch.long)
                
                n_patches = bag_feats.size(0)
                label_repeated = bag_label.repeat(n_patches,1)  
                
                data_dict = {
                    'features': bag_feats,  # [N, 3, 224, 224]
                    'label': label_repeated  # [N]
                }
                
            else:
                bag_feats = torch.tensor(np.array(feats), dtype=torch.float32)
                bag_label = torch.tensor(label, dtype=torch.float32)
                
                data_dict = {
                    'features': bag_feats,
                    'label': bag_label
                }
            
            pt_file_path = os.path.join(temp_train_dir, 
                                      os.path.splitext(feats_csv_path)[0].split(os.sep)[-1] + ".pt")
            with open(pt_file_path, 'wb') as f:
                torch.save(data_dict, f)
                f.flush() 
                os.fsync(f.fileno())  
            
        except Exception as e:
            print(f"Error processing file {i}: {str(e)}")
            continue


def train(args, train_df, milnet, criterion, optimizer):
    milnet.train()
    dirs = shuffle(train_df)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    Tensor = torch.cuda.FloatTensor
    
    for i, item in enumerate(dirs):
        optimizer.zero_grad()
        
        data_dict = torch.load(item, map_location='cuda:0')
        bag_feats = data_dict['features']
        bag_label = data_dict['label'].unsqueeze(0)
        
        if args.mode == 'No Feature':
            batch_size = bag_feats.size(0)
            bag_feats = bag_feats.view(batch_size, 3, 224, 224)
        
        ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
        max_prediction, _ = torch.max(ins_prediction, 0)


        bag_loss = criterion(bag_prediction.view(1, -1), bag_label)
        max_loss = criterion(max_prediction.view(1, -1), torch.zeros_like(max_prediction.view(1, -1)))
        loss = 1.5 *  bag_loss + max_loss
        
        _, predicted = torch.max(bag_prediction.data, 1)
        acc = (predicted == bag_label).sum().item() / bag_label.size(0) * 100
        
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), 1)
        acc_meter.update(acc, 1)  
        
        sys.stdout.write('\r Training bag [%d/%d] loss: %.4f, acc: %.3f%%' % 
                        (i, len(train_df), loss.item(), acc))
    
    print('\nTrain set: Average loss: {:.4f}, Average acc: {:.3f}%\n'.format(
        loss_meter.avg, acc_meter.avg))
        
    return loss_meter.avg, acc_meter.avg

def test(args, test_df, milnet, criterion, return_predictions=False):
    milnet.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()  
    Tensor = torch.cuda.FloatTensor
    
    test_labels = []
    test_predictions = []
    
    with torch.no_grad():
        for i, item in enumerate(test_df):
            data_dict = torch.load(item, map_location='cuda:0')
            bag_feats = data_dict['features']
            bag_label = data_dict['label'].unsqueeze(0)
            
            if args.mode == 'No Feature':
                batch_size = bag_feats.size(0)
                bag_feats = bag_feats.view(batch_size, 3, 224, 224)
            
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)

            bag_loss = criterion(bag_prediction.view(1, -1), bag_label)
            max_loss = criterion(max_prediction.view(1, -1), torch.zeros_like(max_prediction.view(1, -1)))
            loss = 1.5 * bag_loss + max_loss
            
            _, predicted = torch.max(bag_prediction, 1)
            bag_label_arg = torch.argmax(bag_label, dim=1)

            acc = (predicted == bag_label_arg).sum().item() / bag_label.size(0) * 100
            
            loss_meter.update(loss.item(), 1)
            acc_meter.update(acc, 1)


            
            if return_predictions:
                test_labels.extend([bag_label.squeeze().cpu().numpy()])
                test_predictions.extend([torch.softmax(bag_prediction, dim=1).squeeze().cpu().numpy()])
            

    
    print('\nTest set: Average loss: {:.4f}, Average acc: {:.3f}%\n'.format(
        loss_meter.avg, acc_meter.avg))
    
    if return_predictions:
        return loss_meter.avg, acc_meter.avg, test_predictions, test_labels
    return loss_meter.avg, acc_meter.avg

def dropout_patches(feats, p):
    num_rows = feats.size(0)
    num_rows_to_select = int(num_rows * p)
    random_indices = torch.randperm(num_rows)[:num_rows_to_select]
    selected_rows = feats[random_indices]
    return selected_rows

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def calculate_auc(labels, predictions):
    """
    Calculate AUC for multi-class predictions
    """
    # Convert to numpy arrays if they're not already
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    # For binary classification
    if predictions.shape[1] == 2:
        return roc_auc_score(labels, predictions[:, 1])
    
    # For multi-class, calculate macro AUC
    return roc_auc_score(labels, predictions, multi_class='ovr', average='macro')


def save_model(args, fold, run, save_path, model, thresholds_optimal):
    # Construct the filename including the fold number
    save_name = os.path.join(save_path, f'fold_{fold}_{run+1}.pth')
    torch.save(model.state_dict(), save_name)
    print_save_message(args, save_name, thresholds_optimal)
    file_name = os.path.join(save_path, f'fold_{fold}_{run+1}.json')
    with open(file_name, 'w') as f:
        json.dump([float(x) for x in thresholds_optimal], f)

def print_save_message(args, save_name, thresholds_optimal):
    print('Best model saved at: ' + save_name)
    print('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))

def main():
    parser = argparse.ArgumentParser(description='Train on patch features learned by MAE')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=768, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate [0.0001]')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of total training epochs [100]')
    parser.add_argument('--stop_epochs', default=10, type=int, help='Skip remaining epochs if training has not improved after N epochs [10]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight decay [1e-3]')
    parser.add_argument('--dataset', default='GRAPE', type=str, help='Dataset folder name')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='dure', type=str, help='model name')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=False, help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--eval_scheme', default='5-fold-cv', type=str, help='Evaluation scheme [5-fold-cv | 5-fold-cv-standalone-test | 5-time-train+valid+test ]')
    parser.add_argument('--mode', default='Feature', type=str, choices=['Feature', 'No Feature'],help='Processing mode [Feature | No Feature]')

    
    args = parser.parse_args()
    print(args.eval_scheme)

    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    import dure as mil

    def apply_sparse_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def init_model(args):
        if args.mode == 'No Feature':
            i_classifier = mil.ConcatLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        else:
            i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()
        milnet.apply(lambda m: apply_sparse_init(m))
        print(f"Total trainable parameters: {count_parameters(milnet):,}")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.00005)
        return milnet, criterion, optimizer, scheduler
    


    def save_model(args, fold, run, save_path, milnet, acc1, acc5):
        model_path = os.path.join(save_path, f'fold{fold}_run{run}_acc{acc1:.3f}_acc5{acc5:.3f}.pth')
        if isinstance(milnet, torch.nn.DataParallel):
            torch.save(milnet.module.state_dict(), model_path)
        else:
            torch.save(milnet.state_dict(), model_path)
        
    if args.dataset == 'OHTS':
        bags_csv = os.path.join(args.dataset+'_1.csv')
    elif args.dataset == 'GRAPE':
        bags_csv = os.path.join(args.dataset+'_PLR3.csv')

    generate_pt_files(args, pd.read_csv(bags_csv))
 
    if args.eval_scheme == '5-fold-cv':
        bags_path = glob.glob('temp_train/*.pt')
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_results = []

        save_path = os.path.join('weights', datetime.date.today().strftime("%Y%m%d"))
        os.makedirs(save_path, exist_ok=True)
        run = len(glob.glob(os.path.join(save_path, '*.pth')))

        for fold, (train_index, test_index) in enumerate(kf.split(bags_path)):
            print(f"Starting CV fold {fold}.")
            milnet, criterion, optimizer, scheduler = init_model(args)
            train_path = [bags_path[i] for i in train_index]
            test_path = [bags_path[i] for i in test_index]
            fold_best_score = 0
            best_acc = 0
            best_auc = 0
            counter = 0

            for epoch in range(1, args.num_epochs+1):
                counter += 1
                train_loss, train_acc = train(args, train_path, milnet, criterion, optimizer)
                test_loss, test_acc, test_predictions, test_labels = test(args, test_path, milnet, criterion, return_predictions=True)
                test_predictions = np.vstack(test_predictions).reshape(-1, 1)
                test_labels = np.vstack(test_labels).reshape(-1, 1)
                test_auc = calculate_auc(test_labels, test_predictions)
                
                print('Epoch: {}/{}, Train Loss: {:.4f}, Train Acc: {:.3f}%'.format(
                    epoch, args.num_epochs, train_loss, train_acc))
                print('Epoch: {}/{}, Test Loss: {:.4f}, Test Acc: {:.3f}%, Test AUC: {:.3f}'.format(
                    epoch, args.num_epochs, test_loss, test_acc, test_auc))
                
                scheduler.step()

                current_score = test_acc # Convert AUC to percentage scale
                if current_score > fold_best_score:
                    counter = 0
                    fold_best_score = current_score
                    best_acc = test_acc
                    best_auc = test_auc
                    save_model(args, fold, run, save_path, milnet, best_acc, best_auc)
                if counter > args.stop_epochs: 
                    break
                    
            fold_results.append((best_acc, best_auc))
            
        # Calculate mean and std of metrics
        mean_acc = np.mean([result[0] for result in fold_results])
        mean_auc = np.mean([result[1] for result in fold_results])
        std_acc = np.std([result[0] for result in fold_results])
        std_auc = np.std([result[1] for result in fold_results])
        
        print(f"Final results:")
        print(f"Accuracy: {mean_acc:.3f}% (±{std_acc:.3f}%)")
        print(f"AUC: {mean_auc:.3f} (±{std_auc:.3f})")

if __name__ == '__main__':
    main()
