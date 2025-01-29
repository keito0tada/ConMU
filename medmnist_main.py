import sys
import copy
import time

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import utils, arg_parser, ConMU, evaluation_metrics
from data_initialization import data_init, medmnist_data_init

sys.path.append(('../'))
sys.path.append(('../../'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def model_training(model_copy, train_loader, test_loader, args, original=True):
    model_copy = model_copy.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    if original:
        epochs = args.epochs
        lr = args.lr
    else:
        epochs = args.retrain_epoch
        lr = args.retrain_lr
    # optimizer = torch.optim.SGD(model_copy.parameters(), lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model_copy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    utils.train_model(model=model_copy,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      criterion=criterion,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      epochs=epochs,
                      device=device)


def main(data_flag, rate, index, retain_filter_up=0.3, retain_filter_lower=0.17, forget_filter_up=3.0, forget_filter_lower=3.0):
    target_model_path = f'/data1/keito/bachelor/model/{data_flag}/target/best_model.pth'
    retrain_model_path = f'/data1/keito/bachelor/model/{data_flag}/retrain_{rate}_{index}/best_model.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = arg_parser.parse_args()
    args.incompetent_epoch = 5
    args.further_train_epoch = 20
    args.further_train_lr = 0.01
    args.kl_weight = 0.5
    args.forget_percentage = rate
    args.retain_filter_up = retain_filter_up
    args.retain_filter_lower = retain_filter_lower
    args.forget_filter_up = forget_filter_up
    args.forget_filter_lower = forget_filter_lower
    args.batch_size = 32

    model, incompetent_model, train_loader, test_loader, forget_loader, retain_loader, total_unlearn_time = medmnist_data_init(data_flag, rate, index, args.lr, args.momentum, args.weight_decay, args.gpu, batch_size=args.batch_size, incompetent_epoch=args.incompetent_epoch)

    # ======================================== Original Model ========================================
    print('======================================== Original Model ========================================')
    # start_time = time.time()
    model_copy = copy.deepcopy(model)
    # model_training(model_copy, train_loader, test_loader, args, original=True)
    model_copy.load_state_dict(torch.load(target_model_path, device))
    # end_time = time.time()
    # original_model_training_time = end_time - start_time
    # print("model training (original_model) time: ", original_model_training_time)
    # original_evaluate_result = evaluation_metrics.MIA_Accuracy(model=model_copy,
    #                                                            forget_loader=forget_loader,
    #                                                            retain_loader=retain_loader,
    #                                                            test_loader=test_loader,
    #                                                            device=device,
    #                                                            total_unlearn_time=original_model_training_time,
    #                                                            args=args)
    # utils.save_checkpoint({
    #     'state_dict': model_copy.state_dict(),
    # }, save_path=args.save_dir, filename='_original_model_checkpoint.pth.tar')
    # print("original_evaluate_result: ", original_evaluate_result)

    # ======================================== ConMU ========================================
    print('======================================== ConMU ========================================')

    model_further_train = copy.deepcopy(model)
    # model_checkpoint = utils.load_checkpoint(device, args.save_dir, filename='_original_model_checkpoint.pth.tar')

    # model_further_train.load_state_dict(model_checkpoint["state_dict"])
    model_further_train.load_state_dict(torch.load(target_model_path, device))

    evaluation_result = ConMU.medmnist_further_train(model=model_further_train,
                                            incompetent_model=incompetent_model,
                                            test_loader=test_loader,
                                            retain_loader=retain_loader,
                                            forget_loader=forget_loader,
                                            device=device,
                                            unlearning_time=total_unlearn_time,
                                            save_dir=f'/data1/keito/bachelor/model/{data_flag}/retrain_{rate}_{index}',
                                            save_epochs=[4],
                                            args=args)
    print("ConMU further train result: ", evaluation_result)

    # ======================================== Retrain ========================================
    # start_time = time.time()
    # model_copy_retrain = copy.deepcopy(model)
    # # model_training(model_copy_retrain, retain_loader, test_loader, args, original=False)
    # model_copy_retrain.load_state_dict(torch.load(retrain_model_path, device))
    # end_time = time.time()
    # retrain_model_training_time = end_time - start_time
    # print("model training (retraining) time: ", retrain_model_training_time)
    # retrain_evaluate_result = evaluation_metrics.MIA_Accuracy(model=model_copy_retrain,
    #                                                           forget_loader=forget_loader,
    #                                                           retain_loader=retain_loader,
    #                                                           test_loader=test_loader,
    #                                                           device=device,
    #                                                           total_unlearn_time=retrain_model_training_time,
    #                                                           args=args)
    # utils.save_checkpoint({
    #     'state_dict': model_copy_retrain.state_dict(),
    # }, save_path=args.save_dir, filename='_retrain_model_checkpoint.pth.tar')
    # print("retrain_evaluate_result: ", retrain_evaluate_result)


if __name__ == '__main__':
    print('conmu')
    main('tissuemnist', 0.1, 0, 0.5, 0.5)
    # main('tissuemnist', 0.1, 0, 1.0, 1.0)
    # main('tissuemnist', 0.1, 0, 2.0, 2.0)
    # main('pathmnist', 0.1, 0)
    # main('pathmnist', 0.3, 0)
    # main('pathmnist', 0.5, 0)
    # main('octmnist', 0.1, 0)
    # main('octmnist', 0.3, 0)
    # main('octmnist', 0.5, 0)
    # main('tissuemnist', 0.1, 0)
    # main('tissuemnist', 0.3, 0)
    # main('tissuemnist', 0.5, 0)
