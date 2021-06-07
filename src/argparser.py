import argparse


def parse_train_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Model taining')

    required_group = parser.add_argument_group('required arguments')
    optional_group = parser.add_argument_group('optional arguments')
    store_group = parser.add_mutually_exclusive_group(required=True)
    continue_group = parser.add_mutually_exclusive_group()

    required_group.add_argument('--model-name', type=str, help='name of new model', required=True)
    required_group.add_argument('--targets', nargs='+', type=str, help='targets', required=True)
    optional_group.add_argument('--weights', nargs='+', type=float, help='wheights for targets')

    continue_group.add_argument('--resume', dest='resume', action='store_true', help='load last model weight by model-name and continue training')
    continue_group.add_argument('--fine-tune', dest='fine_tune', action='store_true', help='load last model weight by model-name and continue training without overwriting')

    store_group.add_argument('--store-best-weights', dest='store_best_weights', action='store_true', help='store weight only if it bet best accuracy score')
    store_group.add_argument('--store-every-weight', dest='store_every_weight', action='store_true', help='store every weight')

    optional_group.add_argument('--augment-rate', default=0., type=float, help='token augmentation probability')
    optional_group.add_argument('--augment-type', default='all', type=str, help='which augmentation to use')
    optional_group.add_argument('--sub-style', default='unk', type=str, help='replacement strategy for substitution augment')
    optional_group.add_argument('--alpha-sub', default=0.4, type=float, help='augmentation rate for substitution')
    optional_group.add_argument('--alpha-del', default=0.4, type=float, help='augmentation rate for deletion')

    optional_group.add_argument('--cuda', dest='cuda', action='store_true', help='use cuda if available')
    optional_group.add_argument('--seed', default=0, type=int, help='random seed')
    optional_group.add_argument('--pretrained-model', default='DeepPavlov/rubert-base-cased-sentence', type=str, help='pretrained language model')
    optional_group.add_argument('--freeze-pretrained', dest='freeze_pretrained', action='store_true', help='freeze pretrained model layers')
    optional_group.add_argument('--lstm-dim', default=-1, type=int, help='hidden dimension in LSTM layer, if -1 is set equal to hidden dimension in language model')
    required_group.add_argument('--train-data', nargs='+', type=str, help='path to train/dev/test datasets', required=True,)
    required_group.add_argument('--val-data', nargs='+', type=str, help='path to train/dev/test datasets', required=True)
    optional_group.add_argument('--test-data', nargs='+', type=str, help='path to train/dev/test datasets')
    optional_group.add_argument('--batch-size', default=8, type=int, help='batch size (default: 8)')
    optional_group.add_argument('--sequence-length', default=256, type=int, help='sequence length to use when preparing dataset (default 256)')
    optional_group.add_argument('--lr', default=5e-6, type=float, help='learning rate')
    optional_group.add_argument('--decay', default=0, type=float, help='weight decay (default: 0)')
    optional_group.add_argument('--gradient-clip', default=-1, type=float, help='gradient clipping (default: -1 i.e., none)')
    optional_group.add_argument('--epoch', default=10, type=int, help='total epochs (default: 10)')
    optional_group.add_argument('--labml', dest='labml', action='store_true', help='use labml library')
    optional_group.add_argument('--save-dir', default='models/', type=str, help='model and log save directory')

    parser.set_defaults(resume=False, fine_tune=False, cuda=False, freeze_pretrained=False, labml=False)

    args = parser.parse_args()

    args.targets = {t: i for i, t in enumerate(args.targets)}
    return args


if __name__ == "__main__":
    args = parse_train_arguments()
    print(args.train_data)
    print(args.val_data)
    print(bool(args.test_data))
    print(args.store_best_weights)
    print(args.store_every_weight)
