import argparse
from src.trainer import Trainer

def init_argparser():
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument('--train_path', help='Training data path')
    parser.add_argument('--dev_path', help='Development data path')
    parser.add_argument('--test_path', help='Test data path')
    parser.add_argument('--unlabeled_path', help='Unlabeled data path')
    parser.add_argument('--save_to', default='models/', help='Path to model directory')

    parser.add_argument('--batch_size', type=int,
                        help='Batch size', default=32)
    parser.add_argument('--learning_rate', type=float,
                        help='Learning rate, recommended settings for adam=0.001', default=1e-3)
    parser.add_argument('--weight_decay', type=float,
                        help='Weight decay', default=5e-3)
    parser.add_argument('--dropout_labeled', type=float,
                        help='Dropout probability during supervised regime', default=0.5)
    parser.add_argument('--dropout_unlabeled', type=float,
                        help='Dropout probability during unsupervised regime', default=0.8)
    parser.add_argument('--semi_supervised', dest='semi_supervised', action='store_true',
                        help='Indicates to use cvt or not')
    parser.add_argument('--no-semi_supervised', dest='semi_supervised', action='store_false')
    parser.set_defaults(semi_supervised=True)
    parser.add_argument('--char_compose_method', type=str,
                        help='Whether to use cnn or rnn to compose char embedding',
                        default='cnn', choices=['cnn','rnn'])
    parser.add_argument('--pretrained_embeddings_path', default=None,
                        help='Path to pretrained embeddings')
    parser.add_argument('--use_crf', dest='use_crf', action='store_true',
                        help='Indicates to use cvt or not')
    parser.add_argument('--no_crf', dest='use_crf', action='store_false')
    parser.set_defaults(use_crf=True)

    parser.add_argument('--save_every', type=int,
                        help='Every how many batches the model should be saved', default=2000)
    parser.add_argument('--print_every', type=int,
                        help='Every how many batches to print results', default=301)

    parser.add_argument('--resume-training', action='store_true',
                        help='Indicates if training has to be resumed from the latest checkpoint')

    return parser

def run():
    parser = init_argparser()
    opt = parser.parse_args()
    trainer = Trainer(train_path=opt.train_path,
                      dev_path=opt.dev_path,
                      test_path=opt.test_path,
                      unlabeled_path=opt.unlabeled_path,
                      save_to=opt.save_to,
                      semi_supervised=opt.semi_supervised,
                      batch_size=opt.batch_size,
                      learning_rate=opt.learning_rate,
                      weight_decay=opt.weight_decay,
                      dropout_lab=opt.dropout_labeled,
                      dropout_unlab=opt.dropout_unlabeled,
                      char_compose_method=opt.char_compose_method,
                      pretrained_embeddings_path=opt.pretrained_embeddings_path,
                      use_crf=opt.use_crf,
                      save_every=opt.save_every,
                      print_every=opt.print_every,
                      resume_training=opt.resume_training,
                      )
    trainer.run()

if __name__ == "__main__":
    run()
