
import argparse

parser = argparse.ArgumentParser(description='Available arguments for models benchmarking')

parser.add_argument('model', help='defines the model name to be executed for benchmark')
# parser.add_argument('--img_size', default=[224, 224], type=int, nargs=2, help='image size as width height e.g 128 128')
parser.add_argument('--batch_size', default=1, type=int, help='batch size for input data')
parser.add_argument('--device', default=None, help='device on which model will evaluated e.g. CPU:0 or GPU:0 default is prefered device GPU>CPU')
parser.add_argument('--wandb', default=False, type=bool, help='Optional: True if want to add all the results in wandb (weights and biases)')
parser.add_argument('--project_name', default='benchmarks', help='optional name for wandb project')
parser.add_argument('--optional', default=None,  nargs='+', type=str, help='Pass all the optional arguments here')

args = parser.parse_args()
kwargs = {}
if args.optional:
    kwargs = dict()
    for _kwarg in args.optional:
        key, val = _kwarg.split('=')
        kwargs.update({key: eval(val.replace('[', '(').replace(']', ')').replace('~', '"'))})

import rbm

if __name__=="__main__":
    benchmarker = rbm.utils.Benchmark(model=args.model, batch_size=args.batch_size,
                                      device=args.device, **kwargs)
    benchmarks = benchmarker.execute(wandb=args.wandb, project_name=args.project_name)
    print('*************************** Benchmark Results ***************************')
    benchmarks.update(benchmarks['benchmark'])
    _ = benchmarks.pop('benchmark')
    benchmarks = [f'{key}: {val}' for key, val in benchmarks.items()]
    print('\n'.join(benchmarks))


