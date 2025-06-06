import util
from experiment import train, test

if __name__=='__main__':
    # parse options and make directories
    opt = util.option.parse()
    # run and analyze experiment
    if not any([opt.train, opt.test, opt.analyze]): opt.train = opt.test = opt.analyze = True
    if opt.train: train(opt)
    if opt.test: test(opt)
    # if argv.analyze: analyze(argv)
    exit(0)
