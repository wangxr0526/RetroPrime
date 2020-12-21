from onmt.opts_preprocess import OPT_PREPROCESS 
from onmt.opts_train import OPT_TRAIN 
from onmt.opts_translate import OPT_TRANSLATE 

from preprocess import main as main_preprocess
from train import main as main_train
from translate import main as main_translate

opt_preprocess = OPT_PREPROCESS(reverse=True)
opt_train = OPT_TRAIN(reverse=True)
opt_translate = OPT_TRANSLATE(reverse=True)


if __name__=='__main__':
	main_preprocess(opt_preprocess)
	main_train(opt_train)
	main_translate(opt_translate)
