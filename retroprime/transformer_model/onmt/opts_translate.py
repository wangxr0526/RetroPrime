class OPT_TRANSLATE:
    def __init__(self, models = ['experiments/checkpoints/USPTO-50K/transformer_16w-25w.pt'], src = 'data/USPTO-50K/src-test.txt', reverse = False):
        self.alpha=0.0
        self.attn_debug=False
        self.batch_size=64
        self.beam_size=50
        self.beta=-0.0
        self.block_ngram_repeat=0
        self.coverage_penalty='none'
        self.data_type='text'
        self.dump_beam=''
        self.dynamic_dict=False
        self.fast=False
        self.gpu=0
        self.ignore_when_blocking=[]
        self.image_channel_size=3
        self.length_penalty='none'
        self.log_file=''
        self.log_probs=False
        self.mask_from=''
        self.max_length=200
        self.max_sent_length=None
        self.min_length=0
        self.models=models
        self.n_best=50
        self.output='experiments/results/transformer.txt'
        self.replace_unk=True
        self.report_bleu=False
        self.report_rouge=False
        self.sample_rate=16000
        self.share_vocab=False
        self.src=src
        self.src_dir=''
        self.stepwise_penalty=False
        self.tgt=None
        self.verbose=False
        self.window='hamming'
        self.window_size=0.02
        self.window_stride=0.01

        if reverse:
            models = ['experiments/checkpoints/USPTO-50K-reverse/transformer_16w-25w.pt']