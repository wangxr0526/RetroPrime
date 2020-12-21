class OPT_PREPROCESS:
    def __init__(self, train_src = 'data/USPTO-50K/src-train.txt', train_tgt ='data/USPTO-50K/tgt-train.txt', \
        save_data = 'data/USPTO-50K/USPTO-50K', valid_src = 'data/USPTO-50K/src-val.txt', valid_tgt = 'data/USPTO-50K/tgt-val.txt', reverse = False):
        self.data_type='text'
        self.dynamic_dict=False
        self.features_vocabs_prefix=''
        self.image_channel_size=3
        self.log_file=''
        self.lower=False
        self.max_shard_size=0
        self.report_every=100000
        self.sample_rate=16000
        self.save_data=save_data
        self.seed=3435
        self.shard_size=1000000
        self.share_vocab=True
        self.shuffle=1
        self.src_dir=''
        self.src_seq_length=1000
        self.src_seq_length_trunc=0
        self.src_vocab=''
        self.src_vocab_size=1000
        self.src_words_min_frequency=0
        self.tgt_seq_length=1000
        self.tgt_seq_length_trunc=0
        self.tgt_vocab=''
        self.tgt_vocab_size=1000
        self.tgt_words_min_frequency=0
        
        self.window='hamming'
        self.window_size=0.02
        self.window_stride=0.01

        if reverse:
            train_src = 'data/USPTO-50K-reverse/src-train.txt'
            train_tgt = 'data/USPTO-50K-reverse/tgt-train.txt'
            save_data = 'data/USPTO-50K-reverse/USPTO-50K-reverse'
            valid_src = 'data/USPTO-50K-reverse/src-val.txt'
            valid_tgt = 'data/USPTO-50K-reverse/tgt-val.txt'
        
        self.train_src=train_src
        self.train_tgt=train_tgt
        self.valid_src=valid_src
        self.valid_tgt=valid_tgt
        self.save_data = save_data
        
