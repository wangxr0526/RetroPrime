#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse

from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
import torch

use_cmd = True
if use_cmd:
    import onmt.opts as opts
else:
    from onmt.opts_translate import OPT_TRANSLATE


def main(opt):
    translator = build_translator(opt, report_score=True)
    translator.translate(src_path=opt.src,
                         tgt_path=opt.tgt,
                         src_dir=opt.src_dir,
                         batch_size=opt.batch_size,
                         attn_debug=opt.attn_debug)

# 
# 

if __name__ == "__main__":
    if use_cmd:
        parser = argparse.ArgumentParser(
            description='translate.py',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        onmt.opts.add_md_help_argument(parser)
        onmt.opts.translate_opts(parser)

        opt = parser.parse_args()

    else:
        opt = OPT_TRANSLATE()

    logger = init_logger(opt.log_file)
    main(opt)
