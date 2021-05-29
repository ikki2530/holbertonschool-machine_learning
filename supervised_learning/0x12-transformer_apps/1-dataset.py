#!/usr/bin/env python3
"""Loads and preps a dataset for machine translation"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Loads and preps a dataset for machine translation"""
    def __init__(self):
        """Constructor"""
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        tokenizer_pt, tokenizer_en = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en

    def tokenize_dataset(self, data):
        """Method that creates sub-word tokenizers for our dataset"""
        subw = tfds.features.text.SubwordTextEncoder.build_from_corpus
        tokenizer_en = subw((en.numpy() for pt, en in data),
                            target_vocab_size=2**15)
        tokenizer_pt = subw((pt.numpy() for pt, en in data),
                            target_vocab_size=2**15)
        return (tokenizer_pt, tokenizer_en)

    def encode(self, pt, en):
        """Method that encodes a translation into tokens
        Returns:
        - pt_tokens is a np.ndarray containing the Portuguese tokens
        - en_tokens is a np.ndarray. containing the English tokens
        """
        pt = [self.tokenizer_pt.vocab_size] +\
            self.tokenizer_pt.encode(pt.numpy()) +\
            [self.tokenizer_pt.vocab_size+1]
        en = [self.tokenizer_en.vocab_size] +\
            self.tokenizer_en.encode(en.numpy()) +\
            [self.tokenizer_en.vocab_size+1]
        return (pt, en)