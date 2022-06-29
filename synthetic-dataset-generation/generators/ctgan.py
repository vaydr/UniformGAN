#!/usr/bin/env python
# coding: utf-8

def import_model():
    from ctgan import CTGANSynthesizer 

def train(js, dataset, epochs=200):
    from ctgan import CTGANSynthesizer

    discrete_columns = js['discrete_columns']
    ctgan = CTGANSynthesizer(epochs=epochs)
    ctgan.fit(dataset, discrete_columns)
    return ctgan

def generate(js, model, sample_total=40000):
    samples = model.sample(sample_total)
    return samples