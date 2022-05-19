import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from external.soundchangesim.soundchange import Corpus, RuleBook, Rule
from external.soundchangesim.soundchange.utils import wordlist_iterator, apply_sound_change_wordlist

import numpy as np
from functools import partial

# Define change rates
n_epochs = 5
change_rates = np.linspace(0, 1, n_epochs)

# Import data
dpath = "data/source/parupa/noisy_parupa_0_0.txt"
paruba = Corpus(dpath, wordlist_iterator)
subcorpora = paruba.create_subcorpora(n_epochs, random_seed=42)

# Create output folder
out_directory = "data/experiment/parupa/"
os.makedirs(out_directory, exist_ok=True)
out_directory_control = "data/experiment/parupa_control/"
os.makedirs(out_directory_control, exist_ok=True)

import logging; logging.basicConfig(filename=out_directory+"run.log", level=logging.INFO)

# Define rules and exceptions
vowels = "[ui]"
source = "p"
target = "b"

rule_patterns = ((r" {S} ({V})", r" {T} \1"),)

rules = [Rule(s.format(V=vowels, S=source), t.format(T=target))
            for (s, t) in rule_patterns]

exceptions = []

sound_changes = RuleBook(rules, exceptions)

def to_file(word):
    return word+"\n"

for i, (corpus, change_rate) in enumerate(zip(subcorpora, change_rates)):
    logging.info("~"*84)
    logging.info("~"*84)
    logging.info("Writing data for epoch: "+str(i+1))
    logging.info("Rules: {}".format([str(r) for r in rules]))
    logging.info("Exceptions: {}".format([str(e) for e in exceptions]))
    logging.info("Change rate: "+str(change_rate))

    transform = partial(apply_sound_change_wordlist, rb=sound_changes, p=change_rate)
    corpus.write_to_file(out_directory+"parupa_{}.txt".format(i), 
                            transform=transform, writer=to_file)

    # Without simulations
    corpus.write_to_file(out_directory_control+"parupa_{}.txt".format(i), 
                            transform=None, writer=to_file)