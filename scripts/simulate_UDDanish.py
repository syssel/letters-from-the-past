import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from external.soundchangesim.soundchange import Corpus, RuleBook, Rule, RuleException
from external.soundchangesim.soundchange.utils import apply_sound_change_ud_tokenlist, UD_iterator

import numpy as np
from functools import partial

# Define change rates
n_epochs = 5
change_rates = np.linspace(0, 1, n_epochs)

# Import data
train_path = "data/source/UDDanish/UD_Danish-DDT/da_ddt-ud-train.conllu"
train_corpus = Corpus(train_path, UD_iterator)
train_subcorpora = train_corpus.create_subcorpora(n_epochs, random_seed=42)


# Create output folder
out_directory = "data/experiment/UDDanish/"
out_directory_control = "data/experiment/UDDanish_control/"

os.makedirs(out_directory, exist_ok=True)
os.makedirs(out_directory_control, exist_ok=True)

import logging; logging.basicConfig(filename=out_directory+'run.log', level=logging.INFO)

# Define rules and exceptions
vowels = "[aeoieuæøå]"
source = "g"
target = "k"

rule_patterns = ((r"({V}){S}({V})", r"\1{T}\2"),
                 (r"({V}){S}$", r"\1{T}"),
                 (r"{S}t", r"{T}t"))

rules = [Rule(s.format(V=vowels, S=source), t.format(T=target))
            for (s, t) in rule_patterns]

#exception_patterns = ((r"^({W})$", ("og",), 1),)
#exceptions = [RuleException(s.format(W="|".join(words)), p)
#                for (s, words, p) in exception_patterns]
exceptions = []

sound_changes = RuleBook(rules, exceptions)

def no_sound_change(tokenlist):
    transformed = []
    for token in tokenlist:
        transformed.append(token.form.lower())
    return transformed

def to_file(tokenlist):
    return "\n".join([" ".join(list(token)) for token in tokenlist])+"\n"

# Transform corpus and write to dir
for i, (train_subcorpus, change_rate) in enumerate(zip(train_subcorpora, change_rates)):
    logging.info("~"*84)
    logging.info("~"*84)
    logging.info("Writing data for epoch: "+str(i+1))
    logging.info("Rules: {}".format([str(r) for r in rules]))
    logging.info("Exceptions: {}".format([str(e) for e in exceptions]))
    logging.info("Change rate: "+str(change_rate))

    transform = partial(apply_sound_change_ud_tokenlist, rb=sound_changes, p=change_rate)

    # With simulations
    train_subcorpus.write_to_file(out_directory+"/UDDanish_{}.txt".format(i), 
                            transform=transform, writer=to_file)

    # Without simulations
    train_subcorpus.write_to_file(out_directory_control+"/UDDanish_{}.txt".format(i), 
                            transform=no_sound_change, writer=to_file)
