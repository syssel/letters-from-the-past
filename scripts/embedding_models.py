
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
import pandas as pd
import os


DIR = "data/experiment/"
OUTDIR = "ppmi_embeddings/"

RUN = """python external/distributional_learning/code/VectorModelBuilder.py {corpus} --count_method ngram --n {n} --weighting ppmi --outfile {out} --outdir {outdir}
    """
VOCAB = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'æ', 'ø', 'å',
         'th', "kh", "ph"]
RANDOMSTATE = 64

def build_model(NAME, CORPORA, MODELS_OUT, OUTDIR, N, PREDEFINED_CONTEXTS=None, PREDEFINED_VOCAB=VOCAB):

    # Create model
    print()
    print()
    print("~~~ Creating PPMI/SVD matrices for '{}' from the corpora: [{}]".format(NAME, "\n\t"+"\n\t".join(CORPORA)+"\n\t"))
    print()
    print("~ PPMI")
    print()
    for corpus, model in zip(CORPORA, MODELS_OUT):
        print("Creating '{}' from '{}'".format(model, corpus))
        os.system(RUN.format(corpus=corpus, n=N, out=model, outdir=OUTDIR+NAME))

    E, C, V = [], [], []
    all_vocab = set()
    all_contexts = set()

    print("~ Rewriting output format")
    print()
    # Rewrite format and compare vocabulary and contexts
    for model in MODELS_OUT:
        model_name = OUTDIR+NAME+"/"+model

        # Load vocabulary
        with open(model_name+".sounds", "r") as f:
            vocab = f.readline().strip().split() 
        os.remove(model_name+".sounds")

        # Load contexts
        with open(model_name+".contexts", "r") as f:
            contexts = f.readline().strip().split() 
        os.remove(model_name+".contexts")

        # Load embedding table
        embedding_table = np.loadtxt(model_name+".data")
        os.remove(model_name+".data")

        E.append(embedding_table)
        V.append(vocab)
        C.append(contexts)

        all_vocab.update(vocab)
        all_contexts.update(contexts)
    
    common_vocabulary = set(V[0])
    for v in V:
        common_vocabulary = common_vocabulary & set(v)

    common_contexts = set(C[0])
    for c in C:
        common_contexts = common_contexts & set(c)

    if PREDEFINED_VOCAB:
        common_vocabulary = PREDEFINED_VOCAB

    if PREDEFINED_CONTEXTS:
        common_contexts = PREDEFINED_CONTEXTS

    print("~ Reducing to common dimensions")
    print("\tVocabulary size from", len(all_vocab), "to", len(common_vocabulary))
    print("\tContexts from", len(all_contexts), "to", len(common_contexts))
    print()
    for model, embedding_table, vocab, contexts in zip(MODELS_OUT, E, V, C):
        model_name = OUTDIR+NAME+"/"+model
        print("@", model_name)

        # Collect into one file
        df = pd.DataFrame(embedding_table, index=vocab, columns=contexts)
        
        df = df.reindex(common_vocabulary, axis=0, fill_value=0)

        try:
            df = df.reindex(common_contexts, axis=1, fill_value=0)
        except ValueError as e:
            if str(e) == 'cannot reindex from a duplicate axis':
                print("Found duplicate values in columns:", df.columns[df.columns.duplicated()])
                print("Removing first occurrences...")
                df = df.loc[:,~df.columns.duplicated()]
                df = df.reindex(common_contexts, axis=1, fill_value=0)
            else:
                raise ValueError(e)

        df.sort_index(axis=0, inplace=True)
        df.sort_index(axis=1, inplace=True)

        df.to_csv(model_name+".emb", sep=",", index=True, index_label="CHAR", header=True)

    print("Done!")
    print()
    
    return common_contexts


# Parupa
N = 2
NAME = "parupa"
os.makedirs(OUTDIR+NAME, exist_ok=True)

CORPORA = [DIR+"parupa/parupa_{}.txt".format(i) for i in range(0,5)]
MODELS_OUT = [NAME+"_"+str(i) for i in range(1, len(CORPORA)+1)]

common_contexts = build_model(NAME, CORPORA, MODELS_OUT, OUTDIR, N)

# Parupa (control)
NAME = "parupa_control"
os.makedirs(OUTDIR+NAME, exist_ok=True)

CORPORA = [DIR+"parupa_control/parupa_{}.txt".format(i) for i in range(0,5)]
MODELS_OUT = [NAME+"_"+str(i) for i in range(1, len(CORPORA)+1)]

build_model(NAME, CORPORA, MODELS_OUT, OUTDIR, N, common_contexts)

# UD-Danish
N = 3
NAME = "UDDanish"
os.makedirs(OUTDIR+NAME, exist_ok=True)

CORPORA = [DIR+"UDDanish/UDDanish_{}.txt".format(i) for i in range(0,5)]
MODELS_OUT = [NAME+"_"+str(i) for i in range(1, len(CORPORA)+1)]

common_contexts = build_model(NAME, CORPORA, MODELS_OUT, OUTDIR, N)


# UD-Danish (control)
NAME = "UDDanish_control"
os.makedirs(OUTDIR+NAME, exist_ok=True)

CORPORA = [DIR+"UDDanish_control/UDDanish_{}.txt".format(i) for i in range(0,5)]
MODELS_OUT = [NAME+"_"+str(i) for i in range(1, len(CORPORA)+1)]

build_model(NAME, CORPORA, MODELS_OUT, OUTDIR, N, common_contexts)


# Danmarks Stednavne
N = 4
NAME = "danmarksstednavne"
os.makedirs(OUTDIR+NAME, exist_ok=True)

CORPORA = [DIR+NAME+"/epoch_{}/train.txt".format(i) for i in range(1,12)]
MODELS_OUT = [NAME+"_"+str(i) for i in range(1, len(CORPORA)+1)]

common_contexts = build_model(NAME, CORPORA, MODELS_OUT, OUTDIR, N)


# Danmarks Stednavne (control)
NAME = "danmarksstednavne_control"
os.makedirs(OUTDIR+NAME, exist_ok=True)

CORPORA = [DIR+NAME+"/epoch_{}/train.txt".format(i) for i in range(1,12)]
MODELS_OUT = [NAME+"_"+str(i) for i in range(1, len(CORPORA)+1)]

build_model(NAME, CORPORA, MODELS_OUT, OUTDIR, N, common_contexts)