#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:59:39 2022

Compute the Resnik distances between patient cases using Human Phenotype Ontology.
Takes a hp.obo file as catalog input
Takes a tab-delimited dataframe as input file. Col 1 is ID, col 2 is a comma-delimited string of HPO IDs.
Output is the output path for a pairwise distance matrix

@author: Bradley Bowles
"""

import pandas as pd
import numpy as np
import pronto
import argparse

# parse arguments
parser = argparse.ArgumentParser(description='Compute a Resnik Distance Matrix for a Patient HPO Terms.')
parser.add_argument('-c', '--catalog', type=str, metavar='', required=True, help=r'Path to a downloaded HP.obo file. See https://hpo.jax.org/app/download/ontology.')
parser.add_argument('-i', '--input', type=str, metavar='', required=True, help=r'Tab-delimited input file. Col1="ID", Col2="HPO".')
parser.add_argument('-o', '--output', type=str, metavar='', required=True, help=r'Output file path. Will right a tab-delimited distance matrix to this path with input IDs as rows and cols.')
args = parser.parse_args()


if __name__ == "__main__":

    # import HPO .obo catalog using pronto
    # Docs: https://pronto.readthedocs.io/en/latest/api.html
    hpo = pronto.Ontology(args.catalog)

        
    # import labelled patient data
    df = pd.read_csv(args.input, sep='\t')
    
    
    # create a try/except statement to apply in lambda statements for mapping HPO terms
    def map_ancs(term):
        try:
            return ( [i.id for i in hpo[term].rparents()] )
        except KeyError:
            return([])
        
    def obsolete_terms(term_list):
        if all([bool(map_ancs(i)) for i in term_list]):
            #No obsolete terms detected.
            return(False)
        else:
            # Obsolete terms detected!
            return(True)
           
    def resnik(HPO_list_1, HPO_list_2):
        
        # this function uses list comprehension to create a Resnik semantic similarity distance matrix
        # Specifically it generates the sim.max measure described in the following manuscript:
        # "A Recurrent Missense Variant in AP2M1 Impairs Clathrin-Mediated Endocytosis 
        # and Causes Developmental and Epileptic Encephalopathy"
    
        
        # compute max information score for each value in our HPO lists
        resnik = pd.DataFrame(
            [
                [max([IC[i] for i in ( set(map_ancs(i)) & set(map_ancs(j)) )]) for j in HPO_list_2]
                for i in HPO_list_1]
            )
                
        # Resnik matrix now contains max IC values for each intersection of common ancestors in input lists
        # calculate SimMax:
        # summing over the maximum of all rows and columns with appropriate normalization
        simmax = 0.5 * ( (resnik.max(axis=0).sum()) + resnik.max(axis=1).sum() )
        
        return simmax     
    
    # DATA IMPORTS AND PROCESSING
    
    # drop empty ID/HPO rows
    df.dropna(inplace=True)
    
    # explode HPO terms
    df.loc[:, 'HPO'] = df.HPO.str.split(',')
    df = df.explode('HPO')
    
    # map HPO terms to ancestors/superclasses
    df['ancs'] = df.HPO.apply(lambda x: map_ancs(x) )
    
    # drop HPO terms that could not be mapped - possibly a mistmatch between HPO versions
    percent = round((df.loc[~df.ancs.astype(bool)].shape[0]/df.loc[df.ancs.astype(bool)].shape[0]*100),2)
    if percent > 0:
        print('\n', percent, '% of HPO values could not be matched to ancestors! Do your HPO versions match between your patient data file and the HPO ontology database?\n')
    df = df.loc[df.ancs.astype(bool)] # drop all HPO terms that cannot be mapped
    
    # Take the exploded data and flatten it back to a single row per ID
    df = pd.concat([df.groupby('ID').ancs.sum(),
               df.groupby('ID').HPO.apply(lambda x: list(x))], axis = 1).reset_index()[['ID','HPO','ancs']]
    # Make sure HPO terms are included in their ancestors list
    df.loc[:, 'ancs'] = df.HPO + df.ancs
    # get unique HPO and ancestors in each row
    df.loc[:, 'ancs'] = df.ancs.apply(lambda x: list(np.unique(x)))
    df.loc[:, 'HPO'] = df.HPO.apply(lambda x: list(np.unique(x)))
    
    # check for obsolete terms in our dataframe
    if df.HPO.apply(obsolete_terms).any():
        print('\nObsolete HPO terms detected in the input dataframe!\n')
    
    
    # CALCULATING INFORMATION CONTENT
    
    # create a flattened list of all HPO term ancestors, including duplicates
    all_ancs =  df.ancs.apply(pd.Series).stack().reset_index(drop=True)
    # calculate information content for each HPO ancestor term (-log2(Freq.))
    IC = (-1*(np.log2(all_ancs.value_counts(normalize=True)))).to_dict()
    
    
    # CALCULATE RESNIK DISTANCE
    
    print("Performing pairwise Resnik semantic similarity comparison for", len(df.ID), 'input IDs. This may take some time.')
    
    # use list comprehension and our Resnik function to construct a dataframe with pairwise comparisons between cases
    distance = pd.DataFrame([[resnik(j, i) for i in df.HPO] for j in df.HPO])
    
    # set diagonal values to zero
    for i in distance.columns:
        distance.loc[i,i] = 0
    
    # set ID as row index (already present in column index)
    distance = distance.dropna().set_index(df.ID.values)
    
    # save output
    distance.to_csv(args.output, sep='\t', index=False)
    
    print("Analysis completed! Data is saved to", args.output)
    
    
