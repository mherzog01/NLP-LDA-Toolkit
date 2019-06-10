# -*- coding: utf-8 -*-
"""
Categorize data using LDA

Created on Tue Mar 12 16:31:07 2019

@author: MHerzo

Process
0.  Import module
1.  Create instance of class
2.  Based on needs, call methods to do the following
    - Load docPages
    - Load/create needed data
    - Load/create desired model
    
<See bottom of this module for details>

TODO Determine how to stabilize the category number -- if regen the model, want the category to stay the same
TODO Determine how to combine models, or categories derrived from many models

TODO Get all pages of DRAI, not just pages with matching values
TODO Data cleansing/preparation - lemmatizing/stemming, etc (https://towardsdatascience.com/the-complete-guide-for-topics-extraction-in-python-a6aaa6cedbbc and https://datascienceplus.com/topic-modeling-in-python-with-nltk-and-gensim/)
"""

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

#from sklearn.model_selection import train_test_split

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

import os


from IPython.utils import io

class CatDataLDA():
 
    # Class variables
    
    def __init__(self):
        #Constants
        self.NUM_TOPICS = 30
    
        self.train_dir=r'C:\Users\mherzo\Box Sync\Green Belt\Systems\Machine Learning\Training'
    
        #Other attributes
        self.docPages = pd.DataFrame()
        self.vectorizer = CountVectorizer(min_df=5, max_df=0.9, 
                                         stop_words='english', lowercase=True,
                                         token_pattern='[a-zA-Z][a-zA-Z]{2,}') 
                                         #token_pattern='[a-zA-Z-][a-zA-Z-]{2,}') #-- removed '-' because allowed for '----'
        self.data_vectorized = None
        self.data_train = pd.DataFrame()
        self.data_test = self.data_train.copy()
        self.lda_model = LatentDirichletAllocation()        
        self.lda_Z = self.lda_model.fit_transform([[1]])
        self.lda_Z_dominant = np.argmax(self.lda_Z,axis=1)
        self.feature_names = None
        self.topics = np.empty(0)
    
    
    # TODO - Handle this more elegantly -- shouldn't have to refer to self.model.components_ twice
    def print_topics(self,top_n=10):
        for idx, topic in enumerate(self.topics):
            self.print_topic_given_topic_and_idx(idx,topic,top_n)
    
    def print_topic(self,idx,top_n=10):
        topic_list = self.print_topic_given_topic_and_idx(idx,self.topics[idx],top_n)
        return topic_list

    def print_topic_given_topic_and_idx(self,idx, topic,top_n=10):
        print("Topic %d:" % (idx))
        if top_n is None:
            top_n = len(self.feature_names)
        topic_list = [(self.feature_names[i], topic[i].round(1))
                        for i in topic.argsort()[:-top_n - 1:-1]]
        print(topic_list)
        return topic_list

    #TODO Separate printing and getting values for the "print_doc_features*", "print_topic*", and other functions
    #TODO Remove duplicate code from "print_doc_features*"
    def print_doc_features(self,text):  
        feature_names = self.feature_names
        vector_array = self.vectorizer.transform([text]).toarray()[0]
        feature_df = pd.DataFrame(zip(feature_names,vector_array),columns=['FeatureName','Freq'])
        feature_df.sort_values('Freq',ascending=False,inplace=True)
        print(feature_df.iloc[:10])
        
    def print_doc_features_with_topic(self,text,topic_num):  
        feature_names = self.feature_names
        vector_array = self.vectorizer.transform([text]).toarray()[0]
        feature_df = pd.DataFrame(zip(feature_names,vector_array,self.topics[topic_num]),columns=['FeatureName','Freq','Topic Weight'])
        feature_df.sort_values('Freq',ascending=False,inplace=True)
        print(feature_df.iloc[:10])         
        
    def print_doc_features_idx(self,idx,topic):
        text = self.docPages['pageText'].loc[idx]
        self.print_doc_features_with_topic(text,topic)
        
    def print_model(self):
        print("LDA Model:")
        np.set_printoptions(suppress=True)
        print(self.lda_model.components_.round(2))
        self.print_topics()
        print("=" * 20)
        
    def eval_model_silent(self,idx,src=None):     
        if src is None:
            src=self.data_test
        text = src['pageText'].iloc[idx]
        x = self.lda_model.transform(self.vectorizer.transform([text]))[0]
        x_s = pd.Series(x)
        best_topics = x_s[(-1 * x_s).values.argsort()[:5]]
        #print(best_topics.round(1), x.sum())
        return best_topics
        
    def eval_model_from_text(self,text):
        #TODO create procedure to transform data?
        x = self.lda_model.transform(self.vectorizer.transform([text]))[0]
        x_s = pd.Series(x)
        best_topics = x_s[(-1 * x_s).values.argsort()[:5]]
        self.print_top_topics(best_topics)
        print("-" * 20)
        self.print_topic(best_topics.index[0])
        return best_topics
    
    #TODO merge eval_model and eval_model_from... -- just change source
    #TODL Add display of index to printout
    def eval_model_from_corpus_page(self,idx):   
        text = self.docPages['pageText'].iloc[idx]
        print("=" * 20)
        print('File name:',self.docPages['FileName'].iloc[idx])
        print('Page #:',self.docPages['pageNum'].iloc[idx])
        print("-" * 20)
        print(text)
        print("-" * 20)
        best_topics = self.eval_model_from_text(text)
        print("-" * 20)
        self.print_doc_features_idx(idx,best_topics.index[0])
        return best_topics
        
    def eval_model(self,idx,print_data=True):     
        text = self.data_test['pageText'].iloc[idx]
        print("=" * 20)
        if print_data:
            self.print_data(idx)
        else:
            print("Index:  " + str(idx))
            print('File name:',self.data_test['FileName'].iloc[idx])
            print('Page #:',self.data_test['pageNum'].iloc[idx])
            print("-" * 20)
        best_topics = self.eval_model_from_text(text)
        return best_topics
    
    #TODO Use decorator to output to file?
    #TODO give status based on time, not number of rows
    def eval_model_pages_to_file(self,idx_range,print_data=True,outfile=r'c:\tmp\data.txt'):
        self.logmsg01('Process begins')
        with io.capture_output() as captured:
            i=0
            for idx in idx_range:
                i+=1
                if i % 1000 == 0: self.logmsg01(f'Read {i} records')
                self.eval_model(idx,print_data)
        self.logmsg01('Outputting results')            
        with open(outfile, 'w') as outfile: 
            outfile.write(str(captured))
        self.logmsg01('Process ends')
        
    def print_data(self,idx,src=None):
        #TODO Assign/display doc page an index that does not change depending on the data set -- e.g. docPages vs data_test
        if src is None:
            src=self.data_test
        print("Index:  " + str(idx))
        print("File:  " + src.iloc[idx,0])
        print("Page: " + str(src.iloc[idx,1]))
        print("-" * 20)
        print(src.iloc[idx,2])
        print("=" * 20)
        
    def print_docs_in_topic(self,idx,max_docs=20):
        if max_docs is None:
            max_docs = float('inf')
        num_docs = 0
        print("=" * 20)
        self.print_topic(idx)
        print("=" * 20)
        for i in [i0 for i0,v in enumerate(self.lda_Z_dominant) if v == idx]:
            if not num_docs is None:
                if num_docs >= max_docs:
                    break
            num_docs = num_docs + 1
            print('File name:',self.data_test['FileName'].iloc[i])
            print('Page #: {0}.  Doc #:  {1}'.format(self.data_test['pageNum'].iloc[i],i))
            print(f'Likelihood:  {self.lda_Z[i,idx]}')
            print("-" * 20)
            print(self.data_test['pageText'].iloc[i])
            print("=" * 20)
        print('Num docs: %0s' % num_docs)
    
    def print_docs_in_topic_to_file(self,idx,max_docs=20,outfile=r'c:\tmp\data.txt'):
        self.logmsg01('Process begins')
        with io.capture_output() as captured:
            self.print_docs_in_topic(idx,max_docs)
        self.logmsg01('Outputting results')            
        with open(outfile, 'w') as outfile: 
            outfile.write(str(captured))
        self.logmsg01('Process ends')
    
    def summarize_docs_by_topic(self):
        n, bins, patches = plt.hist(self.lda_Z_dominant,bins=range(min(self.lda_Z_dominant),max(self.lda_Z_dominant)))
        z=zip(bins[:-1],n)
        lst=[(i,int(j)) for (i,j) in z if j > 0]
        print('Top Categories:')
        top_c = lst.copy()
        top_c.sort(key=lambda tup:-tup[1])
        print(top_c)
        print('Bins:')
        print(lst)
        
    def dump_lda_model(self,model_name,
                   topic_nums,
                   topic_name,
                   target_dir=None):
        if target_dir is None:
            target_dir=self.train_dir
        with open(os.path.join(target_dir,'lda_' + model_name + '.pickle'),'wb') as f:
            pickle.dump(self.lda_model,f)    
        with open(os.path.join(target_dir,'lda_' + model_name + '_Z.pickle'),'wb') as f:
            pickle.dump(self.lda_Z,f)    
        with open(os.path.join(target_dir,'lda_' + model_name + '_topic.txt'),'w') as f:
            f.write('Topic #' + '\t' + 'Topic_name\n')
            for topic_num in topic_nums:
                f.write(str(topic_num) + '\t' + topic_name + '\n')
        #with open(os.path.join(target_dir,'lda_' + model_name + '_pages.txt'),'w') as f:
        #    f.write('File name' + '\t' + 'Page' + '\t' + 'Topic #\n')
        #    for i in [i0 for i0,v in enumerate(self.lda_Z_dominant) if v == topic_num]:
        #        f.write(self.data_test['FileName'].iloc[i] + '\t' + str(self.data_test['pageNum'].iloc[i]) + '\t' + str(topic_num) + '\n')
    
    def load_lda_model(self,model_name):
        self.lda_model = self.load_data('lda_' + model_name)    
        self.lda_Z = self.load_data('lda_' + model_name + '_Z')    
    
    def dump_data(self,name_base,data_to_dump):
        with open(os.path.join(self.train_dir, name_base + ".pickle"),'wb') as f:
            pickle.dump(data_to_dump,f)
        self.logmsg01("Data dumped")
        
    def load_data(self,name_base):
        with open(os.path.join(self.train_dir, name_base + ".pickle"),'rb') as f:
            data_dumped = pickle.load(f)
        self.logmsg01(f"{name_base} loaded")
        return data_dumped
    
    
    def logmsg(self,pMsg):
        print(pMsg)
        
    def logmsg01(self,pMsg):
        self.logmsg(str(datetime.datetime.now()) + ":  " + pMsg)
        
    def find_docs_in_corpus(self,outpath=None,topic_num=None,max_docs=float('inf'),max_hits=float('inf')):
        self.logmsg01('Process begins')
        print(f"outpath={outpath}, len={len(outpath)}, None? {outpath==None}, !=''? {outpath!=''}.")
        if outpath != '':
            if outpath is None:
                outpath = os.path.join(self.train_dir,'docs_in_corpus.txt')
            self.logmsg(f'Output file={outpath}')
            if os.path.isfile(outpath):
                os.remove(outpath)
        num_hits = 0
        for i in range(len(self.docPages)):
            if i % 1000 == 0 and i != 0:
                self.logmsg01(f'Working on page {i}.  # hits={num_hits}')
            if i >= max_docs or num_hits >= max_hits:
                break
            best_topics = self.eval_model_silent(i,src=self.docPages)
            if topic_num is None:
                best_topic_prob = best_topics.iloc[0]
                best_topic = best_topics.index[0]
            else:
                # From https://stackoverflow.com/questions/9542738/python-find-in-list
                best_topic = next((cur_topic_num for cur_topic_num in best_topics.index if cur_topic_num == topic_num), None)
                if best_topic is not None:
                    best_topic_prob = best_topics.loc[topic_num]
            if best_topic is not None:  # and best_topic_prob > 0.8:
                num_hits += 1
                with io.capture_output() as captured:
                    print(f'Topic: {best_topic}.  Probability={round(best_topic_prob,2)}')
                    self.print_top_topics(best_topics)
                    #self.print_topic(best_topic)
                    self.print_data(i,src=self.docPages)
                if outpath != '':
                    with open(outpath, 'a') as outstream: 
                        outstream.write(str(captured))
        self.logmsg(f'Pages procesed={i}.  # hits={num_hits}')
        self.logmsg01('Process ends')
        
    def print_top_topics(self,topic_df):
        print("Top topics:")
        for i,t in topic_df.items():
            print(i,round(t,1))
        
        
    def get_docPages(self):
        frames = []
        
        def get_page_data(filenum):
            self.logmsg01(f'Working on file #{filenum}')
            with open(r"\\ussomgensvm00.allergan.com\lifecell\Depts\Tissue Services\TS Process Manager\ML\Training\docPages" + filenum + ".pickle",'rb') as f:
                df_tmp = pickle.load(f)
                frames.append(df_tmp)
        
        get_page_data("")
        get_page_data("2")
        get_page_data("3")
        get_page_data("4")
        get_page_data("5")
        
        self.logmsg01(f'Creating data frame')
        self.docPages = pd.concat(frames)
        self.logmsg01(f'Complete')
    
    def not_desired_byte(b):
        return b < 32 and b not in (9,10,13)
    
    #TODO:  Move this to the generation of docPages in ProcessTextfiles.py
    #From https://stackoverflow.com/questions/14661701/how-to-drop-a-list-of-rows-from-pandas-dataframe
    def clean_docPages(self):
        self.logmsg01('Begin')
        to_drop = []
        i = 0
        j = 0
        for idx in self.docPages.index:
            i += 1
            if i % 10000 == 0: 
                self.logmsg01(f'Working on index {i}')
            d_txt = self.docPages['pageText'].loc[idx]
            d = d_txt.encode('ansi')
            if any(self.not_desired_byte(b) for b in d):
                to_drop.append(i)
                j += 1
        self.logmsg01(f'Found rec={i}, errors={j}')
        self.docPages.drop(to_drop,inplace=True)
        self.logmsg01('Done')

    def create_model(self,n_components=None):
        self.logmsg01("Process begins")
        
        if n_components is None:
            n_components = self.NUM_TOPICS
        
        self.setup_model('create',n_components)
         
        self.logmsg01("Process complete")


    def setup_model(self,model_action,n_components):
        self.logmsg01("Fitting vectorizer")
        data = self.data_train['pageText'].tolist()
        self.data_vectorized = self.vectorizer.fit_transform(data)
        # Build a Latent Dirichlet Allocation Model
        if model_action == 'create':
            self.logmsg01("Creating model")
            self.lda_model = LatentDirichletAllocation(n_components=n_components, max_iter=10, learning_method='online')        
            self.lda_Z = self.lda_model.fit_transform(self.data_vectorized)
        # Else, assume model is already loaded
        self.logmsg01("Identifying a dominant category for each document")
        self.lda_Z_dominant = np.argmax(self.lda_Z,axis=1)
        self.feature_names = self.vectorizer.get_feature_names()
        self.topics = self.lda_model.components_
        self.logmsg01("Setup complete")
    



    """
    TODO For DRAI, use tokenizer, rather than str.contains
    TODO For DRAI, handle files which contain a DRAI and other data
    TODO For DRAI, use series of pages (e.g. 5 pages in a row), instead of individual pages containing words
    
    TODO Separate model from data - dump/load/export model along with data.  Separate this in CatDataLDA as well
    """
    def manage_structs(self,dataset_type='hd',data_action='load',model_action='load'):
        
        self.logmsg01('Process begins')
        
        if dataset_type == "drai":
            
            name_base='drai'
            if data_action == 'init':
                drai_pages=self.docPages[self.docPages["pageText"].str.contains("DONOR RISK ASSESSMENT INTERVIEW|UDRAI|uniform drai",case=False)]
                # Combine all pages in file which meet initial selection criteria
                # From https://stackoverflow.com/questions/17841149/pandas-groupby-how-to-get-a-union-of-strings
                #drai=drai_pages.groupby('FileName').apply(lambda x: pd.Series(dict(FileName = x['FileName'].iloc[0], pageNum = "%s" % ','.join(x['pageNum'].astype('str')), pageText = "%s" % '\n'.join(x['pageText']))))
                drai = drai_pages
                curdata = drai
                
            elif data_action == 'load':
                drai = self.load_data(name_base)
                curdata = drai
                
            if model_action == 'dump':
                # Assumes CatDataLDA.py has been initialized
                self.dump_lda_model('drai',[2,4],'drai')
        
            elif model_action == 'load':
                # Assumes CatDataLDA.py has been initialized
                self.load_lda_model('drai')
                
            elif model_action == 'create':
                self.create_model(n_components=10)
        
        elif dataset_type == "hd":
            
            name_base='hd'
        
            if data_action == 'init':
                hd=self.docPages[self.docPages["pageText"].str.contains("Hemodilution".upper(),case=False)]
                curdata = hd
                self.dump_data(name_base,hd)
                
            elif data_action == 'load':
                hd = self.load_data(name_base)
                curdata = hd
                
            if model_action == 'dump':
                # Assumes CatDataLDA.py has been initialized
                self.dump_lda_model('plasma_dilution_hemodilution',[5],'Hemodilution')
        
            elif model_action == 'load':
                # Assumes CatDataLDA.py has been initialized
                self.load_lda_model('plasma_dilution_hemodilution')
        
            elif model_action == 'create':
                self.create_model()

        if data_action in ['init','dump']:
            self.dump_data(name_base,curdata)
                
        #TODO data_action='load' and model_action='create' doesn't work because data_train isn't set
        #TODO Ensure data_train and data_test are used in the right properly
        #TODO HD doesn't seem to load properly -- need to regen data and model
        
        # For initial LDA modeling, categorize all documents in the input data set
    	# Testing will consist of manually inspecting results, and then running against a larger data set
        #hd_train, hd_test = train_test_split(hd, test_size=0.2)
        if data_action in ['init','load']:
            self.logmsg01("Creating data sets")
            self.data_train = curdata
            self.data_test = curdata
        
            if model_action not in ['create']:
                self.setup_model(model_action,None)

        self.logmsg01("Process complete")



if __name__ == "__main__":
    
    #init_action = 'init'
    #init_action = 'update_functions'
    
    #if init_action == 'init':
    #TODO The current approach of refreshing functions without clobbering data does not seem to work
    #elif init_action == 'update_functions':


    """
        # The following should be executed outside of this module:
        import cat_data_lda as cdl
        x=cdl.CatDataLDA()
        x.manage_structs()
        x.get_docPages()
        # ... commands against x ...
        #
        # To make code changes:
        from importlib import reload 
        reload(cdl)

        # Command to query model pages
        x.eval_model_pages_to_file([i for i,b in enumerate((x.data_test['pageText'].str.contains('\?')==False) & (x.data_test['pageText'].str.contains('PRE-RELEASE CHECKLIST|CASE NOTES|PROGRESS NOTES|TISSUE NARRATIVE NOTES|Audit Trail Report|ELECTRONIC SIGNATURES|PREVIOUS SIGNATURES|Electronically Signed By')==False)) if b])        

    """
    