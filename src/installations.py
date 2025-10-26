#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install -r requirements.txt')


# In[3]:


import chromadb, fitz
from sentence_transformers import SentenceTransformer

print("Chroma version:", chromadb.__version__)
print("PyMuPDF version:", fitz.__doc__.split()[1])



# In[ ]:




