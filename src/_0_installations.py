#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install --upgrade pip')
get_ipython().system("pip install -r '../requirements.txt'")


# In[2]:


import chromadb, pymupdf
from sentence_transformers import SentenceTransformer

print("Chroma version:", chromadb.__version__)
print("PyMuPDF version:", pymupdf.__doc__)


# In[ ]:




